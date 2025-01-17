//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include "openmp.h"
#include "quantities.h"

#include <deal.II/base/function_parser.h>
#include <deal.II/base/mpi.templates.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_tools.h>

#include <fstream>

DEAL_II_NAMESPACE_OPEN
template <int rank, int dim, typename Number>
bool operator<(const Tensor<rank, dim, Number> &left,
               const Tensor<rank, dim, Number> &right)
{
  return std::lexicographical_compare(
      left.begin_raw(), left.end_raw(), right.begin_raw(), right.end_raw());
}
DEAL_II_NAMESPACE_CLOSE

namespace ryujin
{
  using namespace dealii;

  namespace
  {
    template <typename T>
    const std::string &get_options_from_name(const T &manifolds,
                                             const std::string &name)
    {
      const auto it =
          std::find_if(manifolds.begin(),
                       manifolds.end(),
                       [&, name = std::cref(name)](const auto &element) {
                         return std::get<0>(element) == name.get();
                       });
      Assert(it != manifolds.end(), dealii::ExcInternalError());
      return std::get<2>(*it);
    }
  } // namespace


  template <typename Description, int dim, typename Number>
  Quantities<Description, dim, Number>::Quantities(
      const MPIEnsemble &mpi_ensemble,
      const OfflineData<dim, Number> &offline_data,
      const HyperbolicSystem &hyperbolic_system,
      const ParabolicSystem &parabolic_system,
      const std::string &subsection /*= "Quantities"*/)
      : ParameterAcceptor(subsection)
      , mpi_ensemble_(mpi_ensemble)
      , offline_data_(&offline_data)
      , hyperbolic_system_(&hyperbolic_system)
      , parabolic_system_(&parabolic_system)
      , base_name_("")
      , mesh_files_have_been_written_(false)
  {
    add_parameter("interior manifolds",
                  interior_manifolds_,
                  "List of level set functions describing interior manifolds. "
                  "The description is used to only output point values for "
                  "vertices belonging to a certain level set. "
                  "Format: '<name> : <level set formula> : <options> , [...] "
                  "(options: time_averaged, space_averaged, instantaneous)");

    add_parameter("boundary manifolds",
                  boundary_manifolds_,
                  "List of level set functions describing boundary. The "
                  "description is used to only output point values for "
                  "boundary vertices belonging to a certain level set. "
                  "Format: '<name> : <level set formula> : <options> , [...] "
                  "(options: time_averaged, space_averaged, instantaneous)");

    clear_temporal_statistics_on_writeout_ = true;
    add_parameter("clear statistics on writeout",
                  clear_temporal_statistics_on_writeout_,
                  "If set to true then all temporal statistics (for "
                  "\"time_averaged\" quantities) accumulated so far are reset "
                  "each time a writeout of quantities is performed");
  }


  template <typename Description, int dim, typename Number>
  void Quantities<Description, dim, Number>::prepare(const std::string &name)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Quantities<dim, Number>::prepare()" << std::endl;
#endif

    base_name_ = name;

    /* Force to write to a new time series file: */
    time_series_cycle_.reset();

    const unsigned int n_owned = offline_data_->n_locally_owned();
    const auto &sparsity_simd = offline_data_->sparsity_pattern_simd();

    /*
     * Create interior maps and allocate statistics.
     *
     * We have to loop over the cells and populate the std::map interior_maps_.
     */

    interior_maps_.clear();
    std::transform(
        interior_manifolds_.begin(),
        interior_manifolds_.end(),
        std::inserter(interior_maps_, interior_maps_.end()),
        [this, n_owned, &sparsity_simd](auto it) {
          const auto &[name, expression, option] = it;
          FunctionParser<dim> level_set_function(expression);

          std::vector<interior_point> map;
          std::map<int, interior_point> preliminary_map;

          const auto &discretization = offline_data_->discretization();
          const auto &dof_handler = offline_data_->dof_handler();

          const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;

          const auto support_points =
              dof_handler.get_fe().get_unit_support_points();

          std::vector<dealii::types::global_dof_index> local_dof_indices(
              dofs_per_cell);

          /* Loop over cells */
          for (auto cell : dof_handler.active_cell_iterators()) {

            /* skip if not locally owned */
            if (!cell->is_locally_owned())
              continue;

            cell->get_active_or_mg_dof_indices(local_dof_indices);

            for (unsigned int j = 0; j < dofs_per_cell; ++j) {

              Point<dim> position =
                  discretization.mapping().transform_unit_to_real_cell(
                      cell, support_points[j]);

              /*
               * Insert index, interior mass value and position into
               * a preliminary map if we satisfy level set condition.
               */

              if (std::abs(level_set_function.value(position)) > 1.e-12)
                continue;

              const auto global_index = local_dof_indices[j];
              const auto index =
                  offline_data_->scalar_partitioner()->global_to_local(
                      global_index);

              /* Skip constrained degrees of freedom: */
              const unsigned int row_length = sparsity_simd.row_length(index);
              if (row_length == 1)
                continue;

              if (index >= n_owned)
                continue;

              const Number interior_mass =
                  offline_data_->lumped_mass_matrix().local_element(index);
              // FIXME: change to std::set
              preliminary_map[index] = {index, interior_mass, position};
            }
          }

          /*
           * Now we populate the std::vector(interior_point) object called map.
           */
          // FIXME: use std::copy
          for (const auto &[index, tuple] : preliminary_map) {
            map.push_back(tuple);
          }

          return std::make_pair(name, map);
        });

    /*
     * Create boundary maps and allocate statistics vector:
     *
     * We want to loop over the boundary_map() once and populate the map
     * object boundary_maps_. We have to create a vector of
     * boundary_manifolds.size() that holds a std::vector<boundary_point>
     * for each map entry.
     */

    boundary_maps_.clear();
    std::transform(
        boundary_manifolds_.begin(),
        boundary_manifolds_.end(),
        std::inserter(boundary_maps_, boundary_maps_.end()),
        [this, n_owned](auto it) {
          const auto &[name, expression, option] = it;
          FunctionParser<dim> level_set_function(expression);

          std::vector<boundary_point> map;

          for (const auto &entry : offline_data_->boundary_map()) {
            // [i, normal, normal_mass, boundary_mass, id, position] = entry
            const auto &i = std::get<0>(entry);

            /* skip nonlocal */
            if (i >= n_owned)
              continue;

            /* skip constrained */
            if (offline_data_->affine_constraints().is_constrained(
                    offline_data_->scalar_partitioner()->local_to_global(i)))
              continue;

            const auto &position = std::get<5>(entry);
            if (std::abs(level_set_function.value(position)) < 1.e-12)
              map.push_back(entry);
          }
          return std::make_pair(name, map);
        });

    /* Clear statistics: */
    clear_statistics();

    /* Make sure we output new mesh files: */
    mesh_files_have_been_written_ = false;

    /* Prepare header string: */
    const auto &names = View::primitive_component_names;
    header_ = std::accumulate(
                  std::begin(names),
                  std::end(names),
                  std::string(),
                  [](const std::string &description, const std::string &name) {
                    return description.empty()
                               ? (std::string("primitive state (") + name)
                               : (description + ", " + name);
                  }) +
              ")\t and 2nd moments\n";
  }


  template <typename Description, int dim, typename Number>
  void
  Quantities<Description, dim, Number>::write_mesh_files(unsigned int cycle)
  {
    /*
     * Output interior maps:
     */

    for (const auto &[name, interior_map] : interior_maps_) {
      /* Skip outputting the boundary map for spatial averages. */
      const auto &options = get_options_from_name(interior_manifolds_, name);
      if (options.find("instantaneous") == std::string::npos &&
          options.find("time_averaged") == std::string::npos)
        continue;

      /*
       * FIXME: This currently distributes boundary maps to all MPI ranks.
       * This is unnecessarily wasteful. Ideally, we should do MPI IO with
       * only MPI ranks participating who actually have boundary values.
       */

      const auto received = Utilities::MPI::gather(
          mpi_ensemble_.ensemble_communicator(), interior_map);

      if (Utilities::MPI::this_mpi_process(
              mpi_ensemble_.ensemble_communicator()) == 0) {

        std::ofstream output(base_name_ + "-" + name + "-R" +
                             Utilities::to_string(cycle, 4) + "-points.dat");

        output << std::scientific << std::setprecision(14);

        output << "#\n# position\tinterior mass\n";

        unsigned int rank = 0;
        for (const auto &entries : received) {
          output << "# rank " << rank++ << "\n";
          for (const auto &entry : entries) {
            const auto &[index, mass_i, x_i] = entry;
            output << x_i << "\t" << mass_i << "\n";
          } /*entry*/
        }   /*entries*/

        output << std::flush;
      }
    }

    /*
     * Output boundary maps:
     */

    for (const auto &[name, boundary_map] : boundary_maps_) {
      /* Skip outputting the boundary map for spatial averages. */
      const auto &options = get_options_from_name(boundary_manifolds_, name);
      if (options.find("instantaneous") == std::string::npos &&
          options.find("time_averaged") == std::string::npos)
        continue;

      /*
       * FIXME: This currently distributes boundary maps to all MPI ranks.
       * This is unnecessarily wasteful. Ideally, we should do MPI IO with
       * only MPI ranks participating who actually have boundary values.
       */

      const auto received = Utilities::MPI::gather(
          mpi_ensemble_.ensemble_communicator(), boundary_map);

      if (Utilities::MPI::this_mpi_process(
              mpi_ensemble_.ensemble_communicator()) == 0) {

        std::ofstream output(base_name_ + "-" + name + "-R" +
                             Utilities::to_string(cycle, 4) + "-points.dat");

        output << std::scientific << std::setprecision(14);

        output << "#\n# position\tnormal\tnormal mass\tboundary mass\n";

        unsigned int rank = 0;
        for (const auto &entries : received) {
          output << "# rank " << rank++ << "\n";
          for (const auto &entry : entries) {
            const auto &[index, n_i, nm_i, bm_i, id, x_i] = entry;
            output << x_i << "\t" << n_i << "\t" << nm_i << "\t" << bm_i
                   << "\n";
          } /*entry*/
        }   /*entries*/

        output << std::flush;
      }
    }
  }


  template <typename Description, int dim, typename Number>
  void Quantities<Description, dim, Number>::clear_statistics()
  {
    const auto reset = [](const auto &manifold_map, auto &statistics_map) {
      for (const auto &[name, data_map] : manifold_map) {
        const auto n_entries = data_map.size();
        auto &[val_old, val_new, val_sum, t_old, t_new, t_sum] =
            statistics_map[name];
        val_old.resize(n_entries);
        val_new.resize(n_entries);
        val_sum.resize(n_entries);
        t_old = t_new = t_sum = 0.;
      }
    };

    /* Clear statistics and time series: */

    interior_statistics_.clear();
    reset(interior_maps_, interior_statistics_);
    interior_time_series_.clear();

    boundary_statistics_.clear();
    reset(boundary_maps_, boundary_statistics_);
    boundary_time_series_.clear();
  }


  template <typename Description, int dim, typename Number>
  template <typename point_type, typename value_type>
  value_type Quantities<Description, dim, Number>::internal_accumulate(
      const StateVector &state_vector,
      const std::vector<point_type> &points_vector,
      std::vector<value_type> &val_new)
  {
    const auto &U = std::get<0>(state_vector);

    value_type spatial_average;
    Number mass_sum = Number(0.);

    std::transform(
        points_vector.begin(),
        points_vector.end(),
        val_new.begin(),
        [&](auto point) -> value_type {
          const auto i = std::get<0>(point);
          /*
           * Small trick to get the correct index for retrieving the
           * boundary mass.
           */
          constexpr auto index =
              std::is_same_v<point_type, interior_point> ? 1 : 3;
          const auto mass_i = std::get<index>(point);

          const auto U_i = U.get_tensor(i);
          const auto view = hyperbolic_system_->template view<dim, Number>();
          const auto primitive_state = view.to_primitive_state(U_i);

          value_type result;
          std::get<0>(result) = primitive_state;
          /* Compute second moments of the primitive state: */
          std::get<1>(result) = schur_product(primitive_state, primitive_state);

          mass_sum += mass_i;
          std::get<0>(spatial_average) += mass_i * std::get<0>(result);
          std::get<1>(spatial_average) += mass_i * std::get<1>(result);

          return result;
        });

    /* synchronize MPI ranks (MPI Barrier): */

    mass_sum =
        Utilities::MPI::sum(mass_sum, mpi_ensemble_.ensemble_communicator());

    std::get<0>(spatial_average) = Utilities::MPI::sum(
        std::get<0>(spatial_average), mpi_ensemble_.ensemble_communicator());
    std::get<1>(spatial_average) = Utilities::MPI::sum(
        std::get<1>(spatial_average), mpi_ensemble_.ensemble_communicator());

    /* take average: */

    std::get<0>(spatial_average) /= mass_sum;
    std::get<1>(spatial_average) /= mass_sum;

    return spatial_average;
  }


  template <typename Description, int dim, typename Number>
  template <typename value_type>
  void Quantities<Description, dim, Number>::internal_write_out(
      const std::string &file_name,
      const std::string &time_stamp,
      const std::vector<value_type> &values,
      const Number scale)
  {
    /*
     * FIXME: This currently distributes interior maps to all MPI ranks.
     * This is unnecessarily wasteful. Ideally, we should do MPI IO with
     * only MPI ranks participating who actually have interior values.
     */

    const auto received =
        Utilities::MPI::gather(mpi_ensemble_.ensemble_communicator(), values);

    if (Utilities::MPI::this_mpi_process(
            mpi_ensemble_.ensemble_communicator()) == 0) {

      std::ofstream output(file_name);
      output << std::scientific << std::setprecision(14);
      output << time_stamp << "# " << header_;

      unsigned int rank = 0;
      for (const auto &entries : received) {
        output << "# rank " << rank++ << "\n";
        for (const auto &entry : entries) {
          const auto &[state, state_square] = entry;
          output << scale * state << "\t" << scale * state_square << "\n";
        } /*entry*/
      }   /*entries*/

      output << std::flush;
    }
  }


  template <typename Description, int dim, typename Number>
  template <typename value_type>
  void Quantities<Description, dim, Number>::internal_write_out_time_series(
      const std::string &file_name,
      const std::vector<std::tuple<Number, value_type>> &values,
      bool append)
  {
    if (Utilities::MPI::this_mpi_process(
            mpi_ensemble_.ensemble_communicator()) == 0) {
      std::ofstream output;
      output << std::scientific << std::setprecision(14);

      if (append) {
        output.open(file_name, std::ofstream::out | std::ofstream::app);
      } else {
        output.open(file_name, std::ofstream::out | std::ofstream::trunc);
        output << "# time t\t" << header_;
      }

      for (const auto &entry : values) {
        const auto t = std::get<0>(entry);
        const auto &[state, state_square] = std::get<1>(entry);

        output << t << "\t" << state << "\t" << state_square << "\n";
      }

      output << std::flush;
      output.close();
    }
  }


  template <typename Description, int dim, typename Number>
  void Quantities<Description, dim, Number>::accumulate(
      const StateVector &state_vector, const Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Quantities<dim, Number>::accumulate()" << std::endl;
#endif

    const auto accumulate = [&](const auto &point_maps,
                                const auto &manifolds,
                                auto &statistics,
                                auto &time_series) {
      for (const auto &[name, point_map] : point_maps) {

        /* Find the correct option string in manifolds */
        const auto &options = get_options_from_name(manifolds, name);

        /* skip if we don't average in space or time: */
        if (options.find("time_averaged") == std::string::npos &&
            options.find("space_averaged") == std::string::npos)
          continue;

        auto &[val_old, val_new, val_sum, t_old, t_new, t_sum] =
            statistics[name];

        std::swap(t_old, t_new);
        std::swap(val_old, val_new);

        /* accumulate new values */

        const auto spatial_average =
            internal_accumulate(state_vector, point_map, val_new);

        /* Average in time with trapezoidal rule: */

        if (RYUJIN_UNLIKELY(t_old == Number(0.) && t_new == Number(0.))) {
          /* We have not accumulated any statistics yet: */
          t_old = t - 1.;
          t_new = t;

        } else {

          t_new = t;
          const Number tau = t_new - t_old;

          for (std::size_t i = 0; i < val_sum.size(); ++i) {
            std::get<0>(val_sum[i]) += 0.5 * tau * std::get<0>(val_old[i]);
            std::get<0>(val_sum[i]) += 0.5 * tau * std::get<0>(val_new[i]);
            std::get<1>(val_sum[i]) += 0.5 * tau * std::get<1>(val_old[i]);
            std::get<1>(val_sum[i]) += 0.5 * tau * std::get<1>(val_new[i]);
          }
          t_sum += tau;
        }

        /* Record average in space: */
        time_series[name].push_back({t, spatial_average});
      }
    };

    accumulate(interior_maps_,
               interior_manifolds_,
               interior_statistics_,
               interior_time_series_);

    accumulate(boundary_maps_,
               boundary_manifolds_,
               boundary_statistics_,
               boundary_time_series_);
  }


  template <typename Description, int dim, typename Number>
  void Quantities<Description, dim, Number>::write_out(
      const StateVector &state_vector, const Number t, unsigned int cycle)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "Quantities<dim, Number>::write_out()" << std::endl;
#endif

    /*
     * First, write out mesh files if this hasn't happened yet.
     */
    if (!mesh_files_have_been_written_) {
      write_mesh_files(cycle);
      mesh_files_have_been_written_ = true;
    }

    /*
     * Next write out instantaneous and time_averaged maps, and flush the
     * space_averaged values to the corresponding log files:
     */

    const auto write_out = [&](const auto &point_maps,
                               const auto &manifolds,
                               auto &statistics,
                               auto &time_series) {
      for (const auto &[name, point_map] : point_maps) {

        /* Find the correct option string in manifolds */
        const auto &options = get_options_from_name(manifolds, name);

        const auto prefix =
            base_name_ + "-" + name + "-R" + Utilities::to_string(cycle, 4);

        /*
         * Compute and output instantaneous field:
         */

        if (options.find("instantaneous") != std::string::npos) {

          const std::string file_name = prefix + "-instantaneous.dat";

          auto &[val_old, val_new, val_sum, t_old, t_new, t_sum] =
              statistics[name];

          std::stringstream time_stamp;
          time_stamp << std::scientific << std::setprecision(14);
          time_stamp << "# at t = " << t << std::endl;

          /* We have not computed any updated statistics yet: */

          if (options.find("time_averaged") == std::string::npos &&
              options.find("space_averaged") == std::string::npos)
            internal_accumulate(state_vector, point_map, val_new);
          else
            AssertThrow(t_new == t, dealii::ExcInternalError());

          internal_write_out(file_name, time_stamp.str(), val_new, Number(1.));
        }

        /*
         * Output time averaged field:
         */

        if (options.find("time_averaged") != std::string::npos) {

          const std::string file_name = prefix + "-time_averaged.dat";

          auto &[val_old, val_new, val_sum, t_old, t_new, t_sum] =
              statistics[name];

          /* Check whether we have accumulated any statistics yet: */
          if (t_sum != Number(0.)) {
            std::stringstream time_stamp;
            time_stamp << std::scientific << std::setprecision(14);
            time_stamp << "# averaged from t = " << t_new - t_sum
                       << " to t = " << t_new << std::endl;

            internal_write_out(
                file_name, time_stamp.str(), val_sum, Number(1.) / t_sum);
          }
        }

        /*
         * Output space averaged field:
         */

        if (options.find("space_averaged") != std::string::npos) {
          bool append = true;
          if (!time_series_cycle_.has_value()) {
            time_series_cycle_ = cycle;
            append = false;
          }

          const auto file_name =
              base_name_ + "-" + name + "-R" +
              Utilities::to_string(time_series_cycle_.value(), 4) +
              "-space_averaged_time_series.dat";

          auto &series = time_series[name];
          internal_write_out_time_series(file_name, series, /*append*/ append);
          series.clear();
        }
      }
    };

    write_out(interior_maps_,
              interior_manifolds_,
              interior_statistics_,
              interior_time_series_);

    write_out(boundary_maps_,
              boundary_manifolds_,
              boundary_statistics_,
              boundary_time_series_);

    if (clear_temporal_statistics_on_writeout_)
      clear_statistics();
  }

} /* namespace ryujin */
