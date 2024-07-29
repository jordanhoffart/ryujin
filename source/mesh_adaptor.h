//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
//

#pragma once

#include "mpi_ensemble.h"
#include "offline_data.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/smartpointer.h>

#include <random>

namespace ryujin
{
  /**
   * Controls the spatial mesh adaptation strategy.
   *
   * @ingroup Mesh
   */
  enum class AdaptationStrategy {
    /**
     * Perform a uniform global refinement.
     */
    global_refinement,

    /**
     * Perform random refinement and coarsening with a deterministic
     * Mersenne Twister and a chosen seed. This refinement strategy is only
     * useful for debugging and testing.
     */
    random_adaptation,

    /**
     * Perform local refinement and coarsening based on Kelly error estimator.
     */
    kelly_estimator,
  };

  /**
   * Controls the marking strategy used for mesh adaptation. This
   * configuration option is ignored for the uniform global refinement
   * strategy.
   *
   * @ingroup Mesh
   */
  enum class MarkingStrategy {
    /**
     * Refine and coarsen a configurable selected percentage of cells.
     */
    fixed_number,
  };

  /**
   * Controls the time point selection strategy.
   *
   * @ingroup Mesh
   */
  enum class TimePointSelectionStrategy {
    /**
     * Perform a mesh adaptation cycle at preselected fixed time points.
     */
    fixed_adaptation_time_points,

    /**
     * Perform a mesh adaptation cycle at every nth simulation cycle.
     */
    simulation_cycle_based,
  };
} // namespace ryujin

#ifndef DOXYGEN
DECLARE_ENUM(
    ryujin::AdaptationStrategy,
    LIST({ryujin::AdaptationStrategy::global_refinement, "global refinement"},
         {ryujin::AdaptationStrategy::random_adaptation, "random adaptation"},
         {ryujin::AdaptationStrategy::kelly_estimator, "kelly estimator"}, ));

DECLARE_ENUM(ryujin::MarkingStrategy,
             LIST({ryujin::MarkingStrategy::fixed_number, "fixed number"}, ));

DECLARE_ENUM(
    ryujin::TimePointSelectionStrategy,
    LIST({ryujin::TimePointSelectionStrategy::fixed_adaptation_time_points,
          "fixed adaptation time points"},
         {ryujin::TimePointSelectionStrategy::simulation_cycle_based,
          "simulation cycle based"}, ));
#endif

namespace ryujin
{
  /**
   * The MeshAdaptor class is responsible for performing global or local
   * mesh adaptation.
   *
   * @ingroup Mesh
   */
  template <typename Description, int dim, typename Number = double>
  class MeshAdaptor final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @name Typedefs and constexpr constants
     */
    //@{

    using HyperbolicSystem = typename Description::HyperbolicSystem;
    using ParabolicSystem = typename Description::ParabolicSystem;

    using View =
        typename Description::template HyperbolicSystemView<dim, Number>;

    static constexpr auto problem_dimension = View::problem_dimension;

    using StateVector = typename View::StateVector;

    using ScalarVector = Vectors::ScalarVector<Number>;

    //@}
    /**
     * @name Constructor and setup
     */
    //@{

    /**
     * Constructor.
     */
    MeshAdaptor(const MPIEnsemble &mpi_ensemble,
                const OfflineData<dim, Number> &offline_data,
                const HyperbolicSystem &hyperbolic_system,
                const ParabolicSystem &parabolic_system,
                const ScalarVector &alpha,
                const std::string &subsection = "/MeshAdaptor");

    /**
     * Prepare temporary storage and clean up internal data for the
     * analyze() facility.
     */
    void prepare(const Number t);

    /**
     * Analyze the given StateVector with the configured adaptation strategy
     * and decide whether a mesh adaptation cycle should be performed.
     */
    void analyze(const StateVector &state_vector,
                 const Number t,
                 unsigned int cycle);

    /**
     * A boolean indicating whether we should perform a mesh adapation step
     * in the current cycle. The analyze() method will set this boolean to
     * true whenever the selected adaptation strategy advices to perform an
     * adaptation cycle.
     */
    ACCESSOR_READ_ONLY(need_mesh_adaptation)

    /**
     * Mark cells for coarsening and refinement with the configured marking
     * strategy.
     */
    void mark_cells_for_coarsening_and_refinement(
        dealii::Triangulation<dim> &triangulation) const;

  private:
    /**
     * @name Run time options
     */
    //@{

    AdaptationStrategy adaptation_strategy_;
    std::uint_fast64_t random_adaptation_mersenne_twister_seed_;

    MarkingStrategy marking_strategy_;
    double fixed_number_refinement_fraction_;
    double fixed_number_coarsening_fraction_;

    TimePointSelectionStrategy time_point_selection_strategy_;
    std::vector<Number> adaptation_time_points_;
    unsigned int adaptation_simulation_cycle_;

    std::vector<std::string> kelly_options_;

    //@}
    /**
     * @name Internal fields and methods
     */
    //@{

    const MPIEnsemble &mpi_ensemble_;

    dealii::SmartPointer<const OfflineData<dim, Number>> offline_data_;
    dealii::SmartPointer<const HyperbolicSystem> hyperbolic_system_;
    dealii::SmartPointer<const ParabolicSystem> parabolic_system_;

    bool need_mesh_adaptation_;

    /* random adaptation: */

    mutable std::mt19937_64 mersenne_twister_;

    /* kelly estimator: */

    const ScalarVector &alpha_;

    std::vector<ScalarVector> kelly_quantities_;
    std::vector<
        std::tuple<std::string /*name*/,
                   std::function<void(ScalarVector & /*result*/,
                                      const StateVector & /*state_vector*/)>>>
        quantities_mapping_;
    //@}
  };

} // namespace ryujin
