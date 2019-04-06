#ifndef TIMELOOP_TEMPLATE_H
#define TIMELOOP_TEMPLATE_H

#include "timeloop.h"

#include <helper.h>
#include <indicator.h>
#include <limiter.h>
#include <riemann_solver.h>

#include <deal.II/base/logstream.h>
#include <deal.II/base/revision.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <boost/core/demangle.hpp>

#include <fstream>
#include <iomanip>


using namespace dealii;
using namespace grendel;


namespace
{
  /**
   * A helper function to print formatted section headings.
   */
  void print_head(std::string header, std::string secondary = "")
  {
    const auto header_size = header.size();
    const auto secondary_size = secondary.size();

    deallog << std::endl;
    deallog << "    ####################################################"
            << std::endl;
    deallog << "    #########                                  #########"
            << std::endl;
    deallog << "    #########"                          //
            << std::string((34 - header_size) / 2, ' ') //
            << header                                   //
            << std::string((35 - header_size) / 2, ' ') //
            << "#########"                              //
            << std::endl;
    deallog << "    #########"                             //
            << std::string((34 - secondary_size) / 2, ' ') //
            << secondary                                   //
            << std::string((35 - secondary_size) / 2, ' ') //
            << "#########"                                 //
            << std::endl;
    deallog << "    #########                                  #########"
            << std::endl;
    deallog << "    ####################################################"
            << std::endl;
    deallog << std::endl;
  }
} // namespace


namespace ryujin
{
  template <int dim>
  TimeLoop<dim>::TimeLoop(const MPI_Comm &mpi_comm)
      : ParameterAcceptor("A - TimeLoop")
      , mpi_communicator(mpi_comm)
      , computing_timer(mpi_communicator,
                        timer_output,
                        TimerOutput::never,
                        TimerOutput::cpu_times)
      , discretization(mpi_communicator, computing_timer, "B - Discretization")
      , offline_data(mpi_communicator,
                     computing_timer,
                     discretization,
                     "C - OfflineData")
      , initial_values("D - InitialValues")
      , time_step(mpi_communicator,
                  computing_timer,
                  offline_data,
                  initial_values,
                  "E - TimeStep")
      , schlieren_postprocessor(mpi_communicator,
                                computing_timer,
                                offline_data,
                                "F - SchlierenPostprocessor")
  {
    base_name = "test";
    add_parameter("basename", base_name, "Base name for all output files");

    t_final = 4.;
    add_parameter("final time", t_final, "Final time");

    output_granularity = 0.02;
    add_parameter(
        "output granularity", output_granularity, "time interval for output");

    enable_detailed_output = true;
    add_parameter("enable detailed output",
                  enable_detailed_output,
                  "Flag to control detailed output to deallog");

    enable_compute_error = false;
    add_parameter(
        "enable compute error",
        enable_compute_error,
        "Flag to control whether we compute the Linfty Linf_norm of the "
        "difference to an analytic solution. Implemented only for "
        "certain initial state configurations.");
  }


  template <int dim>
  void TimeLoop<dim>::run()
  {
    /*
     * Initialize deallog:
     */

    initialize();

    deallog << "TimeLoop<dim>::run()" << std::endl;

    /*
     * Create distributed triangulation and output the triangulation to inp
     * files:
     */

    print_head("create triangulation");
    discretization.prepare();

    {
      deallog << "        output triangulation" << std::endl;
      std::ofstream output(
          base_name + "-triangulation-p" +
          std::to_string(Utilities::MPI::this_mpi_process(mpi_communicator)) +
          ".inp");
      GridOut().write_ucd(discretization.triangulation(), output);
    }

    /*
     * Prepare offline data:
     */

    print_head("compute offline data");
    offline_data.prepare();

    print_head("set up time step");
    time_step.prepare();
    schlieren_postprocessor.prepare();

    /*
     * Interpolate initial values:
     */

    print_head("interpolate initial values");

    auto U = interpolate_initial_values();
    double t = 0.;

    output(U, base_name + "-solution", t, 0);
    if (enable_compute_error) {
      output(U, base_name + "-analytic_solution", t, 0);
    }

    print_head("enter main loop");

    /* Disable deallog output: */
    if (!enable_detailed_output)
      deallog.push("SILENCE!");

    /*
     * Loop:
     */

    unsigned int output_cycle = 1;
    for (unsigned int cycle = 1; t < t_final; ++cycle) {

      std::ostringstream head;
      head << "Cycle  " << Utilities::int_to_string(cycle, 6)         //
           << "  ("                                                   //
           << std::fixed << std::setprecision(1) << t / t_final * 100 //
           << "%)";
      std::ostringstream secondary;
      secondary << "at time t = " << std::setprecision(8) << std::fixed << t;

      print_head(head.str(), secondary.str());

      /* Do a time step: */
      const auto tau = time_step.step(U);
      t += tau;

      if (t > output_cycle * output_granularity) {
        if (!enable_detailed_output) {
          deallog.pop();
          print_head(head.str(), secondary.str());
        }

        output(U, base_name + "-solution", t, output_cycle++);
        if (enable_compute_error) {
          const auto analytic = interpolate_initial_values(t);
          output(analytic, base_name + "-analytic_solution", t, output_cycle);
        }

        if (!enable_detailed_output)
          deallog.push("SILENCE!");
      }
    } /* end of loop */

    /* Wait for output thread: */
    if (output_thread.joinable())
      output_thread.join();

    /* Reenable deallog output: */
    if (!enable_detailed_output)
      deallog.pop();

    /* Output final error: */
    if (enable_compute_error)
      compute_error(U, t);

    computing_timer.print_summary();
    deallog << timer_output.str() << std::endl;

    /* Detach deallog: */
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
      deallog.pop();
      deallog.detach();
    }
  }


  /**
   * Set up deallog output, read in parameters and initialize all objects.
   */
  template <int dim>
  void TimeLoop<dim>::initialize()
  {
    /* Read in parameters and initialize all objects: */

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {

      deallog.pop();

      deallog << "[Init] Initiating Flux Capacitor... [ OK ]" << std::endl;
      deallog << "[Init] Bringing Warp Core online... [ OK ]" << std::endl;

      deallog << "[Init] Reading parameters and allocating objects... "
              << std::flush;

      ParameterAcceptor::initialize("ryujin.prm");

      deallog << "[ OK ]" << std::endl;

    } else {

      ParameterAcceptor::initialize("ryujin.prm");
      return;
    }

    /* Print out parameters to a prm file: */

    std::ofstream output(base_name + "-parameter.prm");
    ParameterAcceptor::prm.print_parameters(output, ParameterHandler::Text);

    /* Prepare and attach logfile: */

    filestream.reset(new std::ofstream(base_name + "-deallog.log"));
    deallog.attach(*filestream);

    /* Output commit and library informations: */

    /* clang-format off */
    deallog.depth_console(4);
    deallog << "###" << std::endl;
    deallog << "#" << std::endl;
    deallog << "# deal.II version " << std::setw(8) << DEAL_II_PACKAGE_VERSION
            << "  -  " << DEAL_II_GIT_REVISION << std::endl;
    deallog << "# ryujin  version " << std::setw(8) << RYUJIN_VERSION
            << "  -  " << RYUJIN_GIT_REVISION << std::endl;
    deallog << "#" << std::endl;
    deallog << "###" << std::endl;

    /*
     * Print compile time parameters:
     */

    deallog << "Compile time parameters:" << std::endl;

    deallog << "Indicator<dim>::indicators_ == ";
    switch (Indicator<dim>::indicator_) {
    case Indicator<dim>::Indicators::zero:
      deallog << "Indicator<dim>::Indicators::zero" << std::endl;
      break;
    case Indicator<dim>::Indicators::one:
      deallog << "Indicator<dim>::Indicators::one" << std::endl;
      break;
    case Indicator<dim>::Indicators::entropy_viscosity_commutator:
      deallog << "Indicator<dim>::Indicators::entropy_viscosity_commutator" << std::endl;
      break;
    case Indicator<dim>::Indicators::smoothness_indicator:
      deallog << "Indicator<dim>::Indicators::smoothness_indicator" << std::endl;
    }

    deallog << "Indicator<dim>::smoothness_indicator_ == ";
    switch (Indicator<dim>::smoothness_indicator_) {
    case Indicator<dim>::SmoothnessIndicators::rho:
      deallog << "Indicator<dim>::SmoothnessIndicators::rho" << std::endl;
      break;
    case Indicator<dim>::SmoothnessIndicators::internal_energy:
      deallog << "Indicator<dim>::SmoothnessIndicators::internal_energy" << std::endl;
      break;
    case Indicator<dim>::SmoothnessIndicators::pressure:
      deallog << "Indicator<dim>::SmoothnessIndicators::pressure" << std::endl;
    }

    deallog << "Indicator<dim>::smoothness_indicator_alpha_0_ == "
            << Indicator<dim>::smoothness_indicator_alpha_0_ << std::endl;

    deallog << "Indicator<dim>::smoothness_indicator_power_ == "
            << Indicator<dim>::smoothness_indicator_power_ << std::endl;

    deallog << "Limiter<dim>::limiter_ == ";
    switch (Limiter<dim>::limiter_) {
    case Limiter<dim>::Limiters::none:
      deallog << "Limiter<dim>::Limiters::none" << std::endl;
      break;
    case Limiter<dim>::Limiters::rho:
      deallog << "Limiter<dim>::Limiters::rho" << std::endl;
      break;
    case Limiter<dim>::Limiters::internal_energy:
      deallog << "Limiter<dim>::Limiters::internal_energy" << std::endl;
      break;
    case Limiter<dim>::Limiters::specific_entropy:
      deallog << "Limiter<dim>::Limiters::specific_entropy" << std::endl;
    }

    deallog << "Limiter<dim>::relaxation_order_ == "
            << Limiter<dim>::relaxation_order_ << std::endl;

    deallog << "Limiter<dim>::line_search_eps_ == "
            << Limiter<dim>::line_search_eps_ << std::endl;

    deallog << "Limiter<dim>::line_search_max_iter_ == "
            << Limiter<dim>::line_search_max_iter_ << std::endl;

    deallog << "RiemannSolver<dim>::newton_eps_ == "
            <<  RiemannSolver<dim>::newton_eps_ << std::endl;

    deallog << "RiemannSolver<dim>::newton_max_iter_ == "
            <<  RiemannSolver<dim>::newton_eps_ << std::endl;

    deallog << "TimeStep<dim>::order_ == ";
    switch (TimeStep<dim>::order_) {
    case TimeStep<dim>::Order::first_order:
      deallog << "TimeStep<dim>::Order::first_order" << std::endl;
      break;
    case TimeStep<dim>::Order::second_order:
      deallog << "TimeStep<dim>::Order::second_order" << std::endl;
    }

    deallog << "TimeStep<dim>::smoothen_alpha_ == "
            <<  TimeStep<dim>::smoothen_alpha_ << std::endl;

    /* clang-format on */

    deallog << "Run time parameters:" << std::endl;

    ParameterAcceptor::prm.log_parameters(deallog);

    deallog.push(DEAL_II_GIT_SHORTREV "+" RYUJIN_GIT_SHORTREV);
    deallog.push(base_name);
#ifdef DEBUG
    deallog.depth_console(3);
    deallog.depth_file(3);
    deallog.push("DEBUG");
#else
    deallog.depth_console(2);
    deallog.depth_file(2);
#endif

  }


  template <int dim>
  typename TimeLoop<dim>::vector_type
  TimeLoop<dim>::interpolate_initial_values(double t)
  {
    deallog << "TimeLoop<dim>::interpolate_initial_values(t = " << t << ")"
            << std::endl;
    TimerOutput::Scope timer(computing_timer,
                             "time_loop - setup scratch space");

    vector_type U;

    const auto &locally_owned = offline_data.locally_owned();
    const auto &locally_relevant = offline_data.locally_relevant();
    U[0].reinit(locally_owned, locally_relevant, mpi_communicator);
    for (auto &it : U)
      it.reinit(U[0]);

    constexpr auto problem_dimension =
        ProblemDescription<dim>::problem_dimension;

    const auto callable = [&](const auto &p) {
      return initial_values.initial_state(p, t);
    };

    for (unsigned int i = 0; i < problem_dimension; ++i)
      VectorTools::interpolate(offline_data.dof_handler(),
                               to_function<dim, double>(callable, i),
                               U[i]);

    for (auto &it : U)
      it.update_ghost_values();

    return U;
  }


  template <int dim>
  void
  TimeLoop<dim>::compute_error(const typename TimeLoop<dim>::vector_type &U,
                               const double t)
  {
    deallog << "TimeLoop<dim>::compute_error()" << std::endl;
    TimerOutput::Scope timer(computing_timer, "time_loop - compute error");

    constexpr auto problem_dimension =
        ProblemDescription<dim>::problem_dimension;

    /*
     * Compute L_inf norm:
     */

    Vector<float> difference_per_cell(
        offline_data.discretization().triangulation().n_active_cells());

    double linf_norm = 0.;
    double l1_norm = 0;
    double l2_norm = 0;

    auto analytic = interpolate_initial_values(t);

    for (unsigned int i = 0; i < problem_dimension; ++i) {
      auto &error = analytic[i];

      /*
       * Compute norms of analytic solution:
       */

      const double linf_norm_analytic =
          Utilities::MPI::max(error.linfty_norm(), mpi_communicator);

      VectorTools::integrate_difference(offline_data.dof_handler(),
                                        error,
                                        ZeroFunction<dim, double>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L1_norm);

      const double l1_norm_analytic =
          Utilities::MPI::sum(difference_per_cell.l1_norm(), mpi_communicator);

      VectorTools::integrate_difference(offline_data.dof_handler(),
                                        error,
                                        ZeroFunction<dim, double>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L2_norm);

      const double l2_norm_analytic = std::sqrt(Utilities::MPI::sum(
          std::pow(difference_per_cell.l2_norm(), 2), mpi_communicator));

      /*
       * Compute norms of error:
       */

      error -= U[i];

      const double linf_norm_error =
          Utilities::MPI::max(error.linfty_norm(), mpi_communicator);

      VectorTools::integrate_difference(offline_data.dof_handler(),
                                        error,
                                        ZeroFunction<dim, double>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L1_norm);

      const double l1_norm_error =
          Utilities::MPI::sum(difference_per_cell.l1_norm(), mpi_communicator);

      VectorTools::integrate_difference(offline_data.dof_handler(),
                                        error,
                                        ZeroFunction<dim, double>(),
                                        difference_per_cell,
                                        QGauss<dim>(3),
                                        VectorTools::L2_norm);

      const double l2_norm_error = std::sqrt(Utilities::MPI::sum(
          std::pow(difference_per_cell.l2_norm(), 2), mpi_communicator));

      linf_norm += linf_norm_error / linf_norm_analytic;
      l1_norm += l1_norm_error / l1_norm_analytic;
      l2_norm += l2_norm_error / l2_norm_analytic;
    }

    deallog << "        Normalized consolidated Linf, L1, and L2 errors at "
            << "final time" << std::endl;
    deallog << "        #dofs = " << offline_data.dof_handler().n_dofs()
            << std::endl;
    deallog << "        t     = " << t << std::endl;
    deallog << "        Linf  = " << linf_norm << std::endl;
    deallog << "        L1    = " << l1_norm << std::endl;
    deallog << "        L2    = " << l2_norm << std::endl;
  }


  template <int dim>
  void TimeLoop<dim>::output(const typename TimeLoop<dim>::vector_type &U,
                             const std::string &name,
                             double t,
                             unsigned int cycle)
  {
    deallog << "TimeLoop<dim>::output(t = " << t << ")" << std::endl;

    /*
     * Offload output to a worker thread.
     *
     * We wait for a previous thread to finish before we schedule a new
     * one. This logic also serves as a mutex for output_vector and
     * schlieren_postprocessor.
     */

    deallog << "        Schedule output cycle = " << cycle << std::endl;
    if (output_thread.joinable()) {
      TimerOutput::Scope timer(computing_timer, "time_loop - stalled output");
      output_thread.join();
    }

    /*
     * Copy the current state vector over to output_vector:
     */

    constexpr auto problem_dimension =
        ProblemDescription<dim>::problem_dimension;
    const auto &component_names = ProblemDescription<dim>::component_names;

    for (unsigned int i = 0; i < problem_dimension; ++i) {
      /* This also copies ghost elements: */
      output_vector[i] = U[i];
    }

    schlieren_postprocessor.compute_schlieren(output_vector);
    output_alpha = time_step.alpha();

    /* capture name, t, cycle by value */
    const auto output_worker = [this, name, t, cycle]() {
      const auto &dof_handler = offline_data.dof_handler();
      const auto &triangulation = discretization.triangulation();
      const auto &mapping = discretization.mapping();

      dealii::DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler);

      for (unsigned int i = 0; i < problem_dimension; ++i)
        data_out.add_data_vector(output_vector[i], component_names[i]);

      data_out.add_data_vector(schlieren_postprocessor.schlieren(),
                               "schlieren_plot");

      output_alpha.update_ghost_values();
      data_out.add_data_vector(output_alpha, "alpha");

      data_out.build_patches(mapping,
                             discretization.finite_element().degree - 1);

      DataOutBase::VtkFlags flags(
          t, cycle, true, DataOutBase::VtkFlags::best_speed);
      data_out.set_flags(flags);

      const auto filename = [&](const unsigned int i) -> std::string {
        const auto seq = dealii::Utilities::int_to_string(i, 4);
        return name + "-" + Utilities::int_to_string(cycle, 6) + "-" + seq +
               ".vtu";
      };

      /* Write out local vtu: */
      const unsigned int i = triangulation.locally_owned_subdomain();
      std::ofstream output(filename(i));
      data_out.write_vtu(output);

      if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        /* Write out pvtu control file: */

        const unsigned int n_mpi_processes =
            dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
        std::vector<std::string> filenames;
        for (unsigned int i = 0; i < n_mpi_processes; ++i)
          filenames.push_back(filename(i));

        std::ofstream output(name + "-" + Utilities::int_to_string(cycle, 6) +
                             ".pvtu");
        data_out.write_pvtu_record(output, filenames);
      }

      deallog << "        Commit output cycle = " << cycle << std::endl;
    };

    /*
     * And spawn the thread:
     */
    output_thread = std::move(std::thread(output_worker));
  }

} // namespace ryujin

#endif /* TIMELOOP_TEMPLATE_H */
