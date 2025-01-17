//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "convenience_macros.h"
#include "initial_values.h"
#include "mpi_ensemble.h"
#include "offline_data.h"
#include "sparse_matrix_simd.h"
#include "state_vector.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/sparse_matrix.templates.h>
#include <deal.II/lac/vector.h>

#include <functional>

namespace ryujin
{
  /**
   * An enum controlling the behavior on detection of an invariant domain
   * or CFL violation. Such a case might occur for either aggressive CFL
   * numbers > 1, and/or later stages in the Runge Kutta scheme when the
   * time step tau is prescribed.
   *
   * The invariant domain violation is detected in the limiter and
   * typically implies that the low-order update is already out of bounds.
   *
   * @note Data structures in HyperbolicModule are initialized with the
   * ensemble subrange communicator stored in MPIEnsemble. However, the
   * time step size constraint (i.e. tau_max) is synchronized over the
   * entire global communicator.
   *
   * @ingroup HyperbolicModule
   */
  enum class IDViolationStrategy : std::uint8_t {
    /**
     * Warn about an invariant domain violation but take no further
     * action.
     */
    warn,

    /**
     * Raise a Restart exception on domain violation. This exception can be
     * caught in TimeIntegrator and various different actions (adapt CFL
     * and retry) can be taken depending on chosen strategy.
     */
    raise_exception,
  };


  /**
   * A class signalling a restart, thrown in HyperbolicModule::single_step and
   * caught at various places.
   *
   * @ingroup TimeLoop
   */
  class Restart final
  {
  };


  /**
   * Explicit forward Euler time-stepping for hyperbolic systems with
   * convex limiting.
   *
   * This module is described in detail in @cite ryujin-2021-1, Alg. 1.
   *
   * @ingroup HyperbolicModule
   */
  template <typename Description, int dim, typename Number = double>
  class HyperbolicModule final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @name Typedefs and constexpr constants
     */
    //@{

    using HyperbolicSystem = typename Description::HyperbolicSystem;

    using View =
        typename Description::template HyperbolicSystemView<dim, Number>;

    static constexpr auto problem_dimension = View::problem_dimension;

    using state_type = typename View::state_type;

    using precomputed_type = typename View::precomputed_type;

    using initial_precomputed_type = typename View::initial_precomputed_type;

    using StateVector = typename View::StateVector;

    using InitialPrecomputedVector = typename View::InitialPrecomputedVector;

    static constexpr auto n_precomputation_cycles =
        View::n_precomputation_cycles;

    //@}
    /**
     * @name Constructor and setup
     */
    //@{

    /**
     * Constructor
     */
    HyperbolicModule(
        const MPIEnsemble &mpi_ensemble,
        std::map<std::string, dealii::Timer> &computing_timer,
        const OfflineData<dim, Number> &offline_data,
        const HyperbolicSystem &hyperbolic_system,
        const InitialValues<Description, dim, Number> &initial_values,
        const std::string &subsection = "/HyperbolicModule");

    /**
     * Prepare time stepping. A call to @p prepare() allocates temporary
     * storage and is necessary before any of the following time-stepping
     * functions can be called.
     */
    void prepare();

    //@}
    /**
     * @name Functons for performing explicit time steps
     */
    //@{

    /**
     * This function preprocesses a given state vector @p U in preparation
     * for an explicit euler step performed by the step() function. The
     * function performs the following tasks:
     *
     *  - For a continuous finite element ansatz the method updates the @p U
     *    component of the state vector by enforcing boundary conditions
     *    for the supplied time time @p t. It then updates ghost ranges on
     *    @p U so that the state vector is consistent across MPI ranks.
     *  - For a discontinuous finite element ansatz it populates a local
     *    boundary state vector that is used for computing the boundary
     *    jump terms in the step() function when performing a dG update. It
     *    then updates ghost ranges on @p U so that the state vector is
     *    consistent across MPI ranks.
     *
     *  - The function then runs the precomputation loop that populates the
     *    "precomputed values" component of the state vector and
     *    distributes the result over all MPI ranks by updating ghost
     *    ranges of the precomputed values vector.
     */
    void prepare_state_vector(StateVector &state_vector, Number t) const;

    /**
     * Given a reference to a previous state vector @p old_U perform an
     * explicit euler step (and store the result in @p new_U). The
     * function returns the chosen time step size tau with which the update
     * was performed.
     *
     * The time step is performed with either tau_max (if @p tau is set to
     * 0), or tau (if @p tau is nonzero). Here, tau_max is the minimum of
     * the specified parameter @p tau_max and the computed maximal time
     * step size according to the CFL condition. @p tau is the last
     * parameter of the function.
     *
     * The function takes an optional array of states @p stage_U together
     * with a an array of weights @p stage_weights to construct a
     * modified high-order flux. The standard high-order flux reads
     * (cf @cite ryujin-2021-1, Eq. 12):
     * \f{align}
     *   \newcommand{\bF}{{\boldsymbol F}}
     *   \newcommand{\bU}{{\boldsymbol U}}
     *   \newcommand\bUni{\bU^n_i}
     *   \newcommand\bUnj{\bU^n_j}
     *   \newcommand{\polf}{{\mathbb f}}
     *   \newcommand\Ii{\mathcal{I}(i)}
     *   \newcommand{\bc}{{\boldsymbol c}}
     *   \sum_{j\in\Ii} \frac{m_{ij}}{m_{j}}
     *   \;
     *   \frac{m_{j}}{\tau_n}\big(
     *   \tilde\bU_j^{H,n+1} - \bU_j^{n}\big)
     *   \;=\;
     *   \bF^n_i + \sum_{j\in\Ii}d_{ij}^{H,n}\big(\bUnj-\bUni\big),
     *   \qquad\text{with}\quad
     *   \bF^n_i\;:=\;
     *   \sum_{j\in\Ii}\Big(-(\polf(\bUni)+\polf(\bUnj)) \cdot\bc_{ij}\Big).
     * \f}
     * Instead, the function assembles the modified high-order flux:
     * \f{align}
     *   \newcommand{\bF}{{\boldsymbol F}}
     *   \newcommand{\bU}{{\boldsymbol U}}
     *   \newcommand\bUnis{\bU^{s,n}_i}
     *   \newcommand\bUnjs{\bU^{s,n}_j}
     *   \newcommand{\polf}{{\mathbb f}}
     *   \newcommand\Ii{\mathcal{I}(i)}
     *   \newcommand{\bc}{{\boldsymbol c}}
     *   \tilde{\bF}^n_i\;:=\;
     *   \big(1-\sum_{s=\{1:\text{stages}\}}\omega_s\big)\bF^n_i
     *   \;+\;
     *   \sum_{s=\{1:stages\}}\omega_s \bF^{s,n}_i
     *   \qquad\text{with}\quad
     *   \bF^{s,n}_i\;:=\;
     *   \sum_{j\in\Ii}\Big(-(\polf(\bUnis)+\polf(\bUnjs)) \cdot\bc_{ij}\Big).
     * \f}
     * where \f$\omega_s\f$ denotes the weigths for the given stages
     * \f$\bU^{s,n}\f$.
     *
     * @note The routine only performs an explicit update step on the
     * locally owned dof index range. It neither updates the precomputed
     * block of the state vector, sets boundary conditions (prior) to the
     * update step, nor automatically updates the ghost range of the
     * vector. It is thus necessary to call
     * HyperbolicModule::prepare_state_vector() on @p old_state_vector
     * prior to calling the step function.
     */
    template <int stages>
    Number step(
        const StateVector &old_state_vector,
        std::array<std::reference_wrapper<const StateVector>, stages>
            stage_state_vectors,
        const std::array<Number, stages> stage_weights,
        StateVector &new_state_vector,
        Number tau = Number(0.),
        std::atomic<Number> tau_max = std::numeric_limits<Number>::max()) const;

    /**
     * Sets the relative CFL number used for computing an appropriate
     * time-step size to the given value. The CFL number must be a positive
     * value. If chosen to be within the interval \f$(0,1)\f$ then the
     * low-order update and limiting stages guarantee invariant domain
     * preservation.
     */
    void cfl(Number new_cfl) const
    {
      Assert(cfl_ > Number(0.), dealii::ExcInternalError());
      cfl_ = new_cfl;
    }

    /**
     * Returns the relative CFL number used for computing an appropriate
     * time-step size.
     */
    ACCESSOR_READ_ONLY(cfl)

    /**
     * Return a reference to the OfflineData object
     */
    ACCESSOR_READ_ONLY(offline_data)

    /**
     * Return a reference to the HyperbolicSystem object
     */
    ACCESSOR_READ_ONLY(hyperbolic_system)

    /**
     * Return a reference to the precomputed initial data vector
     */
    ACCESSOR_READ_ONLY(initial_precomputed)

    /**
     * Return a reference to alpha vector storing indicator values. Note
     * that the values stored in alpha correspond to the last step executed
     * by this class.
     */
    ACCESSOR_READ_ONLY(alpha)

    /**
     * The number of restarts issued by the step() function.
     */
    ACCESSOR_READ_ONLY(n_restarts)

    /**
     * The number of ID violation warnings encounterd in the step()
     * function.
     */
    ACCESSOR_READ_ONLY(n_warnings)

    // FIXME: refactor to function
    mutable IDViolationStrategy id_violation_strategy_;

  private:
    //@}
    /**
     * @name Run time options
     */
    //@{
    typename Description::template Indicator<dim, Number>::Parameters
        indicator_parameters_;

    typename Description::template Limiter<dim, Number>::Parameters
        limiter_parameters_;

    typename Description::template RiemannSolver<dim, Number>::Parameters
        riemann_solver_parameters_;

    //@}

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPIEnsemble &mpi_ensemble_;
    std::map<std::string, dealii::Timer> &computing_timer_;

    dealii::SmartPointer<const OfflineData<dim, Number>> offline_data_;
    dealii::SmartPointer<const HyperbolicSystem> hyperbolic_system_;
    dealii::SmartPointer<const InitialValues<Description, dim, Number>>
        initial_values_;

    mutable Number cfl_;

    mutable unsigned int n_restarts_;

    mutable unsigned int n_warnings_;

    InitialPrecomputedVector initial_precomputed_;

    using ScalarVector = typename Vectors::ScalarVector<Number>;
    mutable ScalarVector alpha_;

    static constexpr auto n_bounds =
        Description::template Limiter<dim, Number>::n_bounds;
    mutable Vectors::MultiComponentVector<Number, n_bounds> bounds_;

    using HyperbolicVector =
        Vectors::MultiComponentVector<Number, problem_dimension>;
    mutable HyperbolicVector r_;

    mutable SparseMatrixSIMD<Number> dij_matrix_;
    mutable SparseMatrixSIMD<Number> lij_matrix_;
    mutable SparseMatrixSIMD<Number> lij_matrix_next_;
    mutable SparseMatrixSIMD<Number, problem_dimension> pij_matrix_;

    //@}
  };

} /* namespace ryujin */
