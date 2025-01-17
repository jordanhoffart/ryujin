//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2024 by the ryujin authors
//

#pragma once

#include "hyperbolic_module.h"

#include <compile_time_options.h>

#include <convenience_macros.h>
#include <initial_values.h>
#include <mpi_ensemble.h>
#include <offline_data.h>
#include <simd.h>
#include <sparse_matrix_simd.h>

#include "parabolic_solver_gmg_operators.h"

#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.templates.h>
#include <deal.II/lac/vector.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/multigrid/mg_base.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

namespace ryujin
{
  namespace NavierStokes
  {
    template <int, typename>
    class DiagonalMatrix;

    /**
     * Implicit backward-Euler time stepping for the parabolic limiting
     * equation @cite ryujin-2021-2, Eq. 3.3:
     * \f{align}
     *   \newcommand{\bbm}{{\boldsymbol m}}
     *   \newcommand{\bef}{{\boldsymbol f}}
     *   \newcommand{\bk}{{\boldsymbol k}}
     *   \newcommand{\bu}{{\boldsymbol u}}
     *   \newcommand{\bv}{{\boldsymbol v}}
     *   \newcommand{\bn}{{\boldsymbol n}}
     *   \newcommand{\pols}{{\mathbb s}}
     *   \newcommand{\Hflux}{\bk}
     *   &\partial_t \rho  =  0,
     *   \\
     *   &\partial_t \bbm - \nabla\cdot(\pols(\bv)) = \bef,
     *   \\
     *   &\partial_t E   + \nabla\cdot(\Hflux(\bu)- \pols(\bv) \bv) =
     * \bef\cdot\bv,
     *   \\
     *   &\bv_{|\partial D}=\boldsymbol 0, \qquad \Hflux(\bu)\cdot\bn_{|\partial
     * D}=0 . \f}
     *
     * Internally, the module first performs an implicit backward Euler
     * step updating the velocity (see @cite ryujin-2021-2, Eq. 5.5):
     * \f{align}
     *   \begin{cases}
     *     \newcommand\bsfV{{\textbf V}}
     *     \newcommand{\polB}{{\mathbb B}}
     *     \newcommand{\calI}{{\mathcal I}}
     *     \newcommand\bsfF{{\textbf F}}
     *     \newcommand\bsfM{{\textbf M}}
     *     \newcommand{\upint}{^\circ}
     *     \newcommand{\upbnd}{^\partial}
     *     \newcommand{\dt}{{\tau}}
     *     \newcommand{\calV}{{\mathcal V}}
     *     \varrho^{n}_i m_i \bsfV^{n+1} +
     *     \dt\sum_{j\in\calI(i)} \polB_{ij} \bsfV^{n+1} =
     *     m_i \bsfM_i^{n} + \dt m_i \bsfF_i^{n+1},
     *     & \forall i\in \calV\upint
     *     \\[0.3em]
     *     \bsfV_i^{n+1} = \boldsymbol 0, &  \forall i\in \calV\upbnd,
     *   \end{cases}
     * \f}
     * We then postprocess and compute an internal energy update with an
     * additional backward Euler step, (cf. @cite ryujin-2021-2, Eq. 5.13)
     * \f{align}
     *     \newcommand\bsfV{{\textbf V}}
     *     \newcommand\sfe{{\mathsf e}}
     *     \newcommand{\upHnph}{^{\text{H},n+1}}
     *     \newcommand{\calI}{{\mathcal I}}
     *     \newcommand\sfK{{\mathsf K}}
     *     \newcommand{\calV}{{\mathcal V}}
     *     m_i \varrho_i^{n}(\sfe_i{\upHnph} - \sfe_i^{n})+\dt
     *     \sum_{j\in\calI(i)} \beta_{ij}\sfe_i{\upHnph}
     *     = \tfrac12 m_i\|\bsfV^{n+1}-\bsfV^{n}\|^2
     *     + \dt m_i\sfK_i^{n+1}, \qquad \forall i\in \calV.
     * \f}
     * The result is then transformed back into conserved quantities and
     * written to the output vector.
     *
     * @note The backward Euler scheme is a fundamental building block for
     * higher-order time stepping, including the well-known Crank-Nicolson
     * scheme. The latter can be expressed algebraically as a backward
     * Euler step (from time \f$t\f$ to \f$t+\tau\f$ followed by an
     * extrapolation step \f$U^{n+2}=2U^{n+1}-U^{n}\f$ from time
     * \f$t+\tau\f$ to \f$t+2\tau\f$). This approach differs from the
     * Crank-Nicolson scheme discussed in @cite ryujin-2021-2 where the
     * extrapolation step is performed on the primitive quantities
     * (velocity and internal energy) instead of the conserved quantities.
     *
     * @ingroup NavierStokesEquations
     */
    template <typename Description, int dim, typename Number = double>
    class ParabolicSolver final : public dealii::ParameterAcceptor
    {
    public:
      /**
       * @name Typedefs and constexpr constants
       */
      //@{

      using HyperbolicSystem = typename Description::HyperbolicSystem;

      using View =
          typename Description::template HyperbolicSystemView<dim, Number>;

      using ParabolicSystem = typename Description::ParabolicSystem;

      using ScalarNumber = typename View::ScalarNumber;

      static constexpr auto problem_dimension = View::problem_dimension;

      using state_type = typename View::state_type;

      using StateVector = typename View::StateVector;

      using ScalarVector = Vectors::ScalarVector<Number>;

      using BlockVector = Vectors::BlockVector<Number>;

      //@}
      /**
       * @name Constructor and setup
       */
      //@{

      /**
       * Constructor.
       */
      ParabolicSolver(
          const MPIEnsemble &mpi_ensemble,
          std::map<std::string, dealii::Timer> &computing_timer,
          const HyperbolicSystem &hyperbolic_system,
          const ParabolicSystem &parabolic_system,
          const OfflineData<dim, Number> &offline_data,
          const InitialValues<Description, dim, Number> &initial_values,
          const std::string &subsection = "ParabolicSolver");

      /**
       * Prepare time stepping. A call to @ref prepare() allocates temporary
       * storage and is necessary before any of the following time-stepping
       * functions can be called.
       */
      void prepare();

      //@}
      /**
       * @name Functions for performing implicit time steps
       */
      //@{

      /**
       * Given a reference to a previous state vector @p old_state_vector
       * at time @p old_t and a time-step size @p tau perform an implicit
       * backward Euler step (and store the result in @p new_state_vector).
       */
      void backward_euler_step(const StateVector &old_state_vector,
                               const Number old_t,
                               StateVector &new_state_vector,
                               Number tau,
                               const IDViolationStrategy id_violation_strategy,
                               const bool reinitialize_gmg) const;

      /**
       * Print a status line with solver statistics. This function is used
       * for constructing the status message displayed periodically in the
       * TimeLoop.
       */
      void print_solver_statistics(std::ostream &output) const;

      //@}
      /**
       * @name Accessors
       */
      //@{

      ACCESSOR_READ_ONLY(n_restarts)
      ACCESSOR_READ_ONLY(n_warnings)

      //@}

    private:
      /**
       * @name Run time options
       */
      //@{

      bool use_gmg_velocity_;
      ACCESSOR_READ_ONLY(use_gmg_velocity)

      bool use_gmg_internal_energy_;
      ACCESSOR_READ_ONLY(use_gmg_internal_energy)

      Number tolerance_;
      bool tolerance_linfty_norm_;

      unsigned int gmg_max_iter_vel_;
      unsigned int gmg_max_iter_en_;
      double gmg_smoother_range_vel_;
      double gmg_smoother_range_en_;
      double gmg_smoother_max_eig_vel_;
      double gmg_smoother_max_eig_en_;
      unsigned int gmg_smoother_degree_;
      unsigned int gmg_smoother_n_cg_iter_;
      unsigned int gmg_min_level_;

      //@}
      /**
       * @name Internal data
       */
      //@{

      // FIXME: refactor
      static constexpr unsigned int order_fe = 1;
      static constexpr unsigned int order_quad = 2;

      const MPIEnsemble &mpi_ensemble_;
      std::map<std::string, dealii::Timer> &computing_timer_;

      dealii::SmartPointer<const HyperbolicSystem> hyperbolic_system_;
      dealii::SmartPointer<const ParabolicSystem> parabolic_system_;
      dealii::SmartPointer<const ryujin::OfflineData<dim, Number>>
          offline_data_;
      dealii::SmartPointer<
          const ryujin::InitialValues<Description, dim, Number>>
          initial_values_;

      mutable unsigned int n_restarts_;
      mutable unsigned int n_warnings_;
      mutable double n_iterations_velocity_;
      mutable double n_iterations_internal_energy_;

      mutable dealii::MatrixFree<dim, Number> matrix_free_;

      mutable BlockVector velocity_;
      mutable BlockVector velocity_rhs_;
      mutable ScalarVector internal_energy_;
      mutable ScalarVector internal_energy_rhs_;
      mutable ScalarVector density_;

      mutable dealii::MGLevelObject<dealii::MatrixFree<dim, float>>
          level_matrix_free_;
      mutable dealii::MGConstrainedDoFs mg_constrained_dofs_;
      mutable dealii::MGLevelObject<
          dealii::LinearAlgebra::distributed::Vector<float>>
          level_density_;
      mutable MGTransferVelocity<dim, float> mg_transfer_velocity_;
      mutable dealii::MGLevelObject<VelocityMatrix<dim, float, Number>>
          level_velocity_matrices_;
      mutable MGTransferEnergy<dim, float> mg_transfer_energy_;
      mutable dealii::MGLevelObject<EnergyMatrix<dim, float, Number>>
          level_energy_matrices_;

      mutable dealii::mg::SmootherRelaxation<
          dealii::PreconditionChebyshev<
              VelocityMatrix<dim, float, Number>,
              dealii::LinearAlgebra::distributed::BlockVector<float>,
              DiagonalMatrix<dim, float>>,
          dealii::LinearAlgebra::distributed::BlockVector<float>>
          mg_smoother_velocity_;

      mutable dealii::mg::SmootherRelaxation<
          dealii::PreconditionChebyshev<
              EnergyMatrix<dim, float, Number>,
              dealii::LinearAlgebra::distributed::Vector<float>>,
          dealii::LinearAlgebra::distributed::Vector<float>>
          mg_smoother_energy_;

      //@}
    };

  } // namespace NavierStokes
} /* namespace ryujin */
