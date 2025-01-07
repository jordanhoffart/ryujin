//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 by the ryujin authors
//

#pragma once

#include <initial_values.h>
#include <mpi_ensemble.h>
#include <offline_data.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>

namespace ryujin
{
  /**
   * A stub parabolic solver that does nothing.
   *
   * @todo Remove this class alltogether by generalizing the core
   * ingredients of all StubSolver implementation(s) for the different
   * equations and put them into ParabolicModule (with equation-dependent
   * code in ParabolicSystem).
   *
   * @ingroup ParabolicModule
   */
  template <typename Description, int dim, typename Number>
  class StubSolver final : public dealii::ParameterAcceptor
  {
  public:
    using HyperbolicSystem = typename Description::HyperbolicSystem;
    using ParabolicSystem = typename Description::ParabolicSystem;
    /**
     * Constructor.
     */
    StubSolver(
        const MPIEnsemble & /*mpi_ensemle*/,
        std::map<std::string, dealii::Timer> & /*computing_timer*/,
        const HyperbolicSystem & /*hyperbolic_system*/,
        const ParabolicSystem & /*parabolic_system*/,
        const OfflineData<dim, Number> & /*offline_data*/,
        const InitialValues<Description, dim, Number> & /*initial_values*/,
        const std::string &subsection = "StubSolver")
        : ParameterAcceptor(subsection)
    {
    }
  };
} /* namespace ryujin */
