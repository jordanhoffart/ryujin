//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2024 by the ryujin authors
//

#pragma once

#include "../stub_parabolic_system.h"
#include "../stub_solver.h"
#include "hyperbolic_system.h"
#include "indicator.h"
#include "limiter.h"
#include "riemann_solver.h"

namespace ryujin
{
  namespace Skeleton
  {
    /**
     * A struct that contains all equation specific classes describing the
     * chosen hyperbolic system, the indicator, the limiter and
     * (approximate) Riemann solver.
     *
     * We group all of these templates together in this struct so that we
     * only need to add a single template parameter to the all the
     * algorithm classes, such as HyperbolicModule.
     *
     * @ingroup SkeletonEquations
     */
    struct Description {
      using HyperbolicSystem = Skeleton::HyperbolicSystem;

      template <int dim, typename Number = double>
      using HyperbolicSystemView = Skeleton::HyperbolicSystemView<dim, Number>;

      using ParabolicSystem = ryujin::StubParabolicSystem;

      template <int dim, typename Number = double>
      using ParabolicSolver = ryujin::StubSolver<Description, dim, Number>;

      template <int dim, typename Number = double>
      using Indicator = Skeleton::Indicator<dim, Number>;

      template <int dim, typename Number = double>
      using Limiter = Skeleton::Limiter<dim, Number>;

      template <int dim, typename Number = double>
      using RiemannSolver = Skeleton::RiemannSolver<dim, Number>;
    };
  } // namespace Skeleton
} // namespace ryujin
