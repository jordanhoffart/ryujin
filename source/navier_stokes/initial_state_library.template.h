//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2024 by the ryujin authors
//

#pragma once

#include "description.h"

#include "../euler/initial_state_library_euler.h"
#include <initial_state_library.h>

namespace ryujin
{
  using Description = NavierStokes::Description;

  template <int dim, typename Number>
  class InitialStateLibrary<Description, dim, Number>
  {
  public:
    using HyperbolicSystem = typename Description::HyperbolicSystem;
    using ParabolicSystem = typename Description::ParabolicSystem;

    using View =
        typename Description::template HyperbolicSystemView<dim, Number>;

    using initial_state_list_type =
        std::set<std::unique_ptr<InitialState<Description, dim, Number>>>;

    static void
    populate_initial_state_list(initial_state_list_type &initial_state_list,
                                const HyperbolicSystem &h,
                                const ParabolicSystem & /*p*/,
                                const std::string &s)
    {
      EulerInitialStates::populate_initial_state_list<Description, dim, Number>(
          initial_state_list, h, s);
    }
  };
} // namespace ryujin
