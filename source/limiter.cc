//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#include "limiter.template.h"

using namespace dealii;

namespace ryujin
{
  /* instantiations */

#ifndef OBSESSIVE_INLINING
  template std::tuple<NUMBER, bool>
  Limiter<DIM, NUMBER>::limit<Limiter<DIM, NUMBER>::Limiters::specific_entropy>(
      const ProblemDescription &,
      const std::array<NUMBER, 3> &,
      const state_type &,
      const state_type &,
      const NUMBER,
      const NUMBER);

  template std::tuple<VectorizedArray<NUMBER>, bool>
  Limiter<DIM, VectorizedArray<NUMBER>>::limit<
      Limiter<DIM, VectorizedArray<NUMBER>>::Limiters::specific_entropy>(
      const ProblemDescription &,
      const std::array<VectorizedArray<NUMBER>, 3> &,
      const state_type &,
      const state_type &,
      const VectorizedArray<NUMBER>,
      const VectorizedArray<NUMBER>);
#endif

} // namespace ryujin
