#ifndef RIEMANN_SOLVER_H
#define RIEMANN_SOLVER_H

#include "helper.h"
#include "simd.h"

#include "problem_description.h"

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <functional>

/* FIXME: Currently the handling of compile time constants is a big mess... */
#ifndef NEWTON_MAX_ITER
#define NEWTON_MAX_ITER 0
#endif

namespace grendel
{

  /**
   * A fast approximative Riemann problem solver for the nD compressible
   * Euler problem.
   *
   * FIXME: Desciption
   */
  template <int dim, typename Number = double>
  class RiemannSolver
  {
  public:
    static constexpr unsigned int problem_dimension =
        ProblemDescription<dim, Number>::problem_dimension;

    using rank1_type = typename ProblemDescription<dim, Number>::rank1_type;

    using ScalarNumber = typename get_value_type<Number>::type;

    /*
     * Options:
     */

    static constexpr ScalarNumber newton_eps_ =
        std::is_same<ScalarNumber, double>::value ? ScalarNumber(1.0e-10)
                                                  : ScalarNumber(1.0e-5);

    static constexpr unsigned int newton_max_iter_ = NEWTON_MAX_ITER;

    /**
     * For two given states U_i a U_j and a (normalized) "direction" n_ij
     * compute an estimation of an upper bound for lambda.
     *
     * See [1], page 915, Algorithm 1
     *
     * Returns a tuple consisting of lambda max and the number of Newton
     * iterations used in the solver to find it.
     *
     * References:
     *   [1] J.-L. Guermond, B. Popov. Fast estimation from above for the
     *       maximum wave speed in the Riemann problem for the Euler equations.
     */
    static std::tuple<Number /*lambda_max*/,
                      Number /*p_star*/,
                      unsigned int /*iteration*/>
    compute(const rank1_type U_i,
            const rank1_type U_j,
            const dealii::Tensor<1, dim, Number> &n_ij);


    /**
     * Variant of above function that takes two arrays as input describing
     * the "1D Riemann data" instead of two nD states.
     */
    static std::tuple<Number /*lambda_max*/,
                      Number /*p_star*/,
                      unsigned int /*iteration*/>
    compute(const std::array<Number, 4> &riemann_data_i,
            const std::array<Number, 4> &riemann_data_j);
  };

} /* namespace grendel */

#endif /* RIEMANN_SOLVER_H */
