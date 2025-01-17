//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#include "mpi_ensemble.h"

namespace ryujin
{
  MPIEnsemble::MPIEnsemble(const MPI_Comm &mpi_communicator,
                           const int n_ensembles /* = 1 */,
                           const bool global_synchronization /* = true */)
      : world_communicator_(mpi_communicator)
      , global_synchronization_(true)
      , world_rank_(0)
      , n_world_ranks_(1)
      , ensemble_(0)
      , n_ensembles_(1)
      , ensemble_rank_(0)
      , n_ensemble_ranks_(1)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "MPIEnsemble::prepare()" << std::endl;
#endif

    n_ensembles_ = n_ensembles;
    global_synchronization_ = global_synchronization;

    world_rank_ = dealii::Utilities::MPI::this_mpi_process(world_communicator_);
    n_world_ranks_ =
        dealii::Utilities::MPI::n_mpi_processes(world_communicator_);

    AssertThrow(n_ensembles_ > 0, dealii::ExcInternalError());
    ensemble_ = world_rank_ % n_ensembles_;

    AssertThrow(
        n_world_ranks_ >= n_ensembles_,
        dealii::ExcMessage(
            "The number of (global) MPI processes must be equal to or larger "
            "than the number of ensembles. But we are trying to run " +
            std::to_string(n_ensembles_) + " ensembles on " +
            std::to_string(n_world_ranks_) + " MPI processes."));

    /*
     * For each ensemble we create an MPI group with a ensemble-local
     * communicator. This allows to do ensemble-independent time-stepping.
     */

    auto ierr = MPI_Comm_group(world_communicator_, &world_group_);
    AssertThrowMPI(ierr);

    /* subrange communicator: */

    ensemble_groups_.resize(n_ensembles_);
    for (int ensemble = 0; ensemble < n_ensembles_; ++ensemble) {
      int ranges[1][3]{{ensemble, n_world_ranks_ - 1, n_ensembles_}}; // NOLINT
      ierr = MPI_Group_range_incl(
          world_group_, 1, ranges, &ensemble_groups_[ensemble]);
      AssertThrowMPI(ierr);
    }

    ierr = MPI_Comm_create_group(world_communicator_,
                                 ensemble_groups_[ensemble_],
                                 ensemble_,
                                 &ensemble_communicator_);
    AssertThrowMPI(ierr);

    /* subrange leader communicator: */

    ensemble_rank_ =
        dealii::Utilities::MPI::this_mpi_process(ensemble_communicator_);
    n_ensemble_ranks_ =
        dealii::Utilities::MPI::n_mpi_processes(ensemble_communicator_);
    AssertThrow(
        (ensemble_rank_ == 0 && world_rank_ < n_ensembles_) ||
            (ensemble_rank_ > 0 && world_rank_ >= n_ensembles_),
        dealii::ExcMessage("MPI Ensemble: Something went horribly wrong: Could "
                           "not determine subrange leader."));

    int ranges[1][3]{{0, n_ensembles_ - 1, 1}}; // NOLINT
    ierr =
        MPI_Group_range_incl(world_group_, 1, ranges, &ensemble_leader_group_);
    AssertThrowMPI(ierr);

    ierr = MPI_Comm_create(world_communicator_,
                           ensemble_leader_group_,
                           &ensemble_leader_communicator_);

    /* peer communicator: */

    ierr = MPI_Comm_split(
        world_communicator_, ensemble_rank_, ensemble_, &peer_communicator_);
    AssertThrowMPI(ierr);

#ifdef DEBUG_OUTPUT
    const auto peer_rank =
        dealii::Utilities::MPI::this_mpi_process(peer_communicator_);
    const auto n_peer_ranks =
        dealii::Utilities::MPI::n_mpi_processes(peer_communicator_);
    std::cout << "RANK: (w = " << world_rank_ << ", e = " << ensemble_rank_
              << ", p = " << peer_rank << ") out of (w = " << n_world_ranks_
              << ", e = " << n_ensemble_ranks_ << ", p = " << n_peer_ranks
              << ") -> belongig to ensemble: " << ensemble_ << std::endl;
#endif
  }

  MPIEnsemble::~MPIEnsemble()
  {
    MPI_Group_free(&world_group_);
    for (auto &it : ensemble_groups_)
      MPI_Group_free(&it);
    MPI_Group_free(&ensemble_leader_group_);

    MPI_Comm_free(&ensemble_communicator_);
    MPI_Comm_free(&ensemble_leader_communicator_);
    MPI_Comm_free(&peer_communicator_);
  }

} /* namespace ryujin */
