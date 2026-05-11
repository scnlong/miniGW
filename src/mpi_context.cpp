#include "gw/mpi_context.hpp"

#include <stdexcept>

#ifdef GW_ENABLE_MPI
#include <mpi.h>
#endif

namespace gw {

MpiContext::MpiContext(int& argc, char**& argv) {
#ifdef GW_ENABLE_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(&argc, &argv);
        owns_mpi_ = true;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);

    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank_, MPI_INFO_NULL, &local_comm);
    if (local_comm != MPI_COMM_NULL) {
        MPI_Comm_rank(local_comm, &local_rank_);
        MPI_Comm_size(local_comm, &local_size_);
        MPI_Comm_free(&local_comm);
    }
#else
    (void)argc;
    (void)argv;
#endif
}

MpiContext::~MpiContext() {
#ifdef GW_ENABLE_MPI
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (owns_mpi_ && !finalized) {
        MPI_Finalize();
    }
#endif
}

void mpi_allreduce_sum_in_place(std::vector<Complex>& values) {
#ifdef GW_ENABLE_MPI
    if (values.empty()) {
        return;
    }
#ifdef MPI_CXX_DOUBLE_COMPLEX
    const MPI_Datatype complex_datatype = MPI_CXX_DOUBLE_COMPLEX;
#else
    const MPI_Datatype complex_datatype = MPI_C_DOUBLE_COMPLEX;
#endif
    MPI_Allreduce(MPI_IN_PLACE,
                  values.data(),
                  static_cast<int>(values.size()),
                  complex_datatype,
                  MPI_SUM,
                  MPI_COMM_WORLD);
#else
    (void)values;
#endif
}

void mpi_barrier() {
#ifdef GW_ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}

} // namespace gw
