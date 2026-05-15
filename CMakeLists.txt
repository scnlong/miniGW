cmake_minimum_required(VERSION 3.21)
project(miniGW LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(GW_ENABLE_NATIVE "Enable -march=native when supported" ON)
option(GW_ENABLE_OPENMP "Enable OpenMP support" ON)
option(GW_ENABLE_OPENMP_KERNEL_LOOPS "Enable OpenMP in local kernel loops" ON)
option(GW_ENABLE_OPENMP_FREQUENCY_PARALLEL "Enable experimental OpenMP parallelism over outer frequency points" ON)
option(GW_ENABLE_MPI "Enable MPI support and MPI frequency distribution" ON)

option(GW_ENABLE_BLAS_LAPACK "Build/link a CBLAS/LAPACKE dense CPU backend" ON)
option(GW_ENABLE_SCALAPACK "Build/link a ScaLAPACK replicated-wrapper backend" ON)
option(GW_ENABLE_COSMA "Enable explicit COSMA prefixed pxgemm backend, using cosma_pzgemm_ for distributed GEMM" OFF)
option(GW_ENABLE_CUDA "Build/link a CUDA cuBLAS/cuSolver backend" OFF)

option(GW_ENABLE_TESTS "Enable CTest-based regression tests" ON)
option(GW_ENABLE_PYSCF_TESTS "Enable PySCF-based regression tests" OFF)

add_library(gwcore
    src/linalg/backend_factory.cpp
    src/cli.cpp
    src/execution.cpp
    src/gw/frequency_grids.cpp
    src/gw/pade.cpp
    src/gw/qp_solver.cpp
    src/gw/sigma.cpp
    src/gw/sigma_c_driver.cpp
    src/hdf5_input.cpp
    src/linalg/linalg.cpp
    src/gw/gw.cpp
    src/io.cpp
    src/mpi_context.cpp
    src/memory_footprint.cpp
    src/profiling.cpp
    src/runtime/kernel_policy.cpp
    src/workspace/screening_workspace.cpp
    src/workspace/pq_ph_panel.cpp
)

target_include_directories(gwcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(TARGET HDF5::HDF5)
    target_link_libraries(gwcore PUBLIC HDF5::HDF5)
else()
    target_include_directories(gwcore PUBLIC ${HDF5_INCLUDE_DIRS})
    target_link_libraries(gwcore PUBLIC ${HDF5_LIBRARIES})
endif()


target_compile_options(gwcore PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion>
)

if(GW_ENABLE_SCALAPACK OR GW_ENABLE_COSMA)
    set(GW_ENABLE_MPI ON CACHE BOOL "Enable MPI support" FORCE)
endif()

if(GW_ENABLE_NATIVE)
    target_compile_options(gwcore PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-march=native>
    )
endif()

if(GW_ENABLE_OPENMP_FREQUENCY_PARALLEL)
    set(GW_ENABLE_OPENMP ON CACHE BOOL "Enable OpenMP support" FORCE)
endif()

if(GW_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(gwcore PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(gwcore PUBLIC GW_ENABLE_OPENMP=1)

    if(GW_ENABLE_OPENMP_KERNEL_LOOPS)
        target_compile_definitions(gwcore PUBLIC GW_ENABLE_OPENMP_KERNEL_LOOPS=1)
    endif()

    if(GW_ENABLE_OPENMP_FREQUENCY_PARALLEL)
        target_compile_definitions(gwcore PUBLIC GW_ENABLE_OPENMP_FREQUENCY_PARALLEL=1)
    endif()
endif()

if(GW_ENABLE_MPI OR GW_ENABLE_SCALAPACK OR GW_ENABLE_COSMA)
    find_package(MPI REQUIRED COMPONENTS CXX)

    set(GW_TEST_MPI_LAUNCHER "${MPIEXEC_EXECUTABLE}" CACHE STRING
        "MPI launcher used by CTest MPI regression tests, e.g. mpirun, mpiexec, or srun"
    )

    set(GW_TEST_MPI_NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}" CACHE STRING
        "Flag used by MPI launcher to specify number of ranks, e.g. -n or -np"
    )

    target_link_libraries(gwcore PUBLIC MPI::MPI_CXX)
    target_compile_definitions(gwcore PUBLIC GW_ENABLE_MPI=1)
endif()

add_library(gw_linalg_backends INTERFACE)

if(GW_ENABLE_BLAS_LAPACK)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)

    find_path(CBLAS_INCLUDE_DIR cblas.h REQUIRED)
    find_path(LAPACKE_INCLUDE_DIR lapacke.h REQUIRED)
    find_library(LAPACKE_LIBRARY lapacke REQUIRED)

    add_library(gw_linalg_blas_lapack
        src/linalg/linalg_blas_lapack.cpp
    )

    target_include_directories(gw_linalg_blas_lapack
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE
            ${CBLAS_INCLUDE_DIR}
            ${LAPACKE_INCLUDE_DIR}
    )

    target_link_libraries(gw_linalg_blas_lapack
        PUBLIC
            BLAS::BLAS
            LAPACK::LAPACK
            ${LAPACKE_LIBRARY}
    )

    target_compile_definitions(gw_linalg_blas_lapack
        PUBLIC GW_HAS_BLAS_LAPACK_BACKEND=1
    )

    target_link_libraries(gw_linalg_backends INTERFACE gw_linalg_blas_lapack)
endif()

if(GW_ENABLE_COSMA AND NOT GW_ENABLE_SCALAPACK)
    message(FATAL_ERROR
        "GW_ENABLE_COSMA requires GW_ENABLE_SCALAPACK=ON. "
        "COSMA supplies only the distributed GEMM provider; "
        "ScaLAPACK is still needed for BLACS and pzgetrf/pzgetrs."
    )
endif()

if(GW_ENABLE_SCALAPACK)
    set(SCALAPACK_LIBRARIES "" CACHE STRING
        "ScaLAPACK/BLACS libraries; leave empty to search for a library named scalapack"
    )

    if(SCALAPACK_LIBRARIES STREQUAL "")
        find_library(SCALAPACK_LIBRARY
            NAMES scalapack scalapack-openmpi scalapack-mpich
            REQUIRED
        )
        set(SCALAPACK_LIBRARIES ${SCALAPACK_LIBRARY})
    endif()

    set(GW_COSMA_PXGEMM_LIBRARIES "")
    set(GW_COSMA_GPU_SUPPORT_LIBRARIES "")
    set(GW_COSMA_GPU_CAPABLE OFF)

    if(GW_ENABLE_COSMA)
        set(COSMA_ROOT "" CACHE PATH "COSMA installation prefix")

        if(NOT COSMA_ROOT AND DEFINED ENV{COSMA_ROOT})
            set(COSMA_ROOT "$ENV{COSMA_ROOT}" CACHE PATH "COSMA installation prefix" FORCE)
        endif()

        if(NOT COSMA_ROOT)
            message(FATAL_ERROR "GW_ENABLE_COSMA=ON, but COSMA_ROOT is not set.")
        endif()

        set(_cosma_lib_dirs
            "${COSMA_ROOT}/lib64"
            "${COSMA_ROOT}/lib"
            "${COSMA_ROOT}/usr/local/lib64"
            "${COSMA_ROOT}/usr/local/lib"
        )

        foreach(_dir IN LISTS _cosma_lib_dirs)
            if(NOT COSMA_PREFIXED_PXGEMM_LIBRARY AND EXISTS "${_dir}/libcosma_prefixed_pxgemm.a")
                set(COSMA_PREFIXED_PXGEMM_LIBRARY "${_dir}/libcosma_prefixed_pxgemm.a")
            endif()

            if(NOT COSMA_PXGEMM_CPP_LIBRARY AND EXISTS "${_dir}/libcosma_pxgemm_cpp.a")
                set(COSMA_PXGEMM_CPP_LIBRARY "${_dir}/libcosma_pxgemm_cpp.a")
            endif()

            if(NOT COSMA_LIBRARY AND EXISTS "${_dir}/libcosma.a")
                set(COSMA_LIBRARY "${_dir}/libcosma.a")
            endif()

            if(NOT COSTA_SCALAPACK_LIBRARY AND EXISTS "${_dir}/libcosta_scalapack.a")
                set(COSTA_SCALAPACK_LIBRARY "${_dir}/libcosta_scalapack.a")
            endif()

            if(NOT COSTA_LIBRARY AND EXISTS "${_dir}/libcosta.a")
                set(COSTA_LIBRARY "${_dir}/libcosta.a")
            endif()

            if(NOT TILED_MM_LIBRARY AND EXISTS "${_dir}/libTiled-MM.a")
                set(TILED_MM_LIBRARY "${_dir}/libTiled-MM.a")
            endif()
        endforeach()

        if(NOT COSMA_PREFIXED_PXGEMM_LIBRARY)
            find_library(COSMA_PREFIXED_PXGEMM_LIBRARY
                NAMES cosma_prefixed_pxgemm
                HINTS "${COSMA_ROOT}" "${COSMA_ROOT}/usr/local"
                PATH_SUFFIXES lib64 lib
            )
        endif()

        if(NOT COSMA_PXGEMM_CPP_LIBRARY)
            find_library(COSMA_PXGEMM_CPP_LIBRARY
                NAMES cosma_pxgemm_cpp
                HINTS "${COSMA_ROOT}" "${COSMA_ROOT}/usr/local"
                PATH_SUFFIXES lib64 lib
            )
        endif()

        if(NOT COSMA_LIBRARY)
            find_library(COSMA_LIBRARY
                NAMES cosma
                HINTS "${COSMA_ROOT}" "${COSMA_ROOT}/usr/local"
                PATH_SUFFIXES lib64 lib
            )
        endif()

        if(NOT COSTA_SCALAPACK_LIBRARY)
            find_library(COSTA_SCALAPACK_LIBRARY
                NAMES costa_scalapack
                HINTS "${COSMA_ROOT}" "${COSMA_ROOT}/usr/local"
                PATH_SUFFIXES lib64 lib
            )
        endif()

        if(NOT COSTA_LIBRARY)
            find_library(COSTA_LIBRARY
                NAMES costa
                HINTS "${COSMA_ROOT}" "${COSMA_ROOT}/usr/local"
                PATH_SUFFIXES lib64 lib
            )
        endif()

        if(NOT TILED_MM_LIBRARY)
            find_library(TILED_MM_LIBRARY
                NAMES Tiled-MM tiled_mm
                HINTS "${COSMA_ROOT}" "${COSMA_ROOT}/usr/local"
                PATH_SUFFIXES lib64 lib
            )
        endif()

        if(NOT COSMA_PREFIXED_PXGEMM_LIBRARY OR NOT COSMA_LIBRARY OR NOT COSTA_SCALAPACK_LIBRARY OR NOT COSTA_LIBRARY OR NOT TILED_MM_LIBRARY)
            message(FATAL_ERROR
                "GW_ENABLE_COSMA=ON, but required COSMA libraries were not found under COSMA_ROOT=${COSMA_ROOT}. "
                "Expected libcosma_prefixed_pxgemm.a, libcosma.a, libcosta_scalapack.a, libcosta.a, "
                "and libTiled-MM.a in lib64/, lib/, usr/local/lib64/, or usr/local/lib/. "
                "miniGW calls COSMA through the prefixed PBLAS ABI symbol cosma_pzgemm_, not by "
                "interposing the ordinary ScaLAPACK pzgemm_ symbol."
            )
        endif()

        if(GW_ENABLE_CUDA)
            # COSMA can be compiled with GPU/NCCL support independently from
            # miniGW's own cuBLAS backend.  When CUDA is enabled for this build,
            # add the common CUDA-side COSMA dependencies explicitly so that a
            # GPU-capable COSMA archive is not accidentally reduced to a CPU-only
            # code path by unresolved or unpulled static-archive objects.
            find_package(CUDAToolkit REQUIRED)

            find_library(NCCL_LIBRARY
                NAMES nccl
                HINTS
                    "$ENV{NCCL_ROOT}"
                    "$ENV{EBROOTNCCL}"
                    "$ENV{NCCL_HOME}"
                PATH_SUFFIXES lib lib64
            )

            list(APPEND GW_COSMA_GPU_SUPPORT_LIBRARIES CUDA::cudart CUDA::cublas)

            if(NCCL_LIBRARY)
                list(APPEND GW_COSMA_GPU_SUPPORT_LIBRARIES ${NCCL_LIBRARY})
                set(GW_COSMA_GPU_CAPABLE ON)
                message(STATUS "COSMA GPU support libraries enabled for CUDA/NCCL-capable runs.")
                message(STATUS "  NCCL: ${NCCL_LIBRARY}")
            else()
                message(WARNING
                    "GW_ENABLE_COSMA=ON and GW_ENABLE_CUDA=ON, but NCCL was not found. "
                    "The COSMA CUDA-labelled regression test will still be registered, but "
                    "COSMA may not be able to exercise its NCCL/GPU path. Set NCCL_ROOT or "
                    "NCCL_LIBRARY if your COSMA build requires NCCL.")
            endif()
        endif()

        set(GW_COSMA_PXGEMM_LIBRARIES
            ${COSMA_PREFIXED_PXGEMM_LIBRARY}
            ${COSMA_LIBRARY}
            ${COSTA_SCALAPACK_LIBRARY}
            ${COSTA_LIBRARY}
            ${TILED_MM_LIBRARY}
            ${GW_COSMA_GPU_SUPPORT_LIBRARIES}
        )

        message(STATUS "COSMA explicit prefixed pxgemm backend enabled. miniGW will call cosma_pzgemm_ for distributed GEMM.")
        message(STATUS "  COSMA prefixed pxgemm: ${COSMA_PREFIXED_PXGEMM_LIBRARY}")
        if(COSMA_PXGEMM_CPP_LIBRARY)
            message(STATUS "  COSMA pxgemm C++ helper found but not linked directly: ${COSMA_PXGEMM_CPP_LIBRARY}")
        endif()
        message(STATUS "  COSMA core:           ${COSMA_LIBRARY}")
        message(STATUS "  COSTA ScaLAPACK:      ${COSTA_SCALAPACK_LIBRARY}")
        message(STATUS "  COSTA core:           ${COSTA_LIBRARY}")
        message(STATUS "  Tiled-MM:             ${TILED_MM_LIBRARY}")
        if(GW_COSMA_GPU_SUPPORT_LIBRARIES)
            message(STATUS "  COSMA GPU-side support libraries: ${GW_COSMA_GPU_SUPPORT_LIBRARIES}")
        endif()
    endif()

    add_library(gw_linalg_scalapack
        src/linalg/linalg_scalapack.cpp
        src/matrix/distributed_matrix.cpp
        src/workspace/distributed_screening_workspace.cpp
    )

    target_include_directories(gw_linalg_scalapack
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(gw_linalg_scalapack
        PUBLIC
            MPI::MPI_CXX
            ${SCALAPACK_LIBRARIES}
    )

    target_compile_definitions(gw_linalg_scalapack
        PUBLIC GW_HAS_SCALAPACK_BACKEND=1 GW_ENABLE_MPI=1
    )

    target_compile_definitions(gwcore PUBLIC GW_HAS_SCALAPACK_BACKEND=1)
    target_link_libraries(gw_linalg_backends INTERFACE gw_linalg_scalapack)

    if(GW_ENABLE_COSMA)
        add_library(gw_linalg_cosma
            src/linalg/linalg_cosma.cpp
            src/matrix/cosma_distributed_matrix.cpp
            src/workspace/cosma_distributed_screening_workspace.cpp
        )

        target_include_directories(gw_linalg_cosma
            PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        )

        target_link_libraries(gw_linalg_cosma
            PUBLIC
                gw_linalg_scalapack
                ${GW_COSMA_PXGEMM_LIBRARIES}
                ${SCALAPACK_LIBRARIES}
        )

        target_compile_definitions(gw_linalg_cosma
            PUBLIC GW_HAS_COSMA_BACKEND=1 GW_USE_COSMA_PXGEMM=1 GW_ENABLE_MPI=1
        )
        target_compile_definitions(gwcore PUBLIC GW_HAS_COSMA_BACKEND=1 GW_USE_COSMA_PXGEMM=1)

        if(GW_COSMA_GPU_CAPABLE)
            target_compile_definitions(gw_linalg_cosma PUBLIC GW_COSMA_GPU_CAPABLE=1)
            target_compile_definitions(gwcore PUBLIC GW_COSMA_GPU_CAPABLE=1)
        endif()

        target_link_libraries(gw_linalg_backends INTERFACE gw_linalg_cosma)
    endif()
endif()

if(GW_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(gw_linalg_cublas
        src/linalg/linalg_cublas.cu
        src/workspace/device_screening_workspace.cu
        src/workspace/device_pq_ph_panel.cu
    )

    target_include_directories(gw_linalg_cublas
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(gw_linalg_cublas
        PUBLIC CUDA::cublas CUDA::cusolver CUDA::cudart
    )

    target_compile_features(gw_linalg_cublas PUBLIC cxx_std_20)

    set_target_properties(gw_linalg_cublas
        PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
    )

    target_compile_definitions(gw_linalg_cublas PUBLIC GW_HAS_CUDA_BACKEND=1)
    target_compile_definitions(gwcore PUBLIC GW_HAS_CUDA_BACKEND=1)

    target_link_libraries(gw_linalg_backends INTERFACE gw_linalg_cublas)
endif()

target_link_libraries(gwcore PUBLIC gw_linalg_backends)

add_executable(gw src/main.cpp)
target_link_libraries(gw PRIVATE gwcore)

if(GW_ENABLE_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression_tests")
    enable_testing()
    include(cmake/AddRegressionTests.cmake)
endif()
