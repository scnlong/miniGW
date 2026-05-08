cmake_minimum_required(VERSION 3.20)
project(miniGW LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(GW_ENABLE_NATIVE "Enable -march=native when supported" ON)
option(GW_ENABLE_OPENMP "Enable OpenMP support" ON)
option(GW_ENABLE_OPENMP_KERNEL_LOOPS "Enable OpenMP in local kernel loops" ON)
option(GW_ENABLE_OPENMP_FREQUENCY_PARALLEL "Enable experimental OpenMP parallelism over outer frequency points" OFF)
option(GW_ENABLE_MPI "Enable MPI support and MPI frequency distribution" ON)

option(GW_ENABLE_BLAS_LAPACK "Build/link a CBLAS/LAPACKE dense CPU backend" ON)
option(GW_ENABLE_SCALAPACK "Build/link a ScaLAPACK backend interface placeholder" OFF)
option(GW_ENABLE_COSMA "Build/link a COSMA backend interface placeholder" OFF)
option(GW_ENABLE_CUDA "Build/link a cuBLAS/cuSolver backend interface placeholder" OFF)

add_library(gwcore
    src/backend_factory.cpp
    src/cli.cpp
    src/execution.cpp
    src/frequency_grids.cpp
    src/pade.cpp
    src/npy.cpp
    src/linalg.cpp
    src/gw.cpp
    src/io.cpp
    src/mpi_context.cpp
    src/memory_footprint.cpp
    src/profiling.cpp
    src/workspace/screening_workspace.cpp
)

target_include_directories(gwcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_options(gwcore PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion>
)

if(GW_ENABLE_NATIVE)
    target_compile_options(gwcore PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-march=native>)
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
    target_link_libraries(gwcore PUBLIC MPI::MPI_CXX)
    target_compile_definitions(gwcore PUBLIC GW_ENABLE_MPI=1)
endif()

add_library(gw_linalg_backends INTERFACE)

# Reference backend is always compiled into gwcore through src/linalg.cpp.
# Optional vendor backends should be added below as separate targets, for example:
#   add_library(gw_linalg_openblas src/linalg_openblas.cpp)
#   target_link_libraries(gw_linalg_openblas PUBLIC gwcore BLAS::BLAS LAPACK::LAPACK)
#   target_link_libraries(gw_linalg_backends INTERFACE gw_linalg_openblas)
# The current project only ships the reference backend implementation, so the
# optional targets below are dependency/link placeholders until the corresponding
# backend source files are added.

if(GW_ENABLE_BLAS_LAPACK)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)

    find_path(CBLAS_INCLUDE_DIR cblas.h REQUIRED)
    find_path(LAPACKE_INCLUDE_DIR lapacke.h REQUIRED)
    find_library(LAPACKE_LIBRARY lapacke REQUIRED)

    add_library(gw_linalg_blas_lapack
        src/linalg_blas_lapack.cpp
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

if(GW_ENABLE_SCALAPACK)
    add_library(gw_linalg_scalapack src/linalg_scalapack.cpp)
    target_include_directories(gw_linalg_scalapack PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(gw_linalg_scalapack PUBLIC MPI::MPI_CXX)
    target_compile_definitions(gw_linalg_scalapack PUBLIC GW_HAS_SCALAPACK_BACKEND=1)
    target_link_libraries(gw_linalg_backends INTERFACE gw_linalg_scalapack)
endif()

if(GW_ENABLE_COSMA)
    add_library(gw_linalg_cosma src/linalg_cosma.cpp)
    target_include_directories(gw_linalg_cosma PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(gw_linalg_cosma PUBLIC MPI::MPI_CXX)
    target_compile_definitions(gw_linalg_cosma PUBLIC GW_HAS_COSMA_BACKEND=1)
    target_link_libraries(gw_linalg_backends INTERFACE gw_linalg_cosma)
endif()

if(GW_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(gw_linalg_cublas src/linalg_cublas.cpp)
    target_include_directories(gw_linalg_cublas PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(gw_linalg_cublas PUBLIC CUDA::cublas CUDA::cusolver)
    target_compile_definitions(gw_linalg_cublas PUBLIC GW_HAS_CUDA_BACKEND=1)
    target_link_libraries(gw_linalg_backends INTERFACE gw_linalg_cublas)
endif()

target_link_libraries(gwcore PUBLIC gw_linalg_backends)

add_executable(gw src/main.cpp)
target_link_libraries(gw PRIVATE gwcore)

option(GW_ENABLE_TESTS "Enable CTest-based regression tests" ON)
if(GW_ENABLE_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression_tests")
    enable_testing()

    find_package(Python3 COMPONENTS Interpreter REQUIRED)

	set(GW_REFERENCE_H2O_DIR
        ${CMAKE_CURRENT_SOURCE_DIR}/regression_tests/h2o_serial
    )

	set(GW_TEST_H2O_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/regression_tests/h2o_serial
    )

    add_test(
        NAME serial_h2o_reference_run
        COMMAND
		${CMAKE_COMMAND} -E make_directory ${GW_TEST_H2O_DIR}
    )

    add_test(
        NAME serial_h2o_reference_compute
        COMMAND
		    ${CMAKE_COMMAND} -E env
			OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
			$<TARGET_FILE:gw>
			--input-dir ${GW_REFERENCE_H2O_DIR}
            --freq-points 200
            --pade-params 16
            --state 5
            --output-dir ${GW_TEST_H2O_DIR}
			--frequency-parallel serial 
			--linalg-backend reference
    )

	add_test(
	    NAME serial_h2o_reference_compare
	    COMMAND
	        ${Python3_EXECUTABLE}
	        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_reference.py
        --reference-file ${GW_REFERENCE_H2O_DIR}/E_c_before_Pade.out
        --output-file ${GW_TEST_H2O_DIR}/E_c_before_Pade.out
        --rtol 1e-8
        --atol 1e-8
	        --compare-mode components
	)

	add_test(
	    NAME serial_h2o_energy_summary_compare
	    COMMAND
	        ${Python3_EXECUTABLE}
	        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_reference.py
	        --comparison energy-summary
	        --reference-file ${GW_REFERENCE_H2O_DIR}/gw.out
	        --output-file ${GW_TEST_H2O_DIR}/gw.out
	        --orbital 5
	        --rtol 1e-5
	        --atol 1e-5
	)

    set_tests_properties(
        serial_h2o_reference_compute
        PROPERTIES
            DEPENDS serial_h2o_reference_run
    )

	set_tests_properties(
	    serial_h2o_reference_compare
	    PROPERTIES
	        DEPENDS serial_h2o_reference_compute
	)

	set_tests_properties(
	    serial_h2o_energy_summary_compare
	    PROPERTIES
	        DEPENDS serial_h2o_reference_compute
	)

	if (PYSCF_ENABLE)
		set(PYSCF_REFERENCE_H2O_DIR
		    ${CMAKE_CURRENT_SOURCE_DIR}/regression_tests/h2o_pyscf
		)

		set(PYSCF_TEST_H2O_DIR
		    ${CMAKE_CURRENT_BINARY_DIR}/regression_tests/h2o_pyscf
		)

		add_test(
		    NAME pyscf_h2o_reference_run
		    COMMAND
		    ${CMAKE_COMMAND} -E make_directory ${PYSCF_TEST_H2O_DIR}
		)

		add_test(
    	    NAME pyscf_h2o_reference_compute
    	    COMMAND
    	        gw
				--input-dir ${PYSCF_REFERENCE_H2O_DIR}
    	        --freq-points 200
    	        --pade-params 16
    	        --state 5
				--output-dir ${PYSCF_TEST_H2O_DIR}
    	)

		add_test(
    	    NAME pyscf_h2o_reference_compare
    	    COMMAND
    	        ${Python3_EXECUTABLE}
    	        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_reference.py
				--reference-file ${PYSCF_REFERENCE_H2O_DIR}/pyscf_E_c_before_Pade.out
				--output-file ${PYSCF_TEST_H2O_DIR}/E_c_before_Pade.out
    	        --rtol 1e-2
    	        --atol 1e-2
    	        --compare-mode components
    	)

		set_tests_properties(
    	    pyscf_h2o_reference_compute
    	    PROPERTIES
    	        DEPENDS pyscf_h2o_reference_run
    	)

    	set_tests_properties(
    	    pyscf_h2o_reference_compare
    	    PROPERTIES
    	        DEPENDS pyscf_h2o_reference_compute
    	)
	endif()

	if(GW_ENABLE_BLAS_LAPACK)
		set(GW_TEST_H2O_BLAS_LAPACK_DIR
    	    ${CMAKE_CURRENT_BINARY_DIR}/regression_tests/h2o_blas_lapack
    	)

    	add_test(
    	    NAME blas_lapack_h2o_reference_run
    	    COMMAND
			${CMAKE_COMMAND} -E make_directory ${GW_TEST_H2O_BLAS_LAPACK_DIR}
    	)

		add_test(
    	    NAME blas_lapack_h2o_reference_compute
    	    COMMAND
		        ${CMAKE_COMMAND} -E env
			    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=4
			    $<TARGET_FILE:gw>
				--input-dir ${GW_REFERENCE_H2O_DIR}
    	        --freq-points 200
    	        --pade-params 16
    	        --state 5
				--output-dir ${GW_TEST_H2O_BLAS_LAPACK_DIR}
			    --frequency-parallel serial 
				--linalg-backend blas-lapack
    	)

		add_test(
    	  	NAME blas_lapack_h2o_reference_compare
    	    COMMAND
    	        ${Python3_EXECUTABLE}
    	        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_reference.py
				--reference-file ${GW_REFERENCE_H2O_DIR}/E_c_before_Pade.out
				--output-file ${GW_TEST_H2O_BLAS_LAPACK_DIR}/E_c_before_Pade.out
    	        --rtol 1e-6
    	        --atol 1e-6
    	        --compare-mode components
    	)

		add_test(
		    NAME blas_lapack_h2o_energy_summary_compare
		    COMMAND
		        ${Python3_EXECUTABLE}
		        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_reference.py
		        --comparison energy-summary
		        --reference-file ${GW_REFERENCE_H2O_DIR}/gw.out
				--output-file ${GW_TEST_H2O_BLAS_LAPACK_DIR}/gw.out
		        --orbital 5
		        --rtol 1e-5
		        --atol 1e-5
		)

		set_tests_properties(
			blas_lapack_h2o_reference_compute
    	    PROPERTIES
    	        DEPENDS blas_lapack_h2o_reference_run
    	)

    	set_tests_properties(
    	    blas_lapack_h2o_reference_compare
    	    PROPERTIES
    	        DEPENDS blas_lapack_h2o_reference_compute
    	)

		set_tests_properties(
		    blas_lapack_h2o_energy_summary_compare
		    PROPERTIES
		        DEPENDS blas_lapack_h2o_reference_compute
		)

	endif()

	if(GW_ENABLE_MPI)
		set(GW_TEST_H2O_MPI_DIR
    	    ${CMAKE_CURRENT_BINARY_DIR}/regression_tests/h2o_mpi
    	)

    	add_test(
    	    NAME mpi_h2o_reference_run
    	    COMMAND
			${CMAKE_COMMAND} -E make_directory ${GW_TEST_H2O_MPI_DIR}
    	)

		add_test(
    	    NAME mpi_h2o_reference_compute
    	    COMMAND
		        mpirun -np 4 env
			    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
			    $<TARGET_FILE:gw>
				--input-dir ${GW_REFERENCE_H2O_DIR}
    	        --freq-points 200
    	        --pade-params 16
    	        --state 5
				--output-dir ${GW_TEST_H2O_MPI_DIR}
			    --frequency-parallel mpi 
				--linalg-backend blas-lapack
    	)

		add_test(
    	  	NAME mpi_h2o_reference_compare
    	    COMMAND
    	        ${Python3_EXECUTABLE}
    	        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_reference.py
				--reference-file ${GW_REFERENCE_H2O_DIR}/E_c_before_Pade.out
				--output-file ${GW_TEST_H2O_MPI_DIR}/E_c_before_Pade.out
    	        --rtol 1e-6
    	        --atol 1e-6
    	        --compare-mode components
    	)

		add_test(
		    NAME mpi_h2o_energy_summary_compare
		    COMMAND
		        ${Python3_EXECUTABLE}
		        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_reference.py
		        --comparison energy-summary
		        --reference-file ${GW_REFERENCE_H2O_DIR}/gw.out
				--output-file ${GW_TEST_H2O_MPI_DIR}/gw.out
		        --orbital 5
		        --rtol 1e-5
		        --atol 1e-5
		)

		set_tests_properties(
			mpi_h2o_reference_compute
    	    PROPERTIES
    	        DEPENDS mpi_h2o_reference_run
    	)

    	set_tests_properties(
    	    mpi_h2o_reference_compare
    	    PROPERTIES
    	        DEPENDS mpi_h2o_reference_compute
    	)

		set_tests_properties(
		    mpi_h2o_energy_summary_compare
		    PROPERTIES
		        DEPENDS mpi_h2o_reference_compute
		)

	endif()
endif()
