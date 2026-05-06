cmake_minimum_required(VERSION 3.20)
project(gw_prototype LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(gwcore
    src/cli.cpp
    src/frequency_grids.cpp
    src/pade.cpp
    src/npy.cpp
    src/linalg.cpp
    src/gw.cpp
    src/io.cpp
    src/profiling.cpp
)

target_include_directories(gwcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_options(gwcore PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion>
)

option(GW_ENABLE_NATIVE "Enable -march=native when supported" OFF)
if(GW_ENABLE_NATIVE)
    target_compile_options(gwcore PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-march=native>)
endif()


# Optional backend switches. Each backend gets its own CMake target.
# This is intentionally not a single mutually-exclusive vendor target: a build may
# provide several backend implementations at once, and the program can select one
# through gw::GwSettings::linalg_backend or a future CLI/runtime factory.
option(GW_ENABLE_BLAS_LAPACK "Build/link a CBLAS/LAPACKE dense CPU backend" ON)
option(GW_ENABLE_SCALAPACK "Build/link a ScaLAPACK distributed dense backend" OFF)
option(GW_ENABLE_COSMA "Build/link a COSMA distributed matrix-multiply backend" OFF)
option(GW_ENABLE_CUDA "Build/link CUDA/cuBLAS/cuSolver backend code" OFF)

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
    find_package(MPI REQUIRED COMPONENTS CXX)
    find_package(SCALAPACK REQUIRED)
    add_library(gw_linalg_scalapack INTERFACE)
    target_link_libraries(gw_linalg_scalapack INTERFACE MPI::MPI_CXX SCALAPACK::SCALAPACK)
    target_compile_definitions(gw_linalg_scalapack INTERFACE GW_HAS_SCALAPACK_BACKEND=1)
    target_link_libraries(gw_linalg_backends INTERFACE gw_linalg_scalapack)
endif()

if(GW_ENABLE_COSMA)
    find_package(MPI REQUIRED COMPONENTS CXX)
    find_package(COSMA REQUIRED)
    add_library(gw_linalg_cosma INTERFACE)
    target_link_libraries(gw_linalg_cosma INTERFACE MPI::MPI_CXX COSMA::cosma)
    target_compile_definitions(gw_linalg_cosma INTERFACE GW_HAS_COSMA_BACKEND=1)
    target_link_libraries(gw_linalg_backends INTERFACE gw_linalg_cosma)
endif()

if(GW_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(gw_linalg_cuda INTERFACE)
    target_link_libraries(gw_linalg_cuda INTERFACE CUDA::cublas CUDA::cusolver)
    target_compile_definitions(gw_linalg_cuda INTERFACE GW_HAS_CUDA_BACKEND=1)
    target_link_libraries(gw_linalg_backends INTERFACE gw_linalg_cuda)
endif()

target_link_libraries(gwcore PUBLIC gw_linalg_backends)

add_executable(gw src/main.cpp)
target_link_libraries(gw PRIVATE gwcore)

option(GW_ENABLE_TESTS "Enable CTest-based regression tests" ON)
if(GW_ENABLE_TESTS)
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
            gw
			--input-dir ${GW_REFERENCE_H2O_DIR}
            --freq-points 200
            --pade-params 16
            --state 5
            --output-dir ${GW_TEST_H2O_DIR}
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
    	        gw
				--input-dir ${GW_REFERENCE_H2O_DIR}
    	        --freq-points 200
    	        --pade-params 16
    	        --state 5
				--output-dir ${GW_TEST_H2O_BLAS_LAPACK_DIR}
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
    	        DEPENDS gw_h2o_reference_run
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

endif()
