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
option(GW_ENABLE_COSMA "Build/link a COSMA GEMM facade with ScaLAPACK solve backend" OFF)
option(GW_ENABLE_CUDA "Build/link a CUDA cuBLAS/cuSolver backend" ON)
option(GW_ENABLE_HIP "Build/link a HIP/ROCm hipBLAS/hipSOLVER backend" OFF)

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
    src/workspace/pq_ph_panel.cpp
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
    set(SCALAPACK_LIBRARIES "" CACHE STRING "ScaLAPACK/BLACS libraries; leave empty to search for a library named scalapack")
    if(SCALAPACK_LIBRARIES STREQUAL "")
        find_library(SCALAPACK_LIBRARY NAMES scalapack scalapack-openmpi scalapack-mpich REQUIRED)
        set(SCALAPACK_LIBRARIES ${SCALAPACK_LIBRARY})
    endif()

    add_library(gw_linalg_scalapack
        src/linalg_scalapack.cpp
        src/matrix/distributed_matrix.cpp
        src/workspace/distributed_screening_workspace.cpp
    )
    target_include_directories(gw_linalg_scalapack PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(gw_linalg_scalapack PUBLIC MPI::MPI_CXX ${SCALAPACK_LIBRARIES})
    target_compile_definitions(gw_linalg_scalapack PUBLIC GW_HAS_SCALAPACK_BACKEND=1 GW_ENABLE_MPI=1)
    target_compile_definitions(gwcore PUBLIC GW_HAS_SCALAPACK_BACKEND=1)
    target_link_libraries(gw_linalg_backends INTERFACE gw_linalg_scalapack)
endif()

if(GW_ENABLE_COSMA)
    if(NOT GW_ENABLE_SCALAPACK)
        message(FATAL_ERROR "GW_ENABLE_COSMA currently requires GW_ENABLE_SCALAPACK=ON because solve/factorization is delegated to ScaLAPACK.")
    endif()
    add_library(gw_linalg_cosma src/linalg_cosma.cpp)
    target_include_directories(gw_linalg_cosma PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(gw_linalg_cosma PUBLIC gw_linalg_scalapack MPI::MPI_CXX)
    target_compile_definitions(gw_linalg_cosma PUBLIC GW_HAS_COSMA_BACKEND=1 GW_ENABLE_MPI=1)
    target_link_libraries(gw_linalg_backends INTERFACE gw_linalg_cosma)
endif()

if(GW_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(gw_linalg_cublas
        src/linalg_cublas.cu
        src/workspace/device_screening_workspace.cu
        src/workspace/device_pq_ph_panel.cu
    )
    target_include_directories(gw_linalg_cublas PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(gw_linalg_cublas PUBLIC CUDA::cublas CUDA::cusolver CUDA::cudart)
    target_compile_features(gw_linalg_cublas PUBLIC cxx_std_20)
    set_target_properties(gw_linalg_cublas PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_compile_definitions(gw_linalg_cublas PUBLIC GW_HAS_CUDA_BACKEND=1)
    target_compile_definitions(gwcore PUBLIC GW_HAS_CUDA_BACKEND=1)
    target_link_libraries(gw_linalg_backends INTERFACE gw_linalg_cublas)
endif()


if(GW_ENABLE_HIP)
    enable_language(HIP)
    find_package(hip REQUIRED)
    find_package(hipblas REQUIRED)
    find_package(hipsolver REQUIRED)
    add_library(gw_linalg_hip
        src/linalg_hip.cpp
        src/workspace/hip_screening_workspace.cpp
        src/workspace/hip_pq_ph_panel.cpp
    )
    target_include_directories(gw_linalg_hip PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(gw_linalg_hip PUBLIC hip::host hipblas hipsolver)
    target_compile_features(gw_linalg_hip PUBLIC cxx_std_20)
    set_source_files_properties(
        src/linalg_hip.cpp
        src/workspace/hip_screening_workspace.cpp
        src/workspace/hip_pq_ph_panel.cpp
        PROPERTIES LANGUAGE HIP
    )
    target_compile_definitions(gw_linalg_hip PUBLIC GW_HAS_HIP_BACKEND=1)
    target_compile_definitions(gwcore PUBLIC GW_HAS_HIP_BACKEND=1)
    target_link_libraries(gw_linalg_backends INTERFACE gw_linalg_hip)
endif()

target_link_libraries(gwcore PUBLIC gw_linalg_backends)

add_executable(gw src/main.cpp)
target_link_libraries(gw PRIVATE gwcore)

option(GW_ENABLE_TESTS "Enable CTest-based regression tests" ON)
if(GW_ENABLE_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/regression_tests")
    enable_testing()

    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    set(GW_TEST_MPI_RANKS 4 CACHE STRING
    "Number of MPI ranks used by CTest MPI regression tests")

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
    	        ${CMAKE_CURRENT_BINARY_DIR}/regression_tests/h2o_mpi)

            add_test(
    	        NAME mpi_h2o_reference_run
    	        COMMAND
	        ${CMAKE_COMMAND} -E make_directory ${GW_TEST_H2O_MPI_DIR})

	    add_test(
    	        NAME mpi_h2o_reference_compute
    	        COMMAND
		${CMAKE_COMMAND} -E env
		OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
		${MPIEXEC_EXECUTABLE}
		${MPIEXEC_NUMPROC_FLAG} ${GW_TEST_MPI_RANKS}
		$<TARGET_FILE:gw>
		--input-dir ${GW_REFERENCE_H2O_DIR}
    	        --freq-points 200
    	        --pade-params 16
    	        --state 5
		--output-dir ${GW_TEST_H2O_MPI_DIR}
		--frequency-parallel mpi 
		--linalg-backend blas-lapack)

            add_test(
                NAME mpi_h2o_reference_compare
                COMMAND
    	        ${Python3_EXECUTABLE}
    	        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_reference.py
		--reference-file ${GW_REFERENCE_H2O_DIR}/E_c_before_Pade.out
		--output-file ${GW_TEST_H2O_MPI_DIR}/E_c_before_Pade.out
    	        --rtol 1e-6
    	        --atol 1e-6
    	        --compare-mode components)

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
                --atol 1e-5)

            set_tests_properties(
                mpi_h2o_reference_compute
                PROPERTIES
    	        DEPENDS mpi_h2o_reference_run)

    	    set_tests_properties(
    	        mpi_h2o_reference_compare
    	        PROPERTIES
    	        DEPENDS mpi_h2o_reference_compute)

            set_tests_properties(
                mpi_h2o_energy_summary_compare
                PROPERTIES
                DEPENDS mpi_h2o_reference_compute)

	endif()

	if(GW_ENABLE_CUDA)
		set(GW_TEST_H2O_CUBLAS_SERIAL_DIR
		    ${CMAKE_CURRENT_BINARY_DIR}/regression_tests/h2o_cublas_serial
		)

		add_test(
		    NAME h2o_cublas_serial_run
		    COMMAND
		        ${CMAKE_COMMAND} -E make_directory ${GW_TEST_H2O_CUBLAS_SERIAL_DIR}
		)

		add_test(
		    NAME h2o_cublas_serial
		    COMMAND
		        ${CMAKE_COMMAND} -E env
		        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
		        $<TARGET_FILE:gw>
		        --input-dir ${GW_REFERENCE_H2O_DIR}
		        --freq-points 200
		        --pade-params 16
		        --state 5
		        --output-dir ${GW_TEST_H2O_CUBLAS_SERIAL_DIR}
		        --linalg-backend cublas
		        --frequency-parallel serial
		)

		add_test(
		    NAME h2o_cublas_serial_compare
		    COMMAND
		        ${Python3_EXECUTABLE}
		        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_reference.py
		        --reference-file ${GW_REFERENCE_H2O_DIR}/E_c_before_Pade.out
		        --output-file ${GW_TEST_H2O_CUBLAS_SERIAL_DIR}/E_c_before_Pade.out
		        --rtol 1e-6
		        --atol 1e-6
		        --compare-mode components
		)

		add_test(
		    NAME h2o_cublas_serial_energy_summary_compare
		    COMMAND
		        ${Python3_EXECUTABLE}
		        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_reference.py
		        --comparison energy-summary
		        --reference-file ${GW_REFERENCE_H2O_DIR}/gw.out
		        --output-file ${GW_TEST_H2O_CUBLAS_SERIAL_DIR}/gw.out
		        --orbital 5
		        --rtol 1e-5
		        --atol 1e-5
		)

		set_tests_properties(
		    h2o_cublas_serial
		    PROPERTIES
		        DEPENDS h2o_cublas_serial_run
		)

		set_tests_properties(
		    h2o_cublas_serial_compare
		    PROPERTIES
		        DEPENDS h2o_cublas_serial
		)

		set_tests_properties(
		    h2o_cublas_serial_energy_summary_compare
		    PROPERTIES
		        DEPENDS h2o_cublas_serial
		)

		if(GW_ENABLE_MPI)
			set(GW_TEST_H2O_CUBLAS_MPI_DIR
			    ${CMAKE_CURRENT_BINARY_DIR}/regression_tests/h2o_cublas_mpi
			)

			add_test(
			    NAME h2o_cublas_mpi_run
			    COMMAND
			        ${CMAKE_COMMAND} -E make_directory ${GW_TEST_H2O_CUBLAS_MPI_DIR}
			)

			add_test(
			    NAME h2o_cublas_mpi
			    COMMAND
				${CMAKE_COMMAND} -E env
				OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
				${MPIEXEC_EXECUTABLE}
				${MPIEXEC_NUMPROC_FLAG} ${GW_TEST_MPI_RANKS}
			        $<TARGET_FILE:gw>
			        --input-dir ${GW_REFERENCE_H2O_DIR}
			        --freq-points 200
			        --pade-params 16
			        --state 5
			        --output-dir ${GW_TEST_H2O_CUBLAS_MPI_DIR}
			        --linalg-backend cublas
			        --frequency-parallel mpi
			        --tasks-per-gpu 2
			)

			add_test(
			    NAME h2o_cublas_mpi_compare
			    COMMAND
			        ${Python3_EXECUTABLE}
			        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_reference.py
			        --reference-file ${GW_REFERENCE_H2O_DIR}/E_c_before_Pade.out
			        --output-file ${GW_TEST_H2O_CUBLAS_MPI_DIR}/E_c_before_Pade.out
			        --rtol 1e-6
			        --atol 1e-6
			        --compare-mode components
			)

			add_test(
			    NAME h2o_cublas_mpi_energy_summary_compare
			    COMMAND
			        ${Python3_EXECUTABLE}
			        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_reference.py
			        --comparison energy-summary
			        --reference-file ${GW_REFERENCE_H2O_DIR}/gw.out
			        --output-file ${GW_TEST_H2O_CUBLAS_MPI_DIR}/gw.out
			        --orbital 5
			        --rtol 1e-5
			        --atol 1e-5
			)

			set_tests_properties(
			    h2o_cublas_mpi
			    PROPERTIES
			        DEPENDS h2o_cublas_mpi_run
			)

			set_tests_properties(
			    h2o_cublas_mpi_compare
			    PROPERTIES
			        DEPENDS h2o_cublas_mpi
			)

			set_tests_properties(
			    h2o_cublas_mpi_energy_summary_compare
			    PROPERTIES
			        DEPENDS h2o_cublas_mpi
			)
		endif()
	endif()

endif()
