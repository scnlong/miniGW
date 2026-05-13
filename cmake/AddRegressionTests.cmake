find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(GW_TEST_MPI_RANKS 4 CACHE STRING
    "Number of MPI ranks used by CTest MPI regression tests"
)

set(GW_REFERENCE_H2O_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/regression_tests/h2o_serial"
)

set(GW_REGRESSION_OUTPUT_ROOT
    "${CMAKE_CURRENT_BINARY_DIR}/regression_tests"
)

function(gw_set_test_labels labels)
    set_tests_properties(${ARGN}
        PROPERTIES LABELS "${labels}"
    )
endfunction()

function(gw_add_component_compare_test test_name reference_dir output_dir rtol atol dependency labels)
    add_test(
        NAME ${test_name}
        COMMAND
            ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_reference.py"
            --reference-file "${reference_dir}/E_c_before_Pade.out"
            --output-file "${output_dir}/E_c_before_Pade.out"
            --rtol ${rtol}
            --atol ${atol}
            --compare-mode components
    )

    set_tests_properties(${test_name}
        PROPERTIES
            DEPENDS ${dependency}
            LABELS "${labels}"
    )
endfunction()

function(gw_add_energy_summary_compare_test test_name reference_dir output_dir rtol atol dependency labels)
    add_test(
        NAME ${test_name}
        COMMAND
            ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_reference.py"
            --comparison energy-summary
            --reference-file "${reference_dir}/gw.out"
            --output-file "${output_dir}/gw.out"
            --orbital 5
            --rtol ${rtol}
            --atol ${atol}
    )

    set_tests_properties(${test_name}
        PROPERTIES
            DEPENDS ${dependency}
            LABELS "${labels}"
    )
endfunction()

function(gw_add_h2o_reference_test)
    set(test_dir "${GW_REGRESSION_OUTPUT_ROOT}/h2o_serial")
    set(labels "serial;cpu")

    add_test(
        NAME serial_h2o_reference_run
        COMMAND
            ${CMAKE_COMMAND} -E make_directory "${test_dir}"
    )

    add_test(
        NAME serial_h2o_reference_compute
        COMMAND
            ${CMAKE_COMMAND} -E env
            OMP_NUM_THREADS=1
            OPENBLAS_NUM_THREADS=1
            $<TARGET_FILE:gw>
            --input-dir "${GW_REFERENCE_H2O_DIR}"
            --freq-points 200
            --pade-params 16
            --state 5
            --output-dir "${test_dir}"
            --frequency-parallel serial
            --linalg-backend reference
    )

    gw_add_component_compare_test(
        serial_h2o_reference_compare
        "${GW_REFERENCE_H2O_DIR}"
        "${test_dir}"
        1e-8
        1e-8
        serial_h2o_reference_compute
        "${labels}"
    )

    gw_add_energy_summary_compare_test(
        serial_h2o_energy_summary_compare
        "${GW_REFERENCE_H2O_DIR}"
        "${test_dir}"
        1e-5
        1e-5
        serial_h2o_reference_compute
        "${labels}"
    )

    set_tests_properties(serial_h2o_reference_compute
        PROPERTIES
            PROCESSORS 1
            DEPENDS serial_h2o_reference_run
    )

    gw_set_test_labels("${labels}"
        serial_h2o_reference_run
        serial_h2o_reference_compute
    )
endfunction()

function(gw_add_h2o_pyscf_test)
    if(NOT GW_ENABLE_PYSCF_TESTS)
        return()
    endif()

    set(reference_dir "${CMAKE_CURRENT_SOURCE_DIR}/regression_tests/h2o_pyscf")
    set(test_dir "${GW_REGRESSION_OUTPUT_ROOT}/h2o_pyscf")
    set(labels "serial;cpu;pyscf")

    add_test(
        NAME pyscf_h2o_reference_run
        COMMAND
            ${CMAKE_COMMAND} -E make_directory "${test_dir}"
    )

    add_test(
        NAME pyscf_h2o_reference_compute
        COMMAND
            ${CMAKE_COMMAND} -E env
            OMP_NUM_THREADS=1
            OPENBLAS_NUM_THREADS=1
            $<TARGET_FILE:gw>
            --input-dir "${reference_dir}"
            --freq-points 200
            --pade-params 16
            --state 5
            --output-dir "${test_dir}"
    )

    add_test(
        NAME pyscf_h2o_reference_compare
        COMMAND
            ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_reference.py"
            --reference-file "${reference_dir}/pyscf_E_c_before_Pade.out"
            --output-file "${test_dir}/E_c_before_Pade.out"
            --rtol 1e-2
            --atol 1e-2
            --compare-mode components
    )

    set_tests_properties(pyscf_h2o_reference_compute
        PROPERTIES
            PROCESSORS 1
            DEPENDS pyscf_h2o_reference_run
    )

    set_tests_properties(pyscf_h2o_reference_compare
        PROPERTIES
            DEPENDS pyscf_h2o_reference_compute
    )

    gw_set_test_labels("${labels}"
        pyscf_h2o_reference_run
        pyscf_h2o_reference_compute
        pyscf_h2o_reference_compare
    )
endfunction()

function(gw_add_h2o_blas_lapack_test)
    if(NOT GW_ENABLE_BLAS_LAPACK)
        return()
    endif()

    set(test_dir "${GW_REGRESSION_OUTPUT_ROOT}/h2o_blas_lapack")
    set(labels "serial;cpu;lapack")

    add_test(
        NAME blas_lapack_h2o_reference_run
        COMMAND
            ${CMAKE_COMMAND} -E make_directory "${test_dir}"
    )

    add_test(
        NAME blas_lapack_h2o_reference_compute
        COMMAND
            ${CMAKE_COMMAND} -E env
            OMP_NUM_THREADS=1
            OPENBLAS_NUM_THREADS=4
            $<TARGET_FILE:gw>
            --input-dir "${GW_REFERENCE_H2O_DIR}"
            --freq-points 200
            --pade-params 16
            --state 5
            --output-dir "${test_dir}"
            --frequency-parallel serial
            --linalg-backend blas-lapack
    )

    gw_add_component_compare_test(
        blas_lapack_h2o_reference_compare
        "${GW_REFERENCE_H2O_DIR}"
        "${test_dir}"
        1e-6
        1e-6
        blas_lapack_h2o_reference_compute
        "${labels}"
    )

    gw_add_energy_summary_compare_test(
        blas_lapack_h2o_energy_summary_compare
        "${GW_REFERENCE_H2O_DIR}"
        "${test_dir}"
        1e-5
        1e-5
        blas_lapack_h2o_reference_compute
        "${labels}"
    )

    set_tests_properties(blas_lapack_h2o_reference_compute
        PROPERTIES
            PROCESSORS 4
            DEPENDS blas_lapack_h2o_reference_run
    )

    gw_set_test_labels("${labels}"
        blas_lapack_h2o_reference_run
        blas_lapack_h2o_reference_compute
    )
endfunction()

function(gw_add_h2o_mpi_blas_lapack_test)
    if(NOT GW_ENABLE_MPI OR NOT GW_ENABLE_BLAS_LAPACK)
        return()
    endif()

    set(test_dir "${GW_REGRESSION_OUTPUT_ROOT}/h2o_mpi")
    set(labels "mpi;cpu")

    add_test(
        NAME mpi_h2o_reference_run
        COMMAND
            ${CMAKE_COMMAND} -E make_directory "${test_dir}"
    )

    add_test(
        NAME mpi_h2o_reference_compute
        COMMAND
            ${CMAKE_COMMAND} -E env
            OMP_NUM_THREADS=1
            OPENBLAS_NUM_THREADS=1
            ${GW_TEST_MPI_LAUNCHER}
            ${GW_TEST_MPI_NUMPROC_FLAG} ${GW_TEST_MPI_RANKS}
            $<TARGET_FILE:gw>
            --input-dir "${GW_REFERENCE_H2O_DIR}"
            --freq-points 200
            --pade-params 16
            --state 5
            --output-dir "${test_dir}"
            --frequency-parallel mpi
            --linalg-backend blas-lapack
    )

    gw_add_component_compare_test(
        mpi_h2o_reference_compare
        "${GW_REFERENCE_H2O_DIR}"
        "${test_dir}"
        1e-6
        1e-6
        mpi_h2o_reference_compute
        "${labels}"
    )

    gw_add_energy_summary_compare_test(
        mpi_h2o_energy_summary_compare
        "${GW_REFERENCE_H2O_DIR}"
        "${test_dir}"
        1e-5
        1e-5
        mpi_h2o_reference_compute
        "${labels}"
    )

    set_tests_properties(mpi_h2o_reference_compute
        PROPERTIES
            PROCESSORS 4
            DEPENDS mpi_h2o_reference_run
    )

    gw_set_test_labels("${labels}"
        mpi_h2o_reference_run
        mpi_h2o_reference_compute
    )
endfunction()

function(gw_add_h2o_scalapack_test)
    if(NOT GW_ENABLE_MPI OR NOT GW_ENABLE_SCALAPACK)
        return()
    endif()

    set(test_dir "${GW_REGRESSION_OUTPUT_ROOT}/h2o_scalapack")
    set(labels "mpi;cpu;scalapack")

    add_test(
        NAME scalapack_h2o_reference_run
        COMMAND
            ${CMAKE_COMMAND} -E make_directory "${test_dir}"
    )

    add_test(
        NAME scalapack_h2o_reference_compute
        COMMAND
            ${CMAKE_COMMAND} -E env
            OMP_NUM_THREADS=1
            OPENBLAS_NUM_THREADS=1
            ${GW_TEST_MPI_LAUNCHER}
            ${GW_TEST_MPI_NUMPROC_FLAG} ${GW_TEST_MPI_RANKS}
            $<TARGET_FILE:gw>
            --input-dir "${GW_REFERENCE_H2O_DIR}"
            --freq-points 200
            --pade-params 16
            --state 5
            --output-dir "${test_dir}"
            --frequency-parallel mpi
            --linalg-backend scalapack
    )

    gw_add_component_compare_test(
        scalapack_h2o_reference_compare
        "${GW_REFERENCE_H2O_DIR}"
        "${test_dir}"
        1e-6
        1e-6
        scalapack_h2o_reference_compute
        "${labels}"
    )

    gw_add_energy_summary_compare_test(
        scalapack_h2o_energy_summary_compare
        "${GW_REFERENCE_H2O_DIR}"
        "${test_dir}"
        1e-5
        1e-5
        scalapack_h2o_reference_compute
        "${labels}"
    )

    set_tests_properties(scalapack_h2o_reference_compute
        PROPERTIES
            PROCESSORS 4
            DEPENDS scalapack_h2o_reference_run
    )

    gw_set_test_labels("${labels}"
        scalapack_h2o_reference_run
        scalapack_h2o_reference_compute
    )
endfunction()

function(gw_add_h2o_cosma_test)
    if(NOT GW_ENABLE_MPI OR NOT GW_ENABLE_SCALAPACK OR NOT GW_ENABLE_COSMA)
        return()
    endif()

    set(test_dir "${GW_REGRESSION_OUTPUT_ROOT}/h2o_cosma")
    set(labels "mpi;cpu;cosma")

    add_test(
        NAME cosma_h2o_reference_run
        COMMAND
            ${CMAKE_COMMAND} -E make_directory "${test_dir}"
    )

    add_test(
        NAME cosma_h2o_reference_compute
        COMMAND
            ${CMAKE_COMMAND} -E env
            OMP_NUM_THREADS=1
            OPENBLAS_NUM_THREADS=1
            ${GW_TEST_MPI_LAUNCHER}
            ${GW_TEST_MPI_NUMPROC_FLAG} ${GW_TEST_MPI_RANKS}
            $<TARGET_FILE:gw>
            --input-dir "${GW_REFERENCE_H2O_DIR}"
            --freq-points 200
            --pade-params 16
            --state 5
            --output-dir "${test_dir}"
            --frequency-parallel mpi
            --linalg-backend cosma
    )

    gw_add_component_compare_test(
        cosma_h2o_reference_compare
        "${GW_REFERENCE_H2O_DIR}"
        "${test_dir}"
        1e-6
        1e-6
        cosma_h2o_reference_compute
        "${labels}"
    )

    gw_add_energy_summary_compare_test(
        cosma_h2o_energy_summary_compare
        "${GW_REFERENCE_H2O_DIR}"
        "${test_dir}"
        1e-5
        1e-5
        cosma_h2o_reference_compute
        "${labels}"
    )

    set_tests_properties(cosma_h2o_reference_compute
        PROPERTIES
            PROCESSORS 4
            DEPENDS cosma_h2o_reference_run
    )

    gw_set_test_labels("${labels}"
        cosma_h2o_reference_run
        cosma_h2o_reference_compute
    )
endfunction()

function(gw_add_h2o_cublas_serial_test)
    if(NOT GW_ENABLE_CUDA)
        return()
    endif()

    set(test_dir "${GW_REGRESSION_OUTPUT_ROOT}/h2o_cublas_serial")
    set(labels "serial;cuda")

    add_test(
        NAME h2o_cublas_serial_run
        COMMAND
            ${CMAKE_COMMAND} -E make_directory "${test_dir}"
    )

    add_test(
        NAME h2o_cublas_serial_compute
        COMMAND
            ${CMAKE_COMMAND} -E env
            OMP_NUM_THREADS=1
            OPENBLAS_NUM_THREADS=1
            $<TARGET_FILE:gw>
            --input-dir "${GW_REFERENCE_H2O_DIR}"
            --freq-points 200
            --pade-params 16
            --state 5
            --output-dir "${test_dir}"
            --linalg-backend cublas
            --frequency-parallel serial
    )

    gw_add_component_compare_test(
        h2o_cublas_serial_compare
        "${GW_REFERENCE_H2O_DIR}"
        "${test_dir}"
        1e-6
        1e-6
        h2o_cublas_serial_compute
        "${labels}"
    )

    gw_add_energy_summary_compare_test(
        h2o_cublas_serial_energy_summary_compare
        "${GW_REFERENCE_H2O_DIR}"
        "${test_dir}"
        1e-5
        1e-5
        h2o_cublas_serial_compute
        "${labels}"
    )

    set_tests_properties(h2o_cublas_serial_compute
        PROPERTIES
            PROCESSORS 1
            RESOURCE_LOCK gpu
            DEPENDS h2o_cublas_serial_run
    )

    gw_set_test_labels("${labels}"
        h2o_cublas_serial_run
        h2o_cublas_serial_compute
    )
endfunction()

function(gw_add_h2o_cublas_mpi_test)
    if(NOT GW_ENABLE_CUDA OR NOT GW_ENABLE_MPI)
        return()
    endif()

    set(test_dir "${GW_REGRESSION_OUTPUT_ROOT}/h2o_cublas_mpi")
    set(labels "mpi;cuda")

    add_test(
        NAME h2o_cublas_mpi_run
        COMMAND
            ${CMAKE_COMMAND} -E make_directory "${test_dir}"
    )

    add_test(
        NAME h2o_cublas_mpi_compute
        COMMAND
            ${CMAKE_COMMAND} -E env
            OMP_NUM_THREADS=1
            OPENBLAS_NUM_THREADS=1
            ${GW_TEST_MPI_LAUNCHER}
            ${GW_TEST_MPI_NUMPROC_FLAG} ${GW_TEST_MPI_RANKS}
            $<TARGET_FILE:gw>
            --input-dir "${GW_REFERENCE_H2O_DIR}"
            --freq-points 200
            --pade-params 16
            --state 5
            --output-dir "${test_dir}"
            --linalg-backend cublas
            --frequency-parallel mpi
            --tasks-per-gpu 2
    )

    gw_add_component_compare_test(
        h2o_cublas_mpi_compare
        "${GW_REFERENCE_H2O_DIR}"
        "${test_dir}"
        1e-6
        1e-6
        h2o_cublas_mpi_compute
        "${labels}"
    )

    gw_add_energy_summary_compare_test(
        h2o_cublas_mpi_energy_summary_compare
        "${GW_REFERENCE_H2O_DIR}"
        "${test_dir}"
        1e-5
        1e-5
        h2o_cublas_mpi_compute
        "${labels}"
    )

    set_tests_properties(h2o_cublas_mpi_compute
        PROPERTIES
            PROCESSORS 4
            RESOURCE_LOCK gpu
            DEPENDS h2o_cublas_mpi_run
    )

    gw_set_test_labels("${labels}"
        h2o_cublas_mpi_run
        h2o_cublas_mpi_compute
    )
endfunction()

gw_add_h2o_reference_test()
gw_add_h2o_pyscf_test()
gw_add_h2o_blas_lapack_test()
gw_add_h2o_mpi_blas_lapack_test()
gw_add_h2o_scalapack_test()
gw_add_h2o_cosma_test()
gw_add_h2o_cublas_serial_test()
gw_add_h2o_cublas_mpi_test()
