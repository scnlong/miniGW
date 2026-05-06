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

add_executable(gw src/main.cpp)
target_link_libraries(gw PRIVATE gwcore)

option(GW_ENABLE_TESTS "Enable CTest-based regression tests" ON)
if(GW_ENABLE_TESTS)
    enable_testing()

    find_package(Python3 COMPONENTS Interpreter REQUIRED)

	set(GW_REFERENCE_H2O_DIR
        ${CMAKE_CURRENT_SOURCE_DIR}/regression_tests/reference/h2o_serial
    )

	set(GW_TEST_H2O_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/regression_tests/output/h2o_serial
    )

	#set(PYSCF_REFERENCE_H2O_DIR
	#    ${CMAKE_CURRENT_SOURCE_DIR}/regression_tests/reference/h2o_pyscf
	#)

	#set(PYSCF_TEST_H2O_DIR
	#    ${CMAKE_CURRENT_BINARY_DIR}/regression_tests/output/h2o_pyscf
	#)

	#add_test(
	#    NAME pyscf_h2o_reference_run
	#    COMMAND
	#    ${CMAKE_COMMAND} -E make_directory ${PYSCF_TEST_H2O_DIR}
	#)

    add_test(
        NAME serial_h2o_reference_run
        COMMAND
		${CMAKE_COMMAND} -E make_directory ${GW_TEST_H2O_DIR}
    )

	#add_test(
    #    NAME pyscf_h2o_reference_compute
    #    COMMAND
    #        gw
	#		--input-dir ${PYSCF_REFERENCE_H2O_DIR}
    #        --freq-points 200
    #        --pade-params 16
    #        --state 5
	#		--output-dir ${PYSCF_TEST_H2O_DIR}
    #)

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

	#add_test(
    #    NAME pyscf_h2o_reference_compare
    #    COMMAND
    #        ${Python3_EXECUTABLE}
    #        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_reference.py
	#		--reference-file ${PYSCF_REFERENCE_H2O_DIR}/pyscf_E_c_before_Pade.out
	#		--output-file ${PYSCF_TEST_H2O_DIR}/E_c_before_Pade.out
    #        --rtol 1e-2
    #        --atol 1e-2
    #        --compare-mode components
    #)

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

	#set_tests_properties(
    #    pyscf_h2o_reference_compute
    #    PROPERTIES
    #        DEPENDS pyscf_h2o_reference_run
    #)

    #set_tests_properties(
    #    pyscf_h2o_reference_compare
    #    PROPERTIES
    #        DEPENDS pyscf_h2o_reference_compute
    #)

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
	endif()
