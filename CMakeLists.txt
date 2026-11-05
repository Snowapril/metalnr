cmake_minimum_required(VERSION 3.28)

# Metal 4 APIs are macOS 26+; set before project() so the compiler sees it.
set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" CACHE STRING "Minimum macOS version")

project(cloudnr LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

file(GLOB CLOUDNR_CORE_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.mm)
list(REMOVE_ITEM CLOUDNR_CORE_SOURCES ${CMAKE_SOURCE_DIR}/src/main.mm)

add_library(cloudnr_core STATIC ${CLOUDNR_CORE_SOURCES})
target_include_directories(cloudnr_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
# Renderer version recorded in benchmark records and references (protocol §6).
execute_process(COMMAND git describe --always --dirty
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE CLOUDNR_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE CLOUDNR_GIT_RESULT ERROR_QUIET)
if(NOT CLOUDNR_GIT_RESULT EQUAL 0 OR CLOUDNR_VERSION STREQUAL "")
    set(CLOUDNR_VERSION "nogit")
endif()
target_compile_definitions(cloudnr_core PUBLIC
    CLOUDNR_SOURCE_ROOT="${CMAKE_SOURCE_DIR}"
    CLOUDNR_VERSION="${CLOUDNR_VERSION}")
target_compile_options(cloudnr_core PUBLIC
    -Wall -Wextra
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
target_link_libraries(cloudnr_core PUBLIC
    "-framework Foundation"
    "-framework Metal"
    "-framework QuartzCore"
    "-framework AppKit"
    "-framework ImageIO"
    "-framework CoreGraphics"
    "-framework UniformTypeIdentifiers")

add_executable(cloudnr src/main.mm)
target_link_libraries(cloudnr PRIVATE cloudnr_core)

enable_testing()
file(GLOB CLOUDNR_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/test_*.mm)
foreach(test_source ${CLOUDNR_TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})
    target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(${test_name} PRIVATE cloudnr_core)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
