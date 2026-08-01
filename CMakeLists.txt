cmake_minimum_required(VERSION 3.20)

project(Glimmer VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "Glimmer supports Linux only.")
endif()

include(cmake/ProjectOptions.cmake)
include(cmake/ProjectTesting.cmake)

add_subdirectory(3rdparty)
add_subdirectory(src)

find_program(GLIMMER_CLANG_FORMAT NAMES clang-format)

if(GLIMMER_CLANG_FORMAT)
    file(GLOB_RECURSE GLIMMER_FORMAT_SOURCES CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/include/*.h"
        "${PROJECT_SOURCE_DIR}/include/*.hpp"
        "${PROJECT_SOURCE_DIR}/src/*.cc"
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/*.cu"
        "${PROJECT_SOURCE_DIR}/tests/*.cc"
        "${PROJECT_SOURCE_DIR}/tests/*.cpp"
        "${PROJECT_SOURCE_DIR}/tests/*.cu"
    )

    if(GLIMMER_FORMAT_SOURCES)
        add_custom_target(format
            COMMAND ${GLIMMER_CLANG_FORMAT} -i ${GLIMMER_FORMAT_SOURCES}
            COMMENT "Formatting Glimmer source files"
            VERBATIM
        )

        add_custom_target(format-check
            COMMAND ${GLIMMER_CLANG_FORMAT} --dry-run --Werror ${GLIMMER_FORMAT_SOURCES}
            COMMENT "Checking Glimmer source formatting"
            VERBATIM
        )
    endif()
else()
    message(STATUS "clang-format not found; format targets are unavailable")
endif()
