if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/spdlog/CMakeLists.txt")
    message(FATAL_ERROR
        "spdlog is missing. Run: git submodule update --init --recursive"
    )
endif()

add_subdirectory(spdlog EXCLUDE_FROM_ALL)
