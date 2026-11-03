if(NOT DEFINED GLIMMER_COMPILE_DATABASE OR NOT DEFINED GLIMMER_COMPILE_LINK)
    message(FATAL_ERROR "The compile database and link paths are required.")
endif()

if(IS_SYMLINK "${GLIMMER_COMPILE_LINK}")
    file(REMOVE "${GLIMMER_COMPILE_LINK}")
elseif(EXISTS "${GLIMMER_COMPILE_LINK}")
    message(FATAL_ERROR "Refusing to replace a non-symlink compile database: ${GLIMMER_COMPILE_LINK}")
endif()

file(CREATE_LINK "${GLIMMER_COMPILE_DATABASE}" "${GLIMMER_COMPILE_LINK}"
    SYMBOLIC RESULT link_result)
if(NOT link_result STREQUAL "0")
    message(FATAL_ERROR "Cannot update compile database link: ${link_result}")
endif()
