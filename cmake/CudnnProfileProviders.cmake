# Generate a private immutable identity table from already installed/built ELFs.
# No download, vendor SDK call, or production fixture exception is involved.
function(glimmer_write_cudnn_profile_providers output readelf)
    if(NOT EXISTS "${readelf}" OR NOT ARGN)
        message(FATAL_ERROR "cuDNN provider identity requires readelf and explicit ELF inputs")
    endif()
    set(_glimmer_entries "")
    set(_glimmer_count 0)
    set(_glimmer_seen_paths)
    foreach(_glimmer_library IN LISTS ARGN)
        if(NOT IS_ABSOLUTE "${_glimmer_library}" OR NOT EXISTS "${_glimmer_library}" OR
                _glimmer_library MATCHES "[\r\n|]")
            message(FATAL_ERROR "Invalid cuDNN provider identity path: ${_glimmer_library}")
        endif()
        list(FIND _glimmer_seen_paths "${_glimmer_library}" _glimmer_duplicate)
        if(NOT _glimmer_duplicate EQUAL -1)
            message(FATAL_ERROR "Duplicate cuDNN provider identity path: ${_glimmer_library}")
        endif()
        list(APPEND _glimmer_seen_paths "${_glimmer_library}")
        execute_process(COMMAND "${readelf}" -d "${_glimmer_library}"
            RESULT_VARIABLE _glimmer_dynamic_status OUTPUT_VARIABLE _glimmer_dynamic
            ERROR_VARIABLE _glimmer_dynamic_error TIMEOUT 15)
        execute_process(COMMAND "${readelf}" -SW "${_glimmer_library}"
            RESULT_VARIABLE _glimmer_notes_status OUTPUT_VARIABLE _glimmer_notes
            ERROR_VARIABLE _glimmer_notes_error TIMEOUT 15)
        if(NOT _glimmer_dynamic_status EQUAL 0 OR NOT _glimmer_notes_status EQUAL 0 OR
                NOT _glimmer_dynamic MATCHES "\\(SONAME\\)[^\n]*\\[([^]]+)\\]")
            message(FATAL_ERROR "cuDNN provider needs a valid ELF SONAME: ${_glimmer_library}")
        endif()
        set(_glimmer_soname "${CMAKE_MATCH_1}")
        # Wheel packaging can move this allocated section without repairing
        # PT_NOTE. Bind its actual mapped address and bounded GNU note bytes.
        if(NOT _glimmer_notes MATCHES
                "\\.note\\.gnu\\.build-id[ \t]+NOTE[ \t]+([0-9a-fA-F]+)[ \t]+([0-9a-fA-F]+)[ \t]+([0-9a-fA-F]+)[ \t]")
            message(FATAL_ERROR "cuDNN provider needs an allocated GNU build ID section: ${_glimmer_library}")
        endif()
        set(_glimmer_note_address "${CMAKE_MATCH_1}")
        math(EXPR _glimmer_note_offset "0x${CMAKE_MATCH_2}")
        math(EXPR _glimmer_note_size "0x${CMAKE_MATCH_3}")
        if(_glimmer_note_size LESS 20 OR _glimmer_note_size GREATER 80)
            message(FATAL_ERROR "Unbounded cuDNN GNU build ID note: ${_glimmer_library}")
        endif()
        file(READ "${_glimmer_library}" _glimmer_note OFFSET ${_glimmer_note_offset}
            LIMIT ${_glimmer_note_size} HEX)
        string(TOLOWER "${_glimmer_note}" _glimmer_note)
        if(NOT _glimmer_note MATCHES
                "^04000000([0-9a-f][0-9a-f])00000003000000474e5500")
            message(FATAL_ERROR "Expected a bounded little-endian GNU build ID note: ${_glimmer_library}")
        endif()
        math(EXPR _glimmer_id_bytes "0x${CMAKE_MATCH_1}")
        math(EXPR _glimmer_expected_note_bytes "16 + ((${_glimmer_id_bytes} + 3) / 4) * 4")
        if(_glimmer_id_bytes LESS 1 OR _glimmer_id_bytes GREATER 64 OR
                NOT _glimmer_note_size EQUAL _glimmer_expected_note_bytes)
            message(FATAL_ERROR "Invalid cuDNN GNU build ID note size: ${_glimmer_library}")
        endif()
        math(EXPR _glimmer_id_hex_bytes "${_glimmer_id_bytes} * 2")
        string(SUBSTRING "${_glimmer_note}" 32 ${_glimmer_id_hex_bytes} _glimmer_build_id)
        math(EXPR _glimmer_padding_offset "32 + ${_glimmer_id_hex_bytes}")
        string(SUBSTRING "${_glimmer_note}" ${_glimmer_padding_offset} -1 _glimmer_padding)
        if(_glimmer_padding MATCHES "[^0]")
            message(FATAL_ERROR "Noncanonical cuDNN GNU note padding: ${_glimmer_library}")
        endif()
        foreach(_glimmer_text _glimmer_library _glimmer_soname)
            string(REPLACE "\\" "\\\\" ${_glimmer_text} "${${_glimmer_text}}")
            string(REPLACE "\"" "\\\"" ${_glimmer_text} "${${_glimmer_text}}")
        endforeach()
        string(APPEND _glimmer_entries
            "    ProviderIdentity{\"${_glimmer_soname}\", \"${_glimmer_build_id}\", \"${_glimmer_library}\", 0x${_glimmer_note_address}ULL, ${_glimmer_note_size}},\n")
        math(EXPR _glimmer_count "${_glimmer_count} + 1")
    endforeach()
    get_filename_component(_glimmer_directory "${output}" DIRECTORY)
    file(MAKE_DIRECTORY "${_glimmer_directory}")
    set(_glimmer_header "#pragma once\n#include <array>\n#include <cstddef>\n#include <cstdint>\n#include <string_view>\nnamespace glimmer::internal::cudnn_profile {\nstruct ProviderIdentity { std::string_view soname, build_id, path; std::uintptr_t note_address; std::size_t note_bytes; };\nstatic constexpr std::array<ProviderIdentity, ${_glimmer_count}> kCudnnProfileProviders{{\n${_glimmer_entries}}};\n}\n")
    # Avoid recompiling the frontend merely because configure ran again.
    if(EXISTS "${output}")
        file(READ "${output}" _glimmer_old_header)
    endif()
    if(NOT _glimmer_header STREQUAL _glimmer_old_header)
        file(WRITE "${output}" "${_glimmer_header}")
    endif()
endfunction()

if(DEFINED GLIMMER_PROFILE_OUTPUT AND DEFINED GLIMMER_PROFILE_INPUTS)
    string(REPLACE "|" ";" _glimmer_profile_inputs "${GLIMMER_PROFILE_INPUTS}")
    glimmer_write_cudnn_profile_providers("${GLIMMER_PROFILE_OUTPUT}"
        "${GLIMMER_PROFILE_READELF}" ${_glimmer_profile_inputs})
endif()
