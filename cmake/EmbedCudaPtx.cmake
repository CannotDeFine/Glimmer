file(READ "${INPUT}" source HEX)
if(source STREQUAL "")
    message(FATAL_ERROR "Invalid private PTX embedding input")
endif()
# Integer bytes avoid nonportable overlength string literals in clang-tidy.
string(REGEX REPLACE "(..)" "0x\\1," source "${source}")
file(WRITE "${OUTPUT}" "${source}0\n")
