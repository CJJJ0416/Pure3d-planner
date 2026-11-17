# Install headers
file(GLOB symbolic_headers "*.h")
install(FILES ${symbolic_headers} DESTINATION include/gtsam/symbolic)

# Build tests
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  add_subdirectory(tests)
endif()
