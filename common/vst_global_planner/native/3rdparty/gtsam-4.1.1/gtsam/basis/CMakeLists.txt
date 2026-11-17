# Install headers
file(GLOB basis_headers "*.h")
install(FILES ${basis_headers} DESTINATION include/gtsam/basis)

# Build tests
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  add_subdirectory(tests)
endif()
