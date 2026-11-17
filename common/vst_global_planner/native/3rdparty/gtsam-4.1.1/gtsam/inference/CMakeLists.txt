# Install headers
file(GLOB inference_headers "*.h")
install(FILES ${inference_headers} DESTINATION include/gtsam/inference)

# Build tests
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  add_subdirectory(tests)
endif()
