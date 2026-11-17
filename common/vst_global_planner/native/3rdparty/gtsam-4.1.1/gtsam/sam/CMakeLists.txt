# Install headers
file(GLOB sam_headers "*.h")
install(FILES ${sam_headers} DESTINATION include/gtsam/sam)

# Build tests
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  add_subdirectory(tests)
endif()
