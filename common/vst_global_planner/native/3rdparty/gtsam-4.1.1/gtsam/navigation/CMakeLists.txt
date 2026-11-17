# Install headers
file(GLOB navigation_headers "*.h")
install(FILES ${navigation_headers} DESTINATION include/gtsam/navigation)

# Add all tests
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  add_subdirectory(tests)
endif()
