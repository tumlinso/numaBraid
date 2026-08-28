add_executable(numabraid_header_smoke compile/header_smoke.cc)
target_link_libraries(numabraid_header_smoke PRIVATE numabraid::numabraid)
numabraid_enable_warnings(numabraid_header_smoke)

add_test(NAME numabraid.header-smoke COMMAND numabraid_header_smoke)

if(TARGET numabraid::module)
  add_executable(numabraid_module_smoke compile/module_smoke.cc)
  target_link_libraries(numabraid_module_smoke PRIVATE numabraid::module)
  numabraid_enable_warnings(numabraid_module_smoke)

  add_test(NAME numabraid.module-smoke COMMAND numabraid_module_smoke)
endif()
