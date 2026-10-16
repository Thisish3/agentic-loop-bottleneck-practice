cmake_minimum_required(VERSION 3.16)
project(kv_sim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

# -DKV_SIM_SANITIZE=thread|address builds everything with TSan/ASan.
set(KV_SIM_SANITIZE "" CACHE STRING "Sanitizer to enable (thread or address)")
if(KV_SIM_SANITIZE)
  add_compile_options(-fsanitize=${KV_SIM_SANITIZE} -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=${KV_SIM_SANITIZE})
endif()

find_package(Threads REQUIRED)
add_library(kvsim_core src/sim.cpp)
target_include_directories(kvsim_core PUBLIC src)

add_executable(kv_sim src/main.cpp)
target_link_libraries(kv_sim kvsim_core Threads::Threads)

enable_testing()
add_executable(test_kv tests/test_kv.cpp)
target_compile_options(test_kv PRIVATE -UNDEBUG)  # keep assert() active in Release
target_link_libraries(test_kv kvsim_core)
add_test(NAME test_kv COMMAND test_kv)
