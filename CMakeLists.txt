cmake_minimum_required(VERSION 3.22)
project(softgpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# SG_SANITIZE=thread|address|undefined|off
set(SG_SANITIZE "off" CACHE STRING "Sanitizer to build with")
if(NOT SG_SANITIZE STREQUAL "off")
  add_compile_options(-fsanitize=${SG_SANITIZE} -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=${SG_SANITIZE})
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Wshadow)

find_package(Threads REQUIRED)

# ---- layering ---------------------------------------------------------------
# runtime -> driver -> device. Each library only links the one below it and
# only exposes its own headers, so a violation is a build error, not a review
# comment.

add_library(sg_device STATIC src/device/device.cpp src/device/mmu.cpp)
target_include_directories(sg_device
  PUBLIC  ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(sg_device PUBLIC Threads::Threads)

add_library(sg_driver STATIC src/driver/driver.cpp src/driver/vram_alloc.cpp)
target_include_directories(sg_driver
  PUBLIC  ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(sg_driver PRIVATE sg_device)

add_library(sg_runtime STATIC src/runtime/runtime.cpp)
target_include_directories(sg_runtime
  PUBLIC  ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(sg_runtime PRIVATE sg_driver)

# ---- executables ------------------------------------------------------------
add_executable(sgbench bench/main.cpp bench/benches.cpp)
target_link_libraries(sgbench PRIVATE sg_runtime Threads::Threads)

add_executable(sgtest tests/test_runtime.cpp)
target_link_libraries(sgtest PRIVATE sg_runtime Threads::Threads)

enable_testing()
add_test(NAME runtime COMMAND sgtest)
# Same tests with a 2-slot ring: every second submit hits backpressure.
add_test(NAME runtime_ring2 COMMAND sgtest)
set_tests_properties(runtime_ring2 PROPERTIES ENVIRONMENT "SG_RING_DEPTH=2")
# Degenerate staging pool: one 64 KiB slot, so every pageable copy is many
# chunks and every chunk waits on the previous one.
add_test(NAME runtime_staging1 COMMAND sgtest)
set_tests_properties(runtime_staging1 PROPERTIES ENVIRONMENT "SG_STAGING_SLOTS=1;SG_STAGING_CHUNK=65536")
# Two copy engines: streams alternate engines, so cross-engine fences are
# exercised between copy engines as well as against compute.
add_test(NAME runtime_ce2 COMMAND sgtest)
set_tests_properties(runtime_ce2 PROPERTIES ENVIRONMENT "SG_COPY_ENGINES=2")
# Wait policies: correctness must not depend on how the host waits. A hang
# here is the classic lost-wakeup race, hence the timeouts.
add_test(NAME runtime_block COMMAND sgtest)
set_tests_properties(runtime_block PROPERTIES ENVIRONMENT "SG_WAIT_POLICY=block" TIMEOUT 120)
add_test(NAME runtime_adaptive COMMAND sgtest)
set_tests_properties(runtime_adaptive PROPERTIES ENVIRONMENT "SG_WAIT_POLICY=adaptive" TIMEOUT 120)
# Lock-free ticket submission on a shared channel, and everything on one channel.
add_test(NAME runtime_ticket COMMAND sgtest)
set_tests_properties(runtime_ticket PROPERTIES ENVIRONMENT "SG_SUBMIT_MODE=ticket" TIMEOUT 120)
add_test(NAME runtime_ch1_ticket COMMAND sgtest)
set_tests_properties(runtime_ch1_ticket PROPERTIES ENVIRONMENT "SG_CHANNELS=1;SG_SUBMIT_MODE=ticket" TIMEOUT 400) # yield-publishing is slow under TSan
# Engine idle gating: `sleep` gates the instant the runlist empties, so every
# submit races the engine's arm/re-check — the maximal lost-doorbell exposure.
add_test(NAME runtime_gated COMMAND sgtest)
set_tests_properties(runtime_gated PROPERTIES ENVIRONMENT "SG_ENGINE_IDLE=sleep;SG_WAIT_POLICY=spin" TIMEOUT 180)
add_test(NAME runtime_gated_adaptive COMMAND sgtest)
set_tests_properties(runtime_gated_adaptive PROPERTIES ENVIRONMENT "SG_ENGINE_IDLE=adaptive;SG_ENGINE_IDLE_NS=5000" TIMEOUT 180)
# Managed memory with prefetch off: every page is its own fault.
add_test(NAME runtime_prefetch0 COMMAND sgtest)
set_tests_properties(runtime_prefetch0 PROPERTIES ENVIRONMENT "SG_PREFETCH_PAGES=0")
# One-way managed memory: host writes after the first device touch stay stale.
add_test(NAME runtime_uffd0 COMMAND sgtest)
set_tests_properties(runtime_uffd0 PROPERTIES ENVIRONMENT "SG_UFFD=0")
set_tests_properties(runtime runtime_ring2 runtime_staging1 runtime_ce2 runtime_prefetch0 runtime_uffd0 PROPERTIES TIMEOUT 120)
