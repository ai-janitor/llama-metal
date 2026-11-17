set(TARGET llama-phase-timing)

# This small completion driver owns the L0 receipt boundary for a bounded run.
add_executable(${TARGET} phase-timing.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Pure CPU fixture for the sampling seam; it loads no model or backend.
add_executable(llama-phase-timing-sampling-config-fixture ../../tests/phase_timing_sampling_config_fixture.cpp)
target_include_directories(llama-phase-timing-sampling-config-fixture PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(llama-phase-timing-sampling-config-fixture PRIVATE cxx_std_17)
