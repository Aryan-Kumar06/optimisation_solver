add_library(miqp_engine
    src/branch_and_bound.cpp
    src/feasibility.cpp
    ${CMAKE_SOURCE_DIR}/src/model/model.cpp
)

target_include_directories(miqp_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(miqp_engine PUBLIC qp_engine)
target_compile_features(miqp_engine PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(miqp_engine PRIVATE /W4 /permissive-)
else()
    target_compile_options(miqp_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()
add_executable(test_miqp tests/test_miqp.cpp)
target_link_libraries(test_miqp PRIVATE miqp_engine)
add_test(NAME test_miqp COMMAND test_miqp)

