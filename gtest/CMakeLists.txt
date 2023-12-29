# CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(MathFunctionsTest CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)

# 添加可执行测试文件
add_executable(test_math_functions test_math_functions.cpp)

# 启用 Google Test
find_package(GTest REQUIRED)
target_include_directories(test_math_functions PRIVATE ${GTEST_INCLUDE_DIRS})
target_link_libraries(test_math_functions PRIVATE ${GTEST_BOTH_LIBRARIES})

# 添加链接 math_functions.h 文件
target_include_directories(test_math_functions PRIVATE ${CMAKE_SOURCE_DIR})
