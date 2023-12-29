cmake_minimum_required(VERSION 3.15)
project(CUDAResizeTest CXX)

# Set the path to the CUDA toolkit
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# Add the path to your CUDA header files
include_directories(${CUDA_INCLUDE_DIRS})

# Add the path to your Google Test installation
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# Add your source files
set(SRC_FILES
    resize.cu
    test_resize.cu
)

# Create an executable for your tests
cuda_add_executable(tests ${SRC_FILES})

# Link against Google Test and CUDA libraries
target_link_libraries(tests 
    ${GTEST_BOTH_LIBRARIES} 
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBRARIES})

# Add any other dependencies or compiler flags if needed
# target_link_libraries(tests your_other_libraries)

# Add a test target
enable_testing()
add_test(NAME ResizeTest COMMAND tests)
