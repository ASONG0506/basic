// test_math_functions.cpp
#include "gtest/gtest.h"
#include "math_functions.h"

// 测试 add 函数
TEST(MathFunctionsTest, AddTestaaa) {
    EXPECT_EQ(add(2, 3), 5);
    EXPECT_EQ(add(-2, 3), 1);
    EXPECT_EQ(add(0, 0), 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
