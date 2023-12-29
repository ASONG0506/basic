google test的简单使用

```
mkdir build && cd build
cmake ..
make 
./test_math_functions
```

输出结果

```
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from MathFunctionsTest
[ RUN      ] MathFunctionsTest.AddTestaaa
[       OK ] MathFunctionsTest.AddTestaaa (0 ms)
[----------] 1 test from MathFunctionsTest (0 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (0 ms total)
[  PASSED  ] 1 test.
```