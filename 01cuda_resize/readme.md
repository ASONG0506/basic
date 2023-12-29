测试cuda_resize方法实现结果
```
mkdir build && cd build
cmake ..
make 
./tests
```

2. 使用cuda进行颜色翻转
```
cd build && cmake .. && make
./main
```
![origin](./input.jpg)
![result](./output.jpg)