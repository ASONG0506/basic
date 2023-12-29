#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

__global__ void colorOffsetKernel(uchar3* input, uchar3* output, int width, int height, int offset) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * width + x;
        uchar3 pixel = input[index];
        
        // 对颜色进行
        pixel.x = 255-pixel.x;
        pixel.y = 255-pixel.y;
        pixel.z = 255-pixel.z;

        output[index] = pixel;
    }
}

void colorOffsetCUDA(const uchar3* h_input, uchar3* h_output, int width, int height, int offset) {
    int size = width * height * sizeof(uchar3);

    uchar3* d_input, *d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 blockSize(1, 1);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    colorOffsetKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, offset);

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    cv::Mat image = cv::imread("../input.jpg");
    
    if (image.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    uchar3* h_input = (uchar3*)image.data;
    uchar3* h_output = new uchar3[width * height];

    int offset = 50;  // 调整颜色偏移值

    colorOffsetCUDA(h_input, h_output, width, height, offset);

    cv::Mat output_image(height, width, CV_8UC3, h_output);
    cv::imwrite("output.jpg", output_image);

    delete[] h_output;

    return 0;
}
