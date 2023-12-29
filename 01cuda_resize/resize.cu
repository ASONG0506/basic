#include <thrust/fill.h>
#include "resize.h"

// resize, crop, norm
// sample : Nearest
__global__ void preprocess_nearest_kernel(const uchar* __restrict__ src_dev, 
                                    float* __restrict__ dst_dev, int src_row_step, 
                                    int dst_row_step, int src_img_step, int dst_img_step,
                                    int src_h, int src_w, float radio_h, float radio_w, 
                                    float offset_h, float offset_w, triplet mean, triplet std){
    if (blockIdx.x==0){
        printf("input size is ");
        printf("totally %d inputs\n", blockIdx.x);
    }
	int i = blockIdx.x;
	int j = blockIdx.y;
    int k = threadIdx.x;

	int pX = (int) roundf((i / radio_h) + offset_h);
	int pY = (int) roundf((j / radio_w) + offset_w);
    if (blockIdx.x==0){
        printf("px: %d, py: %d\n", pX, pY);
    }
 
	if (pX < src_h && pX >= 0 && pY < src_w && pY >= 0){
        int s1 = k * src_img_step + 0 * src_img_step / 3 + pX * src_row_step + pY;
        int s2 = k * src_img_step + 1 * src_img_step / 3 + pX * src_row_step + pY;
        int s3 = k * src_img_step + 2 * src_img_step / 3 + pX * src_row_step + pY;

        int d1 = k * dst_img_step + 0 * dst_img_step / 3 + i * dst_row_step + j;
        int d2 = k * dst_img_step + 1 * dst_img_step / 3 + i * dst_row_step + j;
        int d3 = k * dst_img_step + 2 * dst_img_step / 3 + i * dst_row_step + j;

        // 使用printf设备函数
    if (blockIdx.x==0){
        printf("s1: %d, s2: %d, s3: %d, d1: %d, d2: %d, d3: %d\n", s1, s2, s3, d1, d2, d3);
    }
		*(dst_dev + d1) = ((float)*(src_dev + s1) - mean.x) / std.x;
		*(dst_dev + d2) = ((float)*(src_dev + s2) - mean.y) / std.y;
		*(dst_dev + d3) = ((float)*(src_dev + s3) - mean.z) / std.z;
	}
}

__global__ void pixel_invert_kernel(uchar* src_imgs, int src_img_h, int src_img_w,
                                    int src_img_step, int src_row_step){
    int i=blockIdx.x;
    int j=blockIdx.y;
    int k=threadIdx.x;
    int s1 = 0 * src_img_step/3 + i * src_row_step + j;
    int s2 = 1 * src_img_step/3 + i * src_row_step + j;
    int s3 = 2 * src_img_step/3 + i * src_row_step + j;
    src_imgs[s1] += 100;
    src_imgs[s2] += 100;
    src_imgs[s3] += 100;
    // *(src_imgs+s1) = static_cast<uchar>(static_cast<int>(*(src_imgs+s1))-100);
    // *(src_imgs+s2) = static_cast<uchar>(static_cast<int>(*(src_imgs+s1))-100);
    // *(src_imgs+s3) = static_cast<uchar>(static_cast<int>(*(src_imgs+s1))+1);
}


int preprocess( uchar* src_imgs, float* dst_imgs, int n_img, int src_img_h,
                int src_img_w, int dst_img_h, int dst_img_w, float resize_radio_h, 
                float resize_radio_w, int crop_h, int crop_w, triplet mean, 
                triplet std, Sampler sample){
    /*
    src_imgs : 6 * 3 * src_img_h * src_img_w
    dst_imgs : 6 * 3 * dst_img_h * dst_img_w
    crop_h : resize后的图像，纵向自上裁剪范围
    crop_w : 为0
    */


    int src_row_step = src_img_w;
    int dst_row_step = dst_img_w;
    int src_img_step = src_img_w * src_img_h * 3;
    int dst_img_step = dst_img_w * dst_img_h * 3;
    float offset_h = crop_h / resize_radio_h;
    float offset_w = crop_w / resize_radio_w;

    dim3 grid(dst_img_h, dst_img_w);
    dim3 block;
    
    block = dim3(n_img);
    printf("sampler : nearest\n");
    printf("grid dim %d, %d, %d\n", grid.x, grid.y, grid.z);
    printf("block dim %d, %d, %d\n", block.x, block.y, block.z);
    printf("%d, %d, %f, %f\n", src_img_step, dst_img_step, resize_radio_h, resize_radio_w);
    preprocess_nearest_kernel<<<grid, block>>>(src_imgs, dst_imgs, src_row_step, dst_row_step, 
                    src_img_step, dst_img_step, src_img_h, src_img_w, resize_radio_h,
                    resize_radio_w, offset_h, offset_w, mean, std);
    printf("--------");
    pixel_invert_kernel<<<grid, block>>>(src_imgs, src_img_h, src_img_w, src_img_step, src_row_step);
    return EXIT_SUCCESS;
}
