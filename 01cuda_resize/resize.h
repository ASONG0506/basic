#ifndef __PREPROCESS_H__
#define __PREPROCESS_H__

typedef unsigned char uchar;

struct triplet{
    float x;
    float y;
    float z;
};

enum class Sampler{
    nearest,
    bicubic
};

int preprocess(const uchar* src_imgs, float* dst_imgs, int n_img, int src_img_h,
                int src_img_w, int dst_img_h, int dst_img_w, float resize_radio_h, 
                float resize_radio_w, int crop_h, int crop_w, triplet mean, 
                triplet std, Sampler sample);


#endif