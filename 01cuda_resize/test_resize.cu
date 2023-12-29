#include <gtest/gtest.h>

// Include the header file where your preprocess function is declared
#include "resize.h"
#include <opencv2/opencv.hpp>


// Define a test case
TEST(PreprocessTest1, NearestSamplerTest) {
    // set input configs
    int n_img = 1;
    int src_img_h = 720;
    int src_img_w = 1280;
    int dst_img_h = 360;
    int dst_img_w = 640;
    float resize_radio_h = 0.5;
    float resize_radio_w = 0.5;
    int crop_h = 0;
    int crop_w = 0;
    triplet mean = {0,0,0};
    triplet std = {1,1,1};
    Sampler sample = Sampler::nearest;

    //get input images
    cv::Mat img = cv::imread("../input.jpg");
    if(img.empty()){
        std::cerr<<"failed to read the image"<<std::endl;
        FAIL(); // FAIL AND TERMINATE test
    }
    int src_h = img.rows;
    int src_w = img.cols;
    int src_c = img.channels();
    uchar* src_imgs = new uchar[src_h*src_w*src_c];
    memcpy(src_imgs, img.data, src_h*src_w*src_c*sizeof(uchar));

    // set dst images
    float* dst_imgs = new float[dst_img_h*dst_img_w*src_c];
    // Call the preprocess function
    int result = preprocess(src_imgs, dst_imgs, n_img, src_img_h, src_img_w,
                             dst_img_h, dst_img_w, resize_radio_h, resize_radio_w,
                             crop_h, crop_w, mean, std, sample);
    cv::imshow("input", img);

    // prepare output image showing
    int o_img_size=dst_img_h*dst_img_w*src_c;
    uchar* uchar_data = new uchar[o_img_size];
    for(int i=0; i<o_img_size; i++){
        if (i % 5000 == 0){
            std::cout<<static_cast<uint>(src_imgs[i])<<" "<<dst_imgs[i]<<std::endl;
        }
        uchar_data[i]=static_cast<uchar>(dst_imgs[i]);
    }
    cv::Mat dst_img(dst_img_h, dst_img_w, CV_8UC(src_c));
    memcpy(dst_img.data, uchar_data, o_img_size*sizeof(uchar));
    cv::imshow("output", dst_img);
    cv::waitKey(0);

    cv::Mat res1=cv::Mat::zeros(src_h, src_w, CV_8UC3);
    memcpy(res1.data, src_imgs, src_h*src_w*src_c*sizeof(uchar));
    cv::imshow("input1", res1);
    cv::waitKey(0);
    // Add your assertions based on the expected result
    EXPECT_EQ(result, EXIT_SUCCESS);

    // Add more assertions if needed
}

// Add more test cases if necessary

// Entry point for the tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
