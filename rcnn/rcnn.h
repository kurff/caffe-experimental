#ifndef __RCNN_H__
#define __RCNN_H__
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "rcnn/box.h"
#include "caffe/caffe.hpp"
using namespace cv;
namespace rcnn{

    class RCNN{
        public:
            RCNN(){

            }
            ~RCNN(){

            }
            virtual bool init(const std::string& prototxt, const std::string& caffemodel, const int gpu_id)  = 0;
            virtual bool forward(cv::Mat& image, std::vector<Box>* boxes) = 0;

            // can use boxes from previous frames as proposals to generate curr_boxes
            virtual bool forward(cv::Mat& image, const std::vector<Box>& prev_boxes, std::vector<Box>* curr_boxes) = 0;

        


    };



}


#endif