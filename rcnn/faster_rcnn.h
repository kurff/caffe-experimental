#ifndef __FASTER_RCNN_H__
#define __FASTER_RCNN_H__
#include "rcnn/fast_rcnn.h"
#include <memory>

namespace rcnn{
    class FasterRCNN : public FastRCNN{
        public:
            FasterRCNN();
            ~FasterRCNN();

            bool init(const std::string& prototxt, const std::string& caffemodel, const int gpu_id);

            bool forward(cv::Mat& image, std::vector<Box>* boxes);

            bool forward(cv::Mat& image, const std::vector<Box>& prev_boxes, std::vector<Box>* curr_boxes);



        protected:



    };




}


#endif