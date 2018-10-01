#ifndef __MASK_RCNN_H__
#define __MASK_RCNN_H__
#include "rcnn/faster_rcnn.h"
namespace rcnn{
    class MaskRCNN : public FasterRCNN{
        public:
            MaskRCNN() : FasterRCNN(){

            }
            ~MaskRCNN(){

            }
        
        public:
            bool forward(cv::Mat& image, std::vector<Box>* boxes);

            bool forward(cv::Mat& image, const std::vector<Box>& prev_boxes, std::vector<Box>* curr_boxes);

        protected:


    };



}


#endif