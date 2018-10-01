#ifndef __FAST_RCNN_H__
#define __FAST_RCNN_H__
#include "rcnn/rcnn.h"
namespace rcnn{
    class FastRCNN : public RCNN{
        public:
            FastRCNN();
            ~FastRCNN();


            bool init(const std::string& prototxt, const std::string& caffemodel, const int gpu_id);

            // input 
            bool forward(cv::Mat& image, std::vector<Box>* boxes);

            bool forward(cv::Mat& image, const std::vector<Box>& prev_boxes, std::vector<Box>* curr_boxes);

        protected:

            void WrapInputLayer(std::vector<cv::Mat>* input_channels);
            void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

        protected:
            std::shared_ptr<caffe::Net<float> > net_;
            cv::Size input_geometry_;

    };


}// end of namespace rcnn



#endif