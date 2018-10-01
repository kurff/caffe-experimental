#include "rcnn/faster_rcnn.h"

using namespace caffe;
namespace rcnn{
    FasterRCNN::FasterRCNN() : RCNN(){

    }

    FasterRCNN::~FasterRCNN(){

    }

    

    bool FasterRCNN::forward(cv::Mat& image, std::vector<Box>* boxes){
        Blob<float>* input_layer = net_->input_blobs()[0];
        input_layer->Reshape(1, num_channels_,input_geometry_.height, input_geometry_.width);
            /* Forward dimension change to all layers. */
        net_->Reshape();

        std::vector<cv::Mat> input_channels;
        WrapInputLayer(&input_channels);

        Preprocess(img, &input_channels);

        net_->ForwardPrefilled();

            /* Copy the output layer to a std::vector */
        Blob<float>* output_layer = net_->output_blobs()[0];
        const float* begin = output_layer->cpu_data();


    }

    bool FasterRCNN::forward(cv::Mat& image, const std::vector<Box>& prev_boxes, std::vector<Box>* curr_boxes){


    }

    
} // end of namespace rcnn