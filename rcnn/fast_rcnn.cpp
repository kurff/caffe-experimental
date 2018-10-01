
#include "rcnn/fast_rcnn.h"
namespace rcnn{

    FastRCNN::FastRCNN() : RCNN(){

    }
    FastRCNN::~FastRCNN() {

    }

    bool FastRCNN::init(const std::string& prototxt, const std::string& caffemodel, const int gpu_id){
        if(gpu_id !=-1){
            Caffe::set_mode(Caffe::GPU);
            Caffe::SetDevice(gpu_id);
        }
        else{
            Caffe::set_mode(Caffe::CPU);
        }
        net_.reset(new Net<float>(prototxt, TEST));
        net_->CopyTrainedLayersFrom(caffemodel);
        CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
        CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";
        Blob<float>* input_layer = net_->input_blobs()[0];
        num_channels_ = input_layer->channels();
        CHECK(num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";
        input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    }

    void FastRCNN::WrapInputLayer(std::vector<cv::Mat>* input_channels){
        Blob<float>* input_layer = net_->input_blobs()[0];
        int width = input_layer->width();
        int height = input_layer->height();
        float* input_data = input_layer->mutable_cpu_data();
        for (int i = 0; i < input_layer->channels(); ++i) {
            cv::Mat channel(height, width, CV_32FC1, input_data);
            input_channels->push_back(channel);
            input_data += width * height;
        }
    }

    
    bool FastRCNN::forward(cv::Mat& image, std::vector<Box>* boxes){


    }

    bool FastRCNN::forward(cv::Mat& image, const std::vector<Box>& prev_boxes, std::vector<Box>* curr_boxes){


    }


    void FastRCNN::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) {
            /* Convert the input image to the input image format of the network. */
        cv::Mat sample;
        if (img.channels() == 3 && num_channels_ == 1){
            cv::cvtColor(img, sample, CV_BGR2GRAY);
        }
        else if (img.channels() == 4 && num_channels_ == 1){
            cv::cvtColor(img, sample, CV_BGRA2GRAY);
        }
        else if (img.channels() == 4 && num_channels_ == 3){
            cv::cvtColor(img, sample, CV_BGRA2BGR);
        }
        else if (img.channels() == 1 && num_channels_ == 3){
            cv::cvtColor(img, sample, CV_GRAY2BGR);
        }
        else{
            sample = img;
        }
        cv::Mat sample_resized;
        if (sample.size() != input_geometry_){
            cv::resize(sample, sample_resized, input_geometry_);
        }
        else{
            sample_resized = sample;
        }
        cv::Mat sample_float;
        if (num_channels_ == 3){
            sample_resized.convertTo(sample_float, CV_32FC3);
        }
        else{
            sample_resized.convertTo(sample_float, CV_32FC1);
        }
        cv::Mat sample_normalized;
        cv::Mat mean_BGR(sample_float.rows, sample_float.cols, CV_32FC3, cv::Scalar(104, 117, 123));
        cv::subtract(sample_float, mean_BGR, sample_normalized);
        cv::split(sample_normalized, *input_channels);
        CHECK(reinterpret_cast<float*>(input_channels->at(0).data)== net_->input_blobs()[0]->cpu_data())
            << "Input channels are not wrapping the input layer of the network.";
    }

}