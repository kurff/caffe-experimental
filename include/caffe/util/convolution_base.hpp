// convolution_base
// Copyright (c) 2018 kurff
// Written by kurffzhou

#ifndef CAFFE_UTIL_CONVOLUTION_BASE_HPP_
#define CAFFE_UTIL_CONVOLUTION_BASE_HPP_
#include "caffe/caffe.hpp"
#include <vector>
#include <memory>
#include "caffe/proto/caffe.pb.h"

namespace caffe{
    template<typename Dtype>
    class ConvolutionBase{
        public:
            ConvolutionBase(const ConvolutionBaseParameter& conv_base_param) : conv_base_param_(conv_base_param){

            }
            ~ConvolutionBase(){

            }

            void init(const Blob<Dtype>* feat, const Blob<Dtype>* filter, Blob<Dtype>* response);

            void reshape(const Blob<Dtype>* feat, const Blob<Dtype>* filter, Blob<Dtype>* response);

            void forward_cpu(const Blob<Dtype>* feat, const Blob<Dtype>* filter, Blob<Dtype>* response);
            
            void backward_cpu();

        protected:
            void compute_output_shape(const Blob<Dtype>* feat);

            

        protected:
            std::shared_ptr<Blob<Dtype> > col_buffer_;
            Blob<int> kernel_shape_;
            Blob<int> stride_;
            Blob<int> pad_;
            Blob<int> conv_input_shape_;
            std::vector<int> col_buffer_shape_;
            std::vector<int> output_shape_;
            ConvolutionBaseParameter conv_base_param_;
            int num_spatial_axis_;
            int kernel_dim_;
            int channel_axis_;




    };
}


#endif