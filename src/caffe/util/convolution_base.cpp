// convolution_base
// Copyright (c) 2018 kurff
// Written by kurffzhou


#include "caffe/util/convolution_base.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
using namespace std;
namespace caffe{
    // init memory and shape
    // the memory should be NxCxHxW
    template<typename Dtype>
    void ConvolutionBase::init(const Blob<Dtype>* feat, const Blob<Dtype>* filter, Blob<Dtype>* response){
        
        num_spatial_axis_ = 2;
        kernel_shape_.Reshape(vector<int>{1,num_spatial_axis_});
        int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
        for(int i = 0; i < num_spatial_axis_; ++ i){
            kernel_shape_data[i] = filter->shape(i+3);
        }
        stride_.Reshape(vector<int>{1,2});
        int* stride_data = stride_.mutable_cpu_data();
        stride_data[0] = conv_base_param_.stride_h();
        stride_data[1] = conv_base_param_.stride_w();
        pad_.Reshape(vector<int>{1,2});
        int* pad_data = pad_.mutable_cpu_data();
        pad_data[0] = conv_base_param_.pad_h();
        pad_data[1] = conv_base_param_.pad_w();

        col_buffer_.reset(new Blob<Dtype>());
        kernel_dim_ = filter->count(1);
        col_buffer_shape_.clear();
        
        col_buffer_shape_.push_back(kernel_dim_);
        for(int i = 0; i < num_spatial_axis_; ++ i){
            col_buffer_shape_.push_back(output_shape_[i]);
        }
        col_buffer_->Reshape(col_buffer_shape_);
        compute_output_shape(feat);
        response->Reshape(output_shape_);

        channel_axis_ = 1;

        conv_input_shape_.Reshape(vector<int>{1, num_spatial_axis_+1});
        int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
        for(int i = 0 ; i < num_spatial_axis_+ 1; ++ i){
            conv_input_shape_data[i] = feat->shape(channel_axis_ + i);
        }
    }

    template<typename Dtype>
    void ConvolutionBase::compute_output_shape(const Blob<Dtype>* feat){
        const int* kernel_shape_data = kernel_shape_.cpu_data();
        const int* stride_data = stride_.cpu_data();
        const int* pad_data = pad_.cpu_data();
        output_shape_.clear();
        for (int i = 0; i < num_spatial_axis_; ++i) {
            const int input_dim = feat->shape(i +  2);
            const int output_dim = (input_dim + 2 * pad_data[i] - kernel_shape_data[i]) / stride_data[i] + 1;
            output_shape_.push_back(output_dim);
        }

    }

    template<typename Dtype>
    void ConvolutionBase::forward_cpu(const Blob<Dtype>* feat, const Blob<Dtype>* filter, Blob<Dtype>* response){
        // first im2col
        const Dtype* data = feat->cpu_data();
        int in_channels = feat->shape(channel_axis_);
        const Dtype* col_buff = col_buffer_->cpu_data();
        Dtype* output = response->mutable_cpu_data();
        const Dtype* weight = filter->cpu_data();
        im2col_cpu(data, in_channels,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1], 1, 1, col_buff);
        
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ , conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weight, col_buff,
        (Dtype)0., output );





    }

    template<typename Dtype>
    void ConvolutionBase::reshape(const Blob<Dtype>* feat, const Blob<Dtype>* filter, Blob<Dtype>* response){

    }


    template<typename Dtype>
    void ConvolutionBase::backward_cpu(){




    }

    INSTANTIATE_CLASS(ConvolutionBase);


    





} // end of namespace kurff