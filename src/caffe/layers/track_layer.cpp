// convolution_base
// Copyright (c) 2018 kurff
// Written by kurffzhou

#include <cfloat>

#include "caffe/layers/track_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void TrackLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  TrackParameter track_param = this->layer_param_.track_param();
  ROIPoolingParameter object_param = track_param.object_param();
  ROIPoolingParameter context_param = track_param.context_param();
 max_tracker_num_ = track_param.max_track_num();
 CHECK_GT(max_tracker_num_, 0) <<" the max tracker number must be >0 ";


  CHECK_GT(object_param.pooled_h(), 0)
      << "object pooled_h must be > 0";
  CHECK_GT(object_param.pooled_w(), 0)
      << "object pooled_w must be > 0";

  CHECK_GT(context_param.pooled_h(), 0)
      << "context pooled_h must be > 0";
  CHECK_GT(context_param.pooled_w(), 0)
      << "context pooled_w must be > 0";
  
  object_pooled_height_ = object_param.pooled_h();
  object_pooled_width_ = object_param.pooled_w();
  object_grid_height_ = object_pooled_height_ + 1;
  object_grid_width_ = object_pooled_width_ + 1;
  object_spatial_scale_ = object_param.spatial_scale();
  object_pad_ratio_ = object_param.pad_ratio();

  
  context_pooled_height_ = context_param.pooled_h();
  context_pooled_width_ = context_param.pooled_w();
  context_grid_height_ = context_pooled_height_ + 1;
  context_grid_width_ = context_pooled_width_ + 1;
  context_spatial_scale_ = context_param.spatial_scale();
  context_pad_ratio_ = context_param.pad_ratio();

  trackers_.reset(new Blob<Dtype>());
  contexts_.reset(new Blob<Dtype>());
  context_rois_.reset(new Blob<Dtype>());
  context_buffer_.reset(new Blob<Dtype>());

 
}



template <typename Dtype>
void TrackLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  trackers_->Reshape(max_tracker_num_, channels_, object_grid_height_, object_grid_width_);
  contexts_->Reshape(max_tracker_num_, channels_, context_grid_height_, context_grid_width_);
  context_rois_->Reshape(max_tracker_num_, 1, 1, 5);
  top[0]->Reshape(max_tracker_num_,1,1,6);

  context_buffer_shape_.clear();
  context_buffer_shape_.push_back(channels_* object_grid_height_ * object_grid_width_);

  const int output_height = height_ - object_grid_height_ - 1  ;
  const int output_width =  width_ - object_grid_width_ - 1;
  context_buffer_shape_.push_back(output_height);
  context_buffer_shape_.push_back(output_width);

  context_buffer_->Reshape(context_buffer_shape_);
  

  // top[0] [idx of batch, x0, y0, x1, y1, confidence]
  
}

template<typename Dtype>
void TrackLayer<Dtype>::roi_align_feature(const Blob<Dtype>* feat, const Blob<Dtype>* rois, const int pooled_height,
const int pooled_width, const int grid_height, const int grid_width, const Dtype spatial_scale,
const Dtype pad_ratio, Blob<Dtype>* rois_feat){
  const Dtype* bottom_data = feat->cpu_data();
  const Dtype* bottom_rois = rois->cpu_data();
  // Number of ROIs
  int num_rois = rois->num();
  int batch_size = feat->num();
  int top_count = rois_feat->count();
  Dtype* top_data = rois_feat->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind = bottom_rois[0];
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);
    
    // padding
    Dtype pad_w = (bottom_rois[3]-bottom_rois[1]+1)*pad_ratio;
    Dtype pad_h = (bottom_rois[4]-bottom_rois[2]+1)*pad_ratio;
    
    // start and end float coordinates at feature map scale
    Dtype roi_start_w = (bottom_rois[1]-pad_w) * spatial_scale;
    Dtype roi_start_h = (bottom_rois[2]-pad_h) * spatial_scale;
    Dtype roi_end_w = (bottom_rois[3]+pad_w) * spatial_scale;
    Dtype roi_end_h = (bottom_rois[4]+pad_h) * spatial_scale;
    
    // coordinate shift
    roi_start_w -= 0.5; roi_start_h -= 0.5;
    roi_end_w -= 0.5; roi_end_h -= 0.5;
       
    const Dtype roi_height = roi_end_h-roi_start_h;
    const Dtype roi_width = roi_end_w-roi_start_w;
    
    const Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
    const Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

    const Dtype* batch_data = bottom_data + feat->offset(roi_batch_ind);

    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph <= pooled_height; ++ph) {
        for (int pw = 0; pw <= pooled_width; ++pw) {
          const int pool_index = ph * grid_width + pw;
          // set zero for malformed ROIs
          if (roi_height <= 0 || roi_width <= 0) {
            top_data[pool_index] = Dtype(0);
            continue;
          }
          // float grid coordinates
          Dtype hfloat = roi_start_h + static_cast<Dtype>(ph)*bin_size_h;
          Dtype wfloat = roi_start_w + static_cast<Dtype>(pw)*bin_size_w;          
          // set zero when grid is out of feature map
          if (hfloat < -0.5 || hfloat > (height_-0.5) || 
                  wfloat < -0.5 || wfloat > (width_-0.5)) {
            top_data[pool_index] = Dtype(0);
            continue;
          }
          
          // neighboring feature coordinates
          int hfloor = floor(hfloat), wfloor = floor(wfloat);
          int hceil = hfloor+1, wceil = wfloor+1;
          
          // clipping
          hfloat = min(max(hfloat, Dtype(0)), static_cast<Dtype>(height_-1));
          wfloat = min(max(wfloat, Dtype(0)), static_cast<Dtype>(width_-1));
          hfloor = min(max(hfloor, 0), (height_-1));
          wfloor = min(max(wfloor, 0), (width_-1));
          hceil = min(max(hceil, 0), (height_-1));
          wceil = min(max(wceil, 0), (width_-1));

          // coefficients and features for bilinear interpolation
          Dtype lh = hfloat-hfloor, lw = wfloat-wfloor;
          Dtype hh = 1-lh, hw = 1-lw;
          CHECK_GE(lh,0); CHECK_LE(lh,1);
          CHECK_GE(lw,0); CHECK_LE(lw,1);
          CHECK_GE(hh,0); CHECK_LE(hh,1);
          CHECK_GE(hw,0); CHECK_LE(hw,1);
          Dtype w00 = hw*hh, w10 = lw*hh, w01 = hw*lh, w11 = lw*lh;
          
          Dtype v00 = batch_data[hfloor*width_+wfloor];
          Dtype v10 = batch_data[hfloor*width_+wceil];
          Dtype v01 = batch_data[hceil*width_+wfloor];
          Dtype v11 = batch_data[hceil*width_+wceil];
          
          // bilinear interpolation
          Dtype val = w00*v00 + w10*v10 + w01*v01 + w11*v11;
          top_data[pool_index] = val;
        }
      }
      // Increment all data pointers by one channel
      batch_data += feat->offset(0, 1);
      top_data += rois_feat[0]->offset(0, 1);
    }
    // Increment ROI data pointer
    bottom_rois += rois->offset(1);
  }
}


// bottom[0] = KxHxWxC; // previous frame feature map
// bottom[1] = Nx1x1x5; // rois  rois < max_num_tracker
// bottom[2] = KxHxWxC; // current frame feature map
// top[0] = Nx1x1x6;    // current rois
template <typename Dtype>
void TrackLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    roi_align_feature(bottom[0], bottom[1], object_pooled_height_,
    object_pooled_width_, object_grid_height_, object_grid_width_, object_spatial_scale_,
    object_pad_ratio_, trackers_.get());

    roi_align_feature(bottom[2], context_rois_.get(), context_pooled_height_, 
    context_pooled_width_, context_grid_height_, context_grid_width_, context_spatial_scale_, 
    context_pad_ratio_, contexts_.get());

    // convolution
    const int num_rois = bottom[1]->num();
    const int pad_h = 0;
    const int pad_w = 0;
    const int stride_h = 1;
    const int stride_w = 1;
    const int dilation_h = 1;
    const int dilation_w = 1;



    for(int i = 0; i < num_rois; ++ i){
        im2col_cpu(contexts_->cpu_data(), channels_, context_grid_height_,
        context_grid_width_, object_grid_height_, object_grid_width, context_buffer_->mutable_cpu_data());
        //caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, );
    }


    // get the maximum value of the response map 
    // and the corresponding corrdinates

    for(int i = 0; i < num_rios; ++ i){
        //col2im_cpu
    }

    




    









}

template <typename Dtype>
void TrackLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(TrackLayer);
#endif

INSTANTIATE_CLASS(TrackLayer);
REGISTER_LAYER_CLASS(Track);

}  // namespace caffe
