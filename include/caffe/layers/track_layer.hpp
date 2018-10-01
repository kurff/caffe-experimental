// ------------------------------------------------------------------
// TrackLayer
// Copyright (c) 2018 kurff
// Written by kurffzhou


#ifndef CAFFE_TRACK_LAYER_HPP_
#define CAFFE_TRACK_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class TrackLayer : public Layer<Dtype> {
 public:
  explicit TrackLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Track"; }

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void roi_align_feature(const Blob<Dtype>* feat, const Blob<Dtype>* rois, const int pooled_height,
const int pooled_width, const int grid_height, const int grid_width, const Dtype spatial_scale,
const Dtype pad_ratio, Blob<Dtype>* rois_feat);



  int channels_;
  int height_;
  int width_;

  // roi align to extract object features

  int object_pooled_height_;
  int object_pooled_width_;
  int object_grid_height_;
  int object_grid_width_;
  Dtype object_spatial_scale_;
  Dtype object_pad_ratio_;

  // roi align to extract context features
  int context_pooled_height_;
  int context_pooled_width_;
  int context_grid_height_;
  int context_grid_width_;

  Dtype context_spatial_scale_;
  Dtype context_pad_ratio_;


  int max_tracker_num_;
  shared_ptr< Blob<Dtype> > trackers_;
  shared_ptr< Blob<Dtype> > contexts_;

  shared_ptr< Blob<Dtype> > context_rois_;
  shared_ptr< Blob<Dtype> > context_buffer_;
  vector<int> context_buffer_shape_;

  









};

}  // namespace caffe

#endif  // CAFFE_ROI_ALIGN_LAYER_HPP_
