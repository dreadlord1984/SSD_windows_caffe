// Xuanyi Dong Add Copy Faster R-CNN 
// 2016.01.06
#ifndef CAFFE_SMOOTH_L1_LAYER2_HPP_
#define CAFFE_SMOOTH_L1_LAYER2_HPP_

#include <cfloat>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class SmoothL1LossDLayer : public LossLayer<Dtype> {
public:
    explicit SmoothL1LossDLayer(const LayerParameter& param)
        : LossLayer<Dtype>(param), diff_() {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "SmoothL1Loss2"; }

    virtual inline int ExactNumBottomBlobs() const { return -1; }
    virtual inline int MinBottomBlobs() const { return 2; }
    virtual inline int MaxBottomBlobs() const { return 4; }
    /**
     * Unlike most loss layers, in the SmoothL1LossDLayer we can backpropagate
     * to both inputs -- override to return true and always allow force_backward.
     */
    virtual inline bool AllowForceBackward(const int bottom_index) const {
        return true;
    }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    Blob<Dtype> diff_;
    Blob<Dtype> errors_;
    Blob<Dtype> ones_;
    bool has_weights_;
    Dtype sigma2_;
}; 

}

#endif // CAFFE_SMOOTH_L1_LAYER2_HPP_
