/**********************************************************************************************

  * @file   frcnn_param.hpp

  * @brief Thisis a brief description.

  * @date   2017:11:29 

  * @written by xuanyuyt

  * @version <version  1>

 *********************************************************************************************/ 
#ifndef CAFFE_FRCNN_PRARM_HPP_
#define CAFFE_FRCNN_PRARM_HPP_

#include <vector>
#include <string>

namespace caffe{

namespace Frcnn {

class FrcnnParam {
public:
  // ======================================== Train
  // 训练图片信息
	static int im_height;
	static int im_width;
	static int scale_layer_num;
	// rois训练参数
	static int batch_size;
  static float fg_fraction;
  static float fg_thresh;
  // Overlap threshold for a ROI to be considered background (class = 0
  // ifoverlap in [LO, HI))
  static float bg_thresh_hi;
  static float bg_thresh_lo;

  // Normalize the targets (subtract empirical mean, divide by empirical stddev)
  static bool bbox_normalize_targets;
  static float bbox_inside_weights[4];
  static float bbox_normalize_means[4];
  static float bbox_normalize_stds[4];

  // RPN to detect objects
  // If an anchor statisfied by positive and negative conditions set to negative
  static float rpn_nms_thresh;
  static int rpn_pre_nms_top_n;
  static int rpn_post_nms_top_n;

	static float test_rpn_nms_thresh;
	static int test_rpn_pre_nms_top_n;
	static int test_rpn_post_nms_top_n;

  // Proposal height and width both need to be greater than RPN_MIN_SIZE (at
  // orig image scale)
  static float rpn_min_size;
	static float test_rpn_min_size;
  // Deprecated (outside weights)
  static float rpn_bbox_inside_weights[4];
  // Give the positive RPN examples weight of p * 1 / {num positives}
  // and give negatives a weight of (1 - p)
  // Set to -1.0 to use uniform example weighting
  static float rpn_positive_weight;
  // allowed_border, when compute anchors targets, extend the border_
  static float rpn_allowed_border;

  // ========================================
	static int rng_seed;
  static float eps;
  static float inf;
	static bool share_location;
  // ========================================
	static int feat_stride;
  static std::vector<float> anchors;
  static float test_score_thresh;
  static int n_classes;
  // ========================================
	static void print_SSD_param();
  static void load_SSD_param(const std::string default_config_path);
};

}  // namespace detection

}

#endif // CAFFE_FRCNN_PRARM_HPP_
