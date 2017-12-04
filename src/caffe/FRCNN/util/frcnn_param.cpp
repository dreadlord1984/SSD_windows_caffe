#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"
#include "caffe/common.hpp"

namespace caffe {

	using namespace caffe::Frcnn;

	int FrcnnParam::im_height;
	int FrcnnParam::im_width;
	int FrcnnParam::scale_layer_num;

	int FrcnnParam::batch_size;
	float FrcnnParam::fg_fraction;
	float FrcnnParam::fg_thresh;
	// Overlap threshold for a ROI to be considered background (class = 0
	// ifoverlap in [LO, HI))
	float FrcnnParam::bg_thresh_hi;
	float FrcnnParam::bg_thresh_lo;

	// Train bounding-box regressors
	bool FrcnnParam::bbox_normalize_targets;
	float FrcnnParam::bbox_inside_weights[4];
	float FrcnnParam::bbox_normalize_means[4];
	float FrcnnParam::bbox_normalize_stds[4];

	// RPN to detect objects
	float FrcnnParam::rpn_nms_thresh;
	int FrcnnParam::rpn_pre_nms_top_n;
	int FrcnnParam::rpn_post_nms_top_n;
	float FrcnnParam::test_rpn_nms_thresh;
	int FrcnnParam::test_rpn_pre_nms_top_n;
	int FrcnnParam::test_rpn_post_nms_top_n;

	// Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
	float FrcnnParam::rpn_min_size;
	float FrcnnParam::test_rpn_min_size;

	// Deprecated (outside weights)
	float FrcnnParam::rpn_bbox_inside_weights[4];
	// Give the positive RPN examples weight of p * 1 / {num positives}
	// and give negatives a weight of (1 - p)
	// Set to -1.0 to use uniform example weighting
	float FrcnnParam::rpn_positive_weight;
	float FrcnnParam::rpn_allowed_border;


	// ========================================
	int FrcnnParam::rng_seed;
	float FrcnnParam::eps;
	float FrcnnParam::inf;
	bool FrcnnParam::share_location;

	// ======================================== 
	int FrcnnParam::feat_stride;
	std::vector<float> FrcnnParam::anchors;
	float FrcnnParam::test_score_thresh;
	int FrcnnParam::n_classes;



void FrcnnParam::print_SSD_param(){
  LOG(INFO) << "== Train  Parameters ==";
	LOG(INFO) << "im_height          : " << FrcnnParam::im_height;
	LOG(INFO) << "im_width           : " << FrcnnParam::im_width;
	LOG(INFO) << "scale_layer_num    : " << FrcnnParam::scale_layer_num;

  LOG(INFO) << "batch_size        : " << FrcnnParam::batch_size;
  LOG(INFO) << "fg_fraction       : " << FrcnnParam::fg_fraction;
  LOG(INFO) << "fg_thresh         : " << FrcnnParam::fg_thresh; 
  LOG(INFO) << "bg_thresh_hi      : " << FrcnnParam::bg_thresh_hi;
  LOG(INFO) << "bg_thresh_lo      : " << FrcnnParam::bg_thresh_lo;

  LOG(INFO) << "normalize_targets : " << (FrcnnParam::bbox_normalize_targets ? "yes" : "no");


  LOG(INFO) << "rpn_nms_thresh       : " << FrcnnParam::rpn_nms_thresh;
	LOG(INFO) << "rpn_pre_nms_top_n    : " << FrcnnParam::rpn_pre_nms_top_n;
	LOG(INFO) << "rpn_post_nms_top_n   : " << FrcnnParam::rpn_post_nms_top_n;
	LOG(INFO) << "rpn_min_size         : " << FrcnnParam::rpn_min_size;

	LOG(INFO) << "test_rpn_nms_thresh       : " << FrcnnParam::test_rpn_nms_thresh;
	LOG(INFO) << "test_rpn_pre_nms_top_n    : " << FrcnnParam::test_rpn_pre_nms_top_n;
	LOG(INFO) << "test_rpn_post_nms_top_n   : " << FrcnnParam::test_rpn_post_nms_top_n;
	LOG(INFO) << "test_rpn_min_size         : " << FrcnnParam::test_rpn_min_size;

  LOG(INFO) << "rpn_bbox_inside_weights :" << float_to_string(FrcnnParam::rpn_bbox_inside_weights);
  LOG(INFO) << "rpn_positive_weight     :" << FrcnnParam::rpn_positive_weight;
  LOG(INFO) << "rpn_allowed_border      :" << FrcnnParam::rpn_allowed_border;

  LOG(INFO) << "== Global Parameters ==";
  LOG(INFO) << "eps                  : " << FrcnnParam::eps; 
  LOG(INFO) << "inf                  : " << FrcnnParam::inf; 
	LOG(INFO) << "feat_stride          : " << FrcnnParam::feat_stride;
  LOG(INFO) << "anchors_size         : " << FrcnnParam::anchors.size();
	LOG(INFO) << "test_score_thresh    : " << FrcnnParam::test_score_thresh;
	LOG(INFO) << "n_classes            : " << FrcnnParam::n_classes;
}

void FrcnnParam::load_SSD_param(const std::string default_config_path) {
	std::vector<float> v_tmp;

	str_map default_map = parse_json_config(default_config_path);

	FrcnnParam::im_height = extract_int("im_height", default_map);//训练、测试图片高度
	FrcnnParam::im_width = extract_int("im_width", default_map);//训练、测试图片宽度
	FrcnnParam::scale_layer_num = extract_int("scale_layer_num", default_map);//mutile_scale layer

	FrcnnParam::batch_size = extract_int("batch_size", default_map);//ROIs进入训练的总数
	FrcnnParam::fg_fraction = extract_float("fg_fraction", default_map);//ROIs中正样本占比
	FrcnnParam::fg_thresh = extract_float("fg_thresh", default_map);//作为正样本的IOU阈值
	FrcnnParam::bg_thresh_hi = extract_float("bg_thresh_hi", default_map);//作为负样本的IOU高阈值
	FrcnnParam::bg_thresh_lo = extract_float("bg_thresh_lo", default_map);//作为负样本的IOU低阈值
	//bbox归一化参数
	FrcnnParam::bbox_normalize_targets =
		static_cast<bool>(extract_int("bbox_normalize_targets", default_map));
	v_tmp = extract_vector("bbox_inside_weights", default_map);
	std::copy(v_tmp.begin(), v_tmp.end(), FrcnnParam::bbox_inside_weights);
	v_tmp = extract_vector("bbox_normalize_means", default_map);
	std::copy(v_tmp.begin(), v_tmp.end(), FrcnnParam::bbox_normalize_means);
	v_tmp = extract_vector("bbox_normalize_stds", default_map);
	std::copy(v_tmp.begin(), v_tmp.end(), FrcnnParam::bbox_normalize_stds);

	//rpn_rois筛选
	FrcnnParam::rpn_nms_thresh = extract_float("rpn_nms_thresh", default_map);//rpn nms阈值
	FrcnnParam::rpn_pre_nms_top_n = extract_int("rpn_pre_nms_top_n", default_map);//第一次数量限制
	FrcnnParam::rpn_post_nms_top_n = extract_int("rpn_post_nms_top_n", default_map);//第二次数量限制
	FrcnnParam::rpn_min_size = extract_int("rpn_min_size", default_map);//去掉小框

	FrcnnParam::test_rpn_nms_thresh = extract_float("test_rpn_nms_thresh", default_map);//rpn nms阈值
	FrcnnParam::test_rpn_pre_nms_top_n = extract_int("test_rpn_pre_nms_top_n", default_map);//第一次数量限制
	FrcnnParam::test_rpn_post_nms_top_n = extract_int("test_rpn_post_nms_top_n", default_map);//第二次数量限制
	FrcnnParam::test_rpn_min_size = extract_int("test_rpn_min_size", default_map);//去掉小框

	v_tmp = extract_vector("rpn_bbox_inside_weights", default_map);
	std::copy(v_tmp.begin(), v_tmp.end(), FrcnnParam::rpn_bbox_inside_weights);
	FrcnnParam::rpn_positive_weight = extract_float("rpn_positive_weight", default_map);
	FrcnnParam::rpn_allowed_border = extract_float("rpn_allowed_border", default_map);

	// ========================================
	FrcnnParam::rng_seed = extract_int("rng_seed", default_map);
	FrcnnParam::eps = extract_float("eps", default_map);
	FrcnnParam::inf = extract_float("inf", default_map);
	FrcnnParam::share_location =
		static_cast<bool>(extract_int("share_location", default_map));
	// ========================================
	FrcnnParam::feat_stride = extract_int("feat_stride", default_map);
	FrcnnParam::anchors = extract_vector("anchors", default_map);
	FrcnnParam::test_score_thresh = extract_float("test_score_thresh", default_map);
	FrcnnParam::n_classes = extract_int("n_classes", default_map);

}

} // namespace detection
