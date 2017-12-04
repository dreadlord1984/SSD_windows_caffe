/**********************************************************************************************

  * @file   frcnn_proposal_layer.cpp

  * @brief This is a brief description.

  * @date   2017:11:30 

  * @written by xuanyuyt

  * @version <version  1>

 *********************************************************************************************/ 

#include "caffe/FRCNN/frcnn_proposal_layer.hpp"
#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/FRCNN/util/frcnn_helper.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"  
#include <time.h>
#include <iostream>
#include <io.h>
namespace caffe {

namespace Frcnn {

using std::vector;

template <typename Dtype>
void FrcnnProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
  const vector<Blob<Dtype> *> &top) {

#ifndef CPU_ONLY
	
  CUDA_CHECK(cudaMalloc(&anchors_, sizeof(float) * FrcnnParam::anchors.size()));
  CUDA_CHECK(cudaMemcpy(anchors_, &(FrcnnParam::anchors[0]),
                        sizeof(float) * FrcnnParam::anchors.size(), cudaMemcpyHostToDevice));

  const int rpn_pre_nms_top_n = 
    this->phase_ == TRAIN ? FrcnnParam::rpn_pre_nms_top_n : FrcnnParam::test_rpn_pre_nms_top_n;
	CUDA_CHECK(cudaMalloc(&transform_bbox_, bottom[0]->num()* sizeof(float) * rpn_pre_nms_top_n * 4)); // *batch_size
	CUDA_CHECK(cudaMalloc(&selected_flags_, bottom[0]->num() * sizeof(int) * rpn_pre_nms_top_n)); // *batch_size

  const int rpn_post_nms_top_n = 
    this->phase_ == TRAIN ? FrcnnParam::rpn_post_nms_top_n : FrcnnParam::test_rpn_post_nms_top_n;
	CUDA_CHECK(cudaMalloc(&gpu_keep_indices_, bottom[0]->num() * sizeof(int) * rpn_post_nms_top_n)); // *batch_size

#endif
  top[0]->Reshape(1, 5, 1, 1); // rpn_rois
  if (top.size() > 1) {
    top[1]->Reshape(1, 1, 1, 1);
  }
}

template <typename Dtype>
void FrcnnProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
	const vector<Blob<Dtype> *> &top) {
	//Forward_gpu(bottom, top);
	//#if 0
	DLOG(ERROR) << "========== enter proposal layer";
	const int scale_layer_num = FrcnnParam::scale_layer_num;
	const Dtype *conf_data = bottom[0]->cpu_data();  // rpn_cls_prob_reshape
	const Dtype *loc_data = bottom[1]->cpu_data();   // rpn_bbox_pred
	const Dtype *prior_data = bottom[2]->cpu_data();    // pripr box
	const Dtype* match_imfo = bottom[3]->cpu_data(); // match_imfo
	const Dtype* gt_data = bottom[4]->cpu_data(); // gt_data
	/*-------------------------验证代码-------------------------*/
	for (vector<Blob<Dtype> *>::const_iterator iter = bottom.cbegin(); iter != bottom.cend(); iter++)
	{
		cout << (*iter)->num() << " " << (*iter)->channels() << " " << (*iter)->height() << " " << (*iter)->width() << endl;
	}
	/*-------------------------验证代码-------------------------*/
	/***********************************************************************
	* 注意: prototxt中如果多个尺度bottom需要按照confs、locs、boxes的顺序
	***********************************************************************/
	const int batch_size = bottom[0]->num();// batch size
	const int num_priors = bottom[2]->height() / 4;
	bool share_location = FrcnnParam::share_location;
	int loc_classes = FrcnnParam::share_location ? 1 : FrcnnParam::n_classes;
	CHECK_EQ(bottom[0]->num(), bottom[1]->num());
	CHECK_EQ(num_priors * FrcnnParam::n_classes, bottom[0]->channels())
		<< "Number of priors must match number of confidence predictions.";
	CHECK_EQ(num_priors * loc_classes * 4, bottom[1]->channels()) // 如果shared表示所有的类别同用一个location prediction，否则每一类各自预测。
		<< "Number of priors must match number of location predictions.";
	
	const float im_height = FrcnnParam::im_height;
	const float im_width = FrcnnParam::im_width;

	int rpn_pre_nms_top_n;
	int rpn_post_nms_top_n;
	float rpn_nms_thresh;
	int rpn_min_size;
	if (this->phase_ == TRAIN) {
		rpn_pre_nms_top_n = FrcnnParam::rpn_pre_nms_top_n;
		rpn_post_nms_top_n = FrcnnParam::rpn_post_nms_top_n;
		rpn_nms_thresh = FrcnnParam::rpn_nms_thresh;
		rpn_min_size = FrcnnParam::rpn_min_size;
	}
	else {
		rpn_pre_nms_top_n = FrcnnParam::test_rpn_pre_nms_top_n;
		rpn_post_nms_top_n = FrcnnParam::test_rpn_post_nms_top_n;
		rpn_nms_thresh = FrcnnParam::test_rpn_nms_thresh;
		rpn_min_size = FrcnnParam::test_rpn_min_size; // 从输入图像到特征层的操作步长
	}
	LOG_IF(ERROR, rpn_pre_nms_top_n <= 0) << "rpn_pre_nms_top_n : " << rpn_pre_nms_top_n;
	LOG_IF(ERROR, rpn_post_nms_top_n <= 0) << "rpn_post_nms_top_n : " << rpn_post_nms_top_n;
	if (rpn_pre_nms_top_n <= 0 || rpn_post_nms_top_n <= 0) return;

	std::vector<std::vector<Point4f<Dtype> >> batch_box_final;
	std::vector<std::vector<Dtype>> batch_scores_;
	const int background_label_id = 0;
	bool use_difficult_gt = true;
	// 1.Retrieve all ground truth.
	/*const int num_gt = bottom[4]->height();
	map<int, vector<NormalizedBBox> > all_gt_bboxes;
	GetGroundTruth(gt_data, num_gt, background_label_id, use_difficult_gt,
		&all_gt_bboxes);
*/
	// 2. Retrieve all loc predictions.
	vector<LabelBBox> all_loc_preds;
	GetLocPredictions(loc_data, batch_size, num_priors, loc_classes, share_location,
		&all_loc_preds);
	// 3. Retrieve all matching.
	vector<vector<int> > all_match_indices;
	vector<vector<Dtype> > all_match_overlaps;
	vector<vector<Dtype>> all_match_confs;
	for (int batch_index = 0; batch_index < batch_size; batch_index++) {
		int match_begin = batch_index * num_priors;
		vector<int> match_indices;
		vector<Dtype> match_overlaps;
		vector<Dtype> match_confs;
		for (int priors_index = 0; priors_index < num_priors; priors_index++) {
			match_indices.push_back(match_imfo[match_begin * 3 + priors_index * 3]);
			match_overlaps.push_back(match_imfo[match_begin * 3 + priors_index * 3 + 1]);
			match_confs.push_back(match_imfo[match_begin * 3 + priors_index * 3 + 2]);
		}
		all_match_indices.push_back(match_indices);
		all_match_overlaps.push_back(match_overlaps);
		all_match_confs.push_back(match_confs);
	}

	//ofstream  outfile1;
	//if (_access("SSDconf_pred.txt", 0) != -1) // 如果临时文件存在，删除！
	//	remove("SSDconf_pred.txt");
	//outfile1.open("SSDconf_pred.txt", ios::out | ios::app);
	//for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
	//	outfile1 << "batch " << batch_index << endl;
	//	for (int p = 0; p < num_priors; ++p) {
	//		outfile1 << all_conf_preds[batch_index][p] << endl;
	//	}
	//}
	//outfile1.close();

	/*-------------------------验证代码-------------------------*/
	//ofstream  outfile;
	//if (_access("conf_pred.txt", 0) != -1) // 如果临时文件存在，删除！
	// remove("conf_pred.txt");
	//outfile.open("conf_pred.txt", ios::out | ios::app);
	//for (int bottom_index = 5; bottom_index < bottom.size(); bottom_index++)
	//{
	//	outfile << "layer  " << bottom_index-2 << endl;
	//	const int config_n_anchors = bottom[bottom_index]->channels() / 2;
	//	const int height = bottom[bottom_index]->height();
	//	const int width = bottom[bottom_index]->width();
	//	const int num_total = config_n_anchors*height*width;
	//	const Dtype *part_conf_data = bottom[bottom_index]->cpu_data();
	//	for (int batch_index = 0; batch_index < bottom[bottom_index]->num(); batch_index++) {
	//		outfile << "batch " << batch_index << endl;
	//		for (int j = 0; j < height; j++) {
	//			for (int i = 0; i < width; i++) {
	//				for (int k = 0; k < config_n_anchors; k++) {
	//					outfile << part_conf_data[(2 * k) * (height*width) + j*width + i] << " "
	//						<< part_conf_data[(2 * k + 1) * (height*width) + j*width + i] << endl;
	//				}
	//			}
	//		}
	//	}
	//}
	//outfile.close();
	/*-------------------------验证代码-------------------------*/
	DLOG(ERROR) << "========== generate anchors";
	std::vector<Point4f<Dtype> > anchors;
	std::vector<vector<float> > prior_variances;
	GetAnchors(prior_data, num_priors, &anchors, &prior_variances, im_height, im_width);

	for (int batch_index = 0; batch_index < batch_size; batch_index++) {
		typedef pair<Dtype, int> sort_pair;
		std::vector<sort_pair> sort_vector;
		const Dtype bounds[4] = { im_width - 1, im_height - 1, im_width - 1, im_height - 1 };
		const Dtype min_size = rpn_min_size;
		int match_begin = batch_index * num_priors;
		for (int priors_index = 0; priors_index < num_priors; priors_index++) {
			Dtype score = match_imfo[match_begin * 3 + priors_index * 3 + 2];
			int match_gt_index = match_imfo[match_begin * 3 + priors_index * 3];
			Point4f<Dtype> box_delta;
			if (match_gt_index > -1)
			{
				box_delta[0] = loc_data[match_begin * 4] * im_width;
				box_delta[1] = loc_data[match_begin * 4 + 1] * im_height;
				box_delta[2] = loc_data[match_begin * 4 + 2] * im_width;
				box_delta[3] = loc_data[match_begin * 4 + 3] * im_height;
			}
			else if (match_gt_index < -1)
			{
				continue;
			}
			cout << endl;
		}

	}

	for (int batch_index = 0; batch_index < batch_size; batch_index++) {
		std::vector<Point4f<Dtype> > anchors;
		typedef pair<Dtype, int> sort_pair;
		std::vector<sort_pair> sort_vector;
		const Dtype bounds[4] = { im_width - 1, im_height - 1, im_width - 1, im_height - 1 };
		const Dtype min_size = rpn_min_size;

		DLOG(ERROR) << "========== generate anchors";
		// min_size = FrcnnParam::rpn_min_size限制
		int prior_data_begin = 0;
		
		const int channes = bottom[1]->channels();
		const int height = bottom[1]->height();
		const int width = bottom[1]->width();
		const int config_n_anchors = FrcnnParam::anchors.size() / 4;
		DLOG(ERROR) << "========== generate anchors";
		// min_size = FrcnnParam::rpn_min_size限制
		int rpn_score_begin = batch_index * 2 * config_n_anchors * height * width;
		int rpn_bbox_begin = batch_index * 4 * config_n_anchors * height * width;
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				for (int k = 0; k < config_n_anchors; k++) {
					Dtype score = conf_data[rpn_score_begin + config_n_anchors * height * width +
						k * height * width + j * width + i];

					Point4f<Dtype> anchor(
						FrcnnParam::anchors[k * 4 + 0] + i * FrcnnParam::feat_stride,  // shift_x[i][j];
						FrcnnParam::anchors[k * 4 + 1] + j * FrcnnParam::feat_stride,  // shift_y[i][j];
						FrcnnParam::anchors[k * 4 + 2] + i * FrcnnParam::feat_stride,  // shift_x[i][j];
						FrcnnParam::anchors[k * 4 + 3] + j * FrcnnParam::feat_stride); // shift_y[i][j];

					Point4f<Dtype> box_delta(
						loc_data[rpn_bbox_begin + (k * 4 + 0) * height * width + j * width + i],
						loc_data[rpn_bbox_begin + (k * 4 + 1) * height * width + j * width + i],
						loc_data[rpn_bbox_begin + (k * 4 + 2) * height * width + j * width + i],
						loc_data[rpn_bbox_begin + (k * 4 + 3) * height * width + j * width + i]);

					// FrcnnParam::anchors加上box_delta后的框位置
					Point4f<Dtype> cbox = bbox_transform_inv(anchor, box_delta);

					// 2. clip predicted boxes to image
					for (int q = 0; q < 4; q++) {
						cbox.Point[q] = std::max(Dtype(0), std::min(cbox[q], bounds[q]));
					}
					// 3. remove predicted boxes with either height or width < threshold
					if ((cbox[2] - cbox[0] + 1) >= min_size && (cbox[3] - cbox[1] + 1) >= min_size) {
						const int now_index = sort_vector.size();
						sort_vector.push_back(sort_pair(score, now_index));
						anchors.push_back(cbox);
					}
				}
			}
		}

		DLOG(ERROR) << "========== after clip and remove size < threshold box " << (int)sort_vector.size();

		std::sort(sort_vector.begin(), sort_vector.end(), std::greater<sort_pair>());
		const int n_anchors = std::min((int)sort_vector.size(), rpn_pre_nms_top_n);
		sort_vector.erase(sort_vector.begin() + n_anchors, sort_vector.end());

		std::vector<bool> select(n_anchors, true);


		// apply nms 数量限制FrcnnParam::rpn_post_nms_top_n和阈值限制FrcnnParam::rpn_nms_thresh
		DLOG(ERROR) << "========== apply nms, pre nms number is : " << n_anchors;
		std::vector<Point4f<Dtype> > box_final;
		std::vector<Dtype> scores_;
		for (int i = 0; i < n_anchors && box_final.size() < rpn_post_nms_top_n; i++) {
			if (select[i]) {
				const int cur_i = sort_vector[i].second;
				for (int j = i + 1; j < n_anchors; j++)
					if (select[j]) {
						const int cur_j = sort_vector[j].second;
						if (get_iou(anchors[cur_i], anchors[cur_j]) > rpn_nms_thresh) {
							select[j] = false;
						}
					}
				box_final.push_back(anchors[cur_i]);
				scores_.push_back(sort_vector[i].first);
			}
		}
		batch_box_final.push_back(box_final);
		batch_scores_.push_back(scores_);
		DLOG(ERROR) << "rpn number after nms: " << box_final.size();

	}

	int total_boxes = 0;
	for (std::vector<std::vector<Point4f<Dtype> >>::iterator it =
		batch_box_final.begin(); it != batch_box_final.end(); ++it) {
		total_boxes += it->size();
	}

  DLOG(ERROR) << "========== copy to top";

	top[0]->Reshape(total_boxes, 5, 1, 1);
	if (top.size() > 1) {
		top[1]->Reshape(total_boxes, 1, 1, 1);
	}
	Dtype *top_data = top[0]->mutable_cpu_data();
	std::vector<Point4f<Dtype> > box_final;
	std::vector<Dtype> scores_;
	int box_begin = 0;
	for (size_t batch_index = 0; batch_index < batch_box_final.size(); batch_index++) {
		box_final = batch_box_final[batch_index];
		scores_ = batch_scores_[batch_index];
		CHECK_EQ(box_final.size(), scores_.size());
		for (size_t i = 0; i < box_final.size(); i++) {
			Point4f<Dtype> &box = box_final[i];
			top_data[box_begin * 5 + i * 5] = batch_index; // batch index
			for (int j = 1; j < 5; j++) {
				top_data[box_begin * 5 + i * 5 + j] = box[j - 1];
			}
		}

		if (top.size() > 1) {
			for (size_t i = 0; i < box_final.size(); i++) {
				top[1]->mutable_cpu_data()[box_begin + i] = scores_[i];
			}
		}
		box_begin += box_final.size();

	}
  DLOG(ERROR) << "========== exit proposal layer";
//#endif
}

template <typename Dtype>
void FrcnnProposalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
    const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) {
      NOT_IMPLEMENTED;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FrcnnProposalLayer);
#endif

INSTANTIATE_CLASS(FrcnnProposalLayer);
REGISTER_LAYER_CLASS(FrcnnProposal);

} // namespace frcnn

} // namespace caffe
