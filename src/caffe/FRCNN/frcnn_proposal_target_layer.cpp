// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include "caffe/FRCNN/frcnn_proposal_target_layer.hpp"
#include <io.h>

namespace caffe {

namespace Frcnn {

using std::vector;

template <typename Dtype>
void FrcnnProposalTargetLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                                 const vector<Blob<Dtype> *> &top) {
  this->rng_.reset(new Caffe::RNG(static_cast<unsigned int>(FrcnnParam::rng_seed)));
  this->_count_ = this->_bg_num_ = this->_fg_num_ = 0;

  config_n_classes_ = FrcnnParam::n_classes;

  LOG(INFO) << "FrcnnProposalTargetLayer :: " << config_n_classes_ << " classes";
  LOG(INFO) << "FrcnnProposalTargetLayer :: LayerSetUp";
  // sampled rois (0, x1, y1, x2, y2)
  top[0]->Reshape(1, 5, 1, 1); // top: "rois": 给ROIPooling的bottom[1]
  // labels
  top[1]->Reshape(1, 1, 1, 1); // top: "labels": 给Accuracy的bottom[1]和loss_cls的bottom[1]
  // bbox_targets
  top[2]->Reshape(1, config_n_classes_ * 4, 1, 1); // top: "bbox_targets": 给loss_bbox的bottom[1]
  // bbox_inside_weights
  top[3]->Reshape(1, config_n_classes_ * 4, 1, 1); // top: "bbox_inside_weights": 给loss_bbox的bottom[2]
  // bbox_outside_weights
  top[4]->Reshape(1, config_n_classes_ * 4, 1, 1); // top: "bbox_outside_weights": 给loss_bbox的bottom[3]
}

template <typename Dtype>
void FrcnnProposalTargetLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
	//bottom[0]: "rpn_rois" 维度：(box_final.size(), 5, 1, 1);
	//bottom[1]: "gt_boxes" 维度：SSD(1, 1, gt_box.size(),8), faster(gt_box.size(), 5, 1,1);
	/*-------------------------验证代码-------------------------*/
	/*for (vector<Blob<Dtype> *>::const_iterator iter = bottom.cbegin(); iter != bottom.cend(); iter++)
	{
		cout << (*iter)->num() << " " << (*iter)->channels() << " " << (*iter)->height() << " " << (*iter)->width() << endl;
	}*/
	/*-------------------------验证代码-------------------------*/

	/*-------------------------改写-------------------------*/
	map<int, vector<Point4f<Dtype> >> batch_all_rois;
	map<int, vector<Point4f<Dtype> >> batch_gt_boxes;
	map<int, vector<int>> batch_gt_labels;


	if (bottom[1]->channels() == 5) // faster数据类型, batch_size = 1
	{
		for (int i = 0; i < bottom[0]->num(); i++) {
			CHECK_EQ(bottom[0]->data_at(i, 0, 0, 0), 0) << "Only single item batches are supported";
			batch_all_rois[0].push_back(Point4f<Dtype>(
				bottom[0]->data_at(i, 1, 0, 0),
				bottom[0]->data_at(i, 2, 0, 0),
				bottom[0]->data_at(i, 3, 0, 0),
				bottom[0]->data_at(i, 4, 0, 0)));
		}
		for (int i = 0; i < bottom[1]->num(); i++) {
			batch_gt_boxes[0].push_back(Point4f<Dtype>(
				bottom[1]->data_at(i, 0, 0, 0),
				bottom[1]->data_at(i, 1, 0, 0),
				bottom[1]->data_at(i, 2, 0, 0),
				bottom[1]->data_at(i, 3, 0, 0)));
			batch_gt_labels[0].push_back(bottom[1]->data_at(i, 4, 0, 0));
			CHECK_GT(batch_gt_labels[0][i], 0) << "Ground Truth Should Be Greater Than 0";
		}
	}
	else if (bottom[1]->channels() == 1)// SSD数据类型，batch_size >= 1
	{
		 for (int i = 0; i < bottom[0]->num(); i++) {//对于每个anchor
			int batch_index = bottom[0]->data_at(i, 0, 0, 0);
			batch_all_rois[batch_index].push_back(Point4f<Dtype>(
		       bottom[0]->data_at(i,1,0,0),
		       bottom[0]->data_at(i,2,0,0),
		       bottom[0]->data_at(i,3,0,0),
		       bottom[0]->data_at(i,4,0,0)));
		 }
		 /*-------------------------验证代码-------------------------*/
		 //ofstream  outfile;
		 //if (_access("frcnn_top.txt", 0) != -1) // 如果临时文件存在，删除！
			// remove("frcnn_top.txt");
		 //outfile.open("frcnn_top.txt", ios::out | ios::app);
		 //outfile << bottom[0]->num() << endl;
		 //for (int i = 0; i < bottom[0]->num(); i++) {
			// outfile << bottom[0]->data_at(i, 0, 0, 0) << " "
			//	 << bottom[0]->data_at(i, 1, 0, 0) << " "
			//	 << bottom[0]->data_at(i, 2, 0, 0) << " "
			//	 << bottom[0]->data_at(i, 3, 0, 0) << " "
			//	 << bottom[0]->data_at(i, 4, 0, 0) << endl;
		 //}
		 //outfile.close();
		 /*-------------------------验证代码-------------------------*/
		 const Dtype* gt_data = bottom[1]->cpu_data();
		 for (int i = 0; i < bottom[1]->height(); i++) {
		  int start_idx = i * 8;
			int batch_index = gt_data[start_idx]; // batch size index
			if (batch_index == -1) {
				cerr << "batch index is " << batch_index << endl;
		  }
			batch_gt_boxes[batch_index].push_back(Point4f<Dtype>(
			  gt_data[start_idx + 3] * 384,
			  gt_data[start_idx + 4] * 256,
			  gt_data[start_idx + 5] * 384,
			  gt_data[start_idx + 6] * 256));
			batch_gt_labels[batch_index].push_back(gt_data[start_idx + 1]);

			CHECK_GT(batch_gt_labels[batch_index].size(), 0) << "Ground Truth Should Be Greater Than 0";
		 }
	}
	const int image_batch_size = batch_all_rois.size();// batch size

	vector<vector<int >> batch_labels;
	vector<vector<Point4f<Dtype> > > batch_rois;
	vector<vector<vector<Point4f<Dtype> > > > batch_bbox_targets, batch_bbox_inside_weights;

	int batch_batch_size = 0;

	for (int batch_index = 0; batch_index < image_batch_size; batch_index++)
	{
		vector<Point4f<Dtype> > all_rois = batch_all_rois[batch_index];
		vector<Point4f<Dtype> > gt_boxes = batch_gt_boxes[batch_index];
		vector<int> gt_labels = batch_gt_labels[batch_index];
		all_rois.insert(all_rois.end(), gt_boxes.begin(), gt_boxes.end());
		DLOG(ERROR) << "gt boxes size: " << gt_boxes.size();
		const int num_images = 1;
		const int rois_per_image = FrcnnParam::batch_size / num_images; // 每个图片选择的一次进入训练的rois框数目
		const int fg_rois_per_image = rois_per_image * FrcnnParam::fg_fraction; // 这些rois框中正样本占比

		//Sample rois with classification labels and bounding box regression
		//targets
		vector<int> labels;
		vector<Point4f<Dtype> > rois;
		vector<vector<Point4f<Dtype> > > bbox_targets, bbox_inside_weights;

		_sample_rois(all_rois, gt_boxes, gt_labels, fg_rois_per_image, rois_per_image, labels, rois, bbox_targets, bbox_inside_weights);

		CHECK_EQ(labels.size(), rois.size());
		CHECK_EQ(labels.size(), bbox_targets.size());
		CHECK_EQ(labels.size(), bbox_inside_weights.size());
		batch_batch_size += rois.size();

		batch_labels.push_back(labels);
		batch_rois.push_back(rois);
		batch_bbox_targets.push_back(bbox_targets);
		batch_bbox_inside_weights.push_back(bbox_inside_weights);
	}

	DLOG(ERROR) << "top[0]-> " << batch_batch_size << " , 5, 1, 1";
  // 1. top[0]:sampled rois
	top[0]->Reshape(batch_batch_size, 5, 1, 1); // rois
  caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
  Dtype *rois_data = top[0]->mutable_cpu_data();
	// 2. top[1]:classification labels
	top[1]->Reshape(batch_batch_size, 1, 1, 1);
	Dtype *label_data = top[1]->mutable_cpu_data();
	// 3. top[2]:bbox_targets
	top[2]->Reshape(batch_batch_size, this->config_n_classes_ * 4, 1, 1);
	caffe_set(top[2]->count(), Dtype(0), top[2]->mutable_cpu_data());
	Dtype *target_data = top[2]->mutable_cpu_data();
	// 4. top[3],top[4]:bbox_inside_weights and bbox_outside_weights
	top[3]->Reshape(batch_batch_size, this->config_n_classes_ * 4, 1, 1); //bbox_inside_weights
	caffe_set(top[3]->count(), Dtype(0), top[3]->mutable_cpu_data());
	Dtype *bbox_inside_data = top[3]->mutable_cpu_data();
	top[4]->Reshape(batch_batch_size, this->config_n_classes_ * 4, 1, 1); //bbox_outside_weights
	caffe_set(top[4]->count(), Dtype(0), top[4]->mutable_cpu_data());
	Dtype *bbox_outside_data = top[4]->mutable_cpu_data();

	int rois_begin = 0;
	for (int batch_index = 0; batch_index < image_batch_size; batch_index++)
	{
		const int batch_size = batch_rois[batch_index].size();
		// sampled rois
		for (int i = 0; i < batch_size; i++) {
			rois_data[top[0]->offset(rois_begin + i, 1, 0, 0)] = batch_rois[batch_index][i][0];
			rois_data[top[0]->offset(rois_begin + i, 2, 0, 0)] = batch_rois[batch_index][i][1];
			rois_data[top[0]->offset(rois_begin + i, 3, 0, 0)] = batch_rois[batch_index][i][2];
			rois_data[top[0]->offset(rois_begin + i, 4, 0, 0)] = batch_rois[batch_index][i][3];

			label_data[top[1]->offset(rois_begin + i, 0, 0, 0)] = batch_labels[batch_index][i];

			for (int j = 0; j < this->config_n_classes_; j++) {
				for (int cor = 0; cor < 4; cor++) {
					target_data[top[2]->offset(rois_begin + i, j * 4 + cor, 0, 0)] = batch_bbox_targets[batch_index][i][j][cor];
					bbox_inside_data[top[2]->offset(rois_begin + i, j * 4 + cor, 0, 0)] = batch_bbox_inside_weights[batch_index][i][j][cor];
					bbox_outside_data[top[2]->offset(rois_begin + i, j * 4 + cor, 0, 0)] = batch_bbox_inside_weights[batch_index][i][j][cor] > 0;
				}
			}
		}

		rois_begin += batch_size;
	}
	/*cout << "top[0]: " << top[0]->num() << " " << top[0]->channels() << " " << top[0]->height() << " "
		<< top[0]->width() << endl;
	cout << "top[1]: " << top[1]->num() << " " << top[1]->channels() << " " << top[1]->height() << " "
		<< top[1]->width() << endl;
	cout << "top[2]: " << top[2]->num() << " " << top[2]->channels() << " " << top[2]->height() << " "
		<< top[2]->width() << endl;
	cout << "top[3]: " << top[3]->num() << " " << top[3]->channels() << " " << top[3]->height() << " "
		<< top[3]->width() << endl;
	cout << "top[4]: " << top[4]->num() << " " << top[4]->channels() << " " << top[4]->height() << " "
		<< top[4]->width() << endl;*/
  
	/*-------------------------改写-------------------------*/
  DLOG(INFO) << "FrcnnProposalTargetLayer::Forward_cpu End";
}

template <typename Dtype>
void FrcnnProposalTargetLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) {
      NOT_IMPLEMENTED;
    }
  }
}

template <typename Dtype>
void FrcnnProposalTargetLayer<Dtype>::_sample_rois(const vector<Point4f<Dtype> > &all_rois, const vector<Point4f<Dtype> > &gt_boxes, 
        const vector<int> &gt_label, const int fg_rois_per_image, const int rois_per_image, vector<int> &labels, 
        vector<Point4f<Dtype> > &rois, vector<vector<Point4f<Dtype> > > &bbox_targets, vector<vector<Point4f<Dtype> > > &bbox_inside_weights) {
// Generate a random sample of RoIs comprising foreground and background examples.

  CHECK_EQ(gt_label.size(), gt_boxes.size());
  // overlaps: (rois x gt_boxes):相同overlap时（例如都是0）将最大匹配给最后一个gt_box
  std::vector<std::vector<Dtype> > overlaps = get_ious(all_rois, gt_boxes); 
  std::vector<Dtype> max_overlaps(all_rois.size(), 0);
  std::vector<int> gt_assignment(all_rois.size(), -1);
  std::vector<int> _labels(all_rois.size());
  for (int i = 0; i < all_rois.size(); ++ i) {
    for (int j = 0; j < gt_boxes.size(); ++ j) {
      if (max_overlaps[i] <= overlaps[i][j]) {
        max_overlaps[i] = overlaps[i][j];
        gt_assignment[i] = j;       
      }
    }
  }
  DLOG(INFO) << "sample_rois : all_rois: " << all_rois.size() << ", gt_box: " << gt_boxes.size();
  for (size_t i = 0; i < all_rois.size(); ++i) {
    if (gt_assignment[i] >= 0 ) {
      CHECK_LT(gt_assignment[i], gt_label.size());
      _labels[i] = gt_label[gt_assignment[i]];
    } else {
      _labels[i] = 0;
    }
  }
  
  // Select foreground RoIs as those with >= FG_THRESH overlap
  std::vector<int> fg_inds;
  for (int i = 0; i < all_rois.size(); ++i) {
    if (max_overlaps[i] >= FrcnnParam::fg_thresh) {
      fg_inds.push_back(i);
    }
  }
  // Guard against the case when an image has fewer than fg_rois_per_image
  // foreground RoIs
  const int fg_rois_per_this_image = std::min(fg_rois_per_image, int(fg_inds.size()));
  DLOG(INFO) << "fg_inds [PRE,AFT] : [" << fg_inds.size() << "," << fg_rois_per_this_image << "] FG_THRESH : " << FrcnnParam::fg_thresh;
  // Sample foreground regions without replacement
  if (fg_inds.size() > 0) {//打乱截取
    shuffle(fg_inds.begin(), fg_inds.end(), (caffe::rng_t *) this->rng_->generator());
    fg_inds.resize(fg_rois_per_this_image);
  }
  
  // Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
  std::vector<int> bg_inds;
  for (int i = 0; i < all_rois.size(); ++i) {
    if (max_overlaps[i] >= FrcnnParam::bg_thresh_lo 
        && max_overlaps[i] < FrcnnParam::bg_thresh_hi) {
      bg_inds.push_back(i);
    }
  }
  // Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
  const int bg_rois_per_this_image = std::min(rois_per_image-fg_rois_per_this_image, int(bg_inds.size()));
  DLOG(INFO) << "bg_inds [PRE,AFT] : [" << bg_inds.size() << "," << bg_rois_per_this_image << "] BG_THRESH : [" << FrcnnParam::bg_thresh_lo << ", " << FrcnnParam::bg_thresh_hi << ")" ;
  // Sample background regions without replacement
  if (bg_inds.size() > 0) {//打乱截取
    shuffle(bg_inds.begin(), bg_inds.end(), (caffe::rng_t *) this->rng_->generator());
    bg_inds.resize(bg_rois_per_this_image);
  }

  // The indices that we're selecting (both fg and bg)
  std::vector<int> keep_inds(fg_inds);
  keep_inds.insert(keep_inds.end(), bg_inds.begin(), bg_inds.end());
  // Select sampled values from various arrays:
  labels.resize(keep_inds.size());
  rois.resize(keep_inds.size());
  std::vector<Point4f<Dtype> > _gt_boxes(keep_inds.size());
  for (size_t i = 0; i < keep_inds.size(); ++ i) {
    labels[i] = _labels[keep_inds[i]];
    rois[i] = all_rois[keep_inds[i]];
    _gt_boxes[i] =
        gt_assignment[keep_inds[i]] >= 0 ? gt_boxes[gt_assignment[keep_inds[i]]] : Point4f<Dtype>();
    // Clamp labels for the background RoIs to 0
    if ( i >= fg_rois_per_this_image ) 
        labels[i] = 0;
  }

#ifdef DEBUG
  DLOG(INFO) << "num fg : " << labels.size() - std::count(labels.begin(), labels.end(), 0);
  DLOG(INFO) << "num bg : " << std::count(labels.begin(), labels.end(), 0);
  CHECK_EQ(std::count(labels.begin(), labels.end(), -1), 0);
  this->_count_ += 1;
  this->_fg_num_ += labels.size() - std::count(labels.begin(), labels.end(), 0);
  this->_bg_num_ += std::count(labels.begin(), labels.end(), 0);
  DLOG(INFO) << "num fg avg : " << this->_fg_num_ * 1. / this->_count_;
  DLOG(INFO) << "num bg avg : " << this->_bg_num_ * 1. / this->_count_;
  DLOG(INFO) << "ratio : " << float(this->_fg_num_) / float(this->_bg_num_+FrcnnParam::eps);
  DLOG(INFO) << "FrcnnParam::bbox_normalize_targets : " << (FrcnnParam::bbox_normalize_targets ? "True" : "False");
  DLOG(INFO) << "FrcnnParam::bbox_normalize_means : " << FrcnnParam::bbox_normalize_means[0] << ", " << FrcnnParam::bbox_normalize_means[1]
        << ", " << FrcnnParam::bbox_normalize_means[2] << ", " << FrcnnParam::bbox_normalize_means[3];
  DLOG(INFO) << "FrcnnParam::bbox_normalize_stds : " << FrcnnParam::bbox_normalize_stds[0] << ", " << FrcnnParam::bbox_normalize_stds[1]
        << ", " << FrcnnParam::bbox_normalize_stds[2] << ", " << FrcnnParam::bbox_normalize_stds[3];
#endif

  //def _compute_targets(ex_rois, gt_rois, labels):
	//targets_dx = (gt_ctr_x - ex_ctr_x) / ex_width;
	//targets_dy = (gt_ctr_y - ex_ctr_y) / ex_height;
	//targets_dw = log(gt_widths / ex_width);
	//targets_dh = log(gt_heights / ex_height);
  CHECK_EQ(rois.size(), _gt_boxes.size());
  CHECK_EQ(rois.size(), labels.size());
  std::vector<Point4f<Dtype> > bbox_targets_data = bbox_transform(rois, _gt_boxes);
  if ( FrcnnParam::bbox_normalize_targets ) {
    // Optionally normalize targets by a precomputed mean and stdev
    for (size_t index = 0; index < bbox_targets_data.size(); index ++) {
      for (int j = 0; j < 4; j ++) {
        bbox_targets_data[index][j] = (bbox_targets_data[index][j]-FrcnnParam::bbox_normalize_means[j]) / FrcnnParam::bbox_normalize_stds[j];
      }
    }
  }

  // Compute boxes target 
  bbox_targets = std::vector<std::vector<Point4f<Dtype> > >(
          keep_inds.size(), std::vector<Point4f<Dtype> >(this->config_n_classes_));
  bbox_inside_weights = std::vector<std::vector<Point4f<Dtype> > >(
          keep_inds.size(), std::vector<Point4f<Dtype> >(this->config_n_classes_));
  for (size_t i = 0; i < labels.size(); ++i) if (labels[i] > 0) {
    int cls = labels[i];
    //get bbox_targets and bbox_inside_weights
    bbox_targets[i][cls] = bbox_targets_data[i];
    bbox_inside_weights[i][cls] = Point4f<Dtype>(FrcnnParam::bbox_inside_weights);//正样本给权重1，负样本给0
  }

}

#ifdef CPU_ONLY
STUB_GPU(FrcnnProposalTargetLayer);
#endif

INSTANTIATE_CLASS(FrcnnProposalTargetLayer);
REGISTER_LAYER_CLASS(FrcnnProposalTarget);

} // namespace frcnn

} // namespace caffe
