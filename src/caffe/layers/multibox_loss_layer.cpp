#include <algorithm>
#include <io.h>
#include <map>
#include <utility>
#include <vector>

#include "caffe/layers/multibox_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {

	template <typename Dtype>
	void MultiBoxLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		/*-------------------------��д-------------------------*/
		for (int topIndex = 1; topIndex < top.size(); topIndex++)
		{
			this->layer_param_.add_loss_weight(Dtype(0));//for top[>0]
		}
		/*-------------------------��д-------------------------*/
		if (this->layer_param_.propagate_down_size() == 0) {
			this->layer_param_.add_propagate_down(true);
			this->layer_param_.add_propagate_down(true);
			this->layer_param_.add_propagate_down(false);
			this->layer_param_.add_propagate_down(false);
		}
		const MultiBoxLossParameter& multibox_loss_param =
			this->layer_param_.multibox_loss_param();
		multibox_loss_param_ = this->layer_param_.multibox_loss_param();

		num_ = bottom[0]->num();	//�����batchsize
		num_priors_ = bottom[2]->height() / 4; //����������default boxes������ÿ������������ϽǺ����½ǵĵ����ꡣ
		// Get other parameters.
		CHECK(multibox_loss_param.has_num_classes()) << "Must provide num_classes."; //������һ��Ҫ�ṩ
		num_classes_ = multibox_loss_param.num_classes();	//������
		CHECK_GE(num_classes_, 1) << "num_classes should not be less than 1.";
		share_location_ = multibox_loss_param.share_location();//�������λ��Ԥ��
		loc_classes_ = share_location_ ? 1 : num_classes_;//���shared��ʾ���е����ͬ��һ��location prediction������ÿһ�����Ԥ�⡣
		background_label_id_ = multibox_loss_param.background_label_id();//background��id
		use_difficult_gt_ = multibox_loss_param.use_difficult_gt();//�Ƿ�ʹ��difficutlt��ground truth
		mining_type_ = multibox_loss_param.mining_type();
		if (multibox_loss_param.has_do_neg_mining()) {
			LOG(WARNING) << "do_neg_mining is deprecated, use mining_type instead.";
			do_neg_mining_ = multibox_loss_param.do_neg_mining();
			CHECK_EQ(do_neg_mining_,
				mining_type_ != MultiBoxLossParameter_MiningType_NONE);
		}
		do_neg_mining_ = mining_type_ != MultiBoxLossParameter_MiningType_NONE;

		if (!this->layer_param_.loss_param().has_normalization() &&
			this->layer_param_.loss_param().has_normalize()) {
			normalization_ = this->layer_param_.loss_param().normalize() ?
			LossParameter_NormalizationMode_VALID :
												  LossParameter_NormalizationMode_BATCH_SIZE;
		}
		else {
			normalization_ = this->layer_param_.loss_param().normalization();
		}

		if (do_neg_mining_) {
			CHECK(share_location_)
				<< "Currently only support negative mining if share_location is true.";
		}

		vector<int> loss_shape(1, 1);
		// Set up localization loss layer.
		loc_weight_ = multibox_loss_param.loc_weight();//location loss ��� confidence loss �� weight
		loc_loss_type_ = multibox_loss_param.loc_loss_type();//loss������
		// fake shape.
		vector<int> loc_shape(1, 1);
		loc_shape.push_back(4);
		loc_pred_.Reshape(loc_shape);
		loc_gt_.Reshape(loc_shape);
		loc_bottom_vec_.push_back(&loc_pred_);
		loc_bottom_vec_.push_back(&loc_gt_);
		loc_loss_.Reshape(loss_shape);//location��loss
		loc_top_vec_.push_back(&loc_loss_);
		//�½�һ���㣬ʵ�ֶ�locationloss�ļ��㡣
		if (loc_loss_type_ == MultiBoxLossParameter_LocLossType_L2) {
			LayerParameter layer_param;
			layer_param.set_name(this->layer_param_.name() + "_l2_loc");
			layer_param.set_type("EuclideanLoss");
			layer_param.add_loss_weight(loc_weight_);
			loc_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
			loc_loss_layer_->SetUp(loc_bottom_vec_, loc_top_vec_);
		}
		else if (loc_loss_type_ == MultiBoxLossParameter_LocLossType_SMOOTH_L1) {
			LayerParameter layer_param;
			layer_param.set_name(this->layer_param_.name() + "_smooth_L1_loc");
			layer_param.set_type("SmoothL1Loss");
			layer_param.add_loss_weight(loc_weight_);
			loc_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
			loc_loss_layer_->SetUp(loc_bottom_vec_, loc_top_vec_);
		}
		else {
			LOG(FATAL) << "Unknown localization loss type.";
		}
		// Set up confidence loss layer.
		//�½�һ���㣬ʵ�ֵ��Ƕ�confidence loss�ļ��㡣
		conf_loss_type_ = multibox_loss_param.conf_loss_type();
		conf_bottom_vec_.push_back(&conf_pred_);
		conf_bottom_vec_.push_back(&conf_gt_);
		conf_loss_.Reshape(loss_shape);
		conf_top_vec_.push_back(&conf_loss_);
		if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
			CHECK_GE(background_label_id_, 0)
				<< "background_label_id should be within [0, num_classes) for Softmax.";
			CHECK_LT(background_label_id_, num_classes_)
				<< "background_label_id should be within [0, num_classes) for Softmax.";
			LayerParameter layer_param;
			layer_param.set_name(this->layer_param_.name() + "_softmax_conf");
			layer_param.set_type("SoftmaxWithLoss");
			layer_param.add_loss_weight(Dtype(1.));
			layer_param.mutable_loss_param()->set_normalization(
				LossParameter_NormalizationMode_NONE);



			SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
			softmax_param->set_axis(1);
			// Fake reshape.
			vector<int> conf_shape(1, 1);
			conf_gt_.Reshape(conf_shape);
			conf_shape.push_back(num_classes_);
			conf_pred_.Reshape(conf_shape);
			conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
			conf_loss_layer_->SetUp(conf_bottom_vec_, conf_top_vec_);
		}
		else if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
			LayerParameter layer_param;
			layer_param.set_name(this->layer_param_.name() + "_logistic_conf");
			layer_param.set_type("SigmoidCrossEntropyLoss");
			layer_param.add_loss_weight(Dtype(1.));
			// Fake reshape.
			vector<int> conf_shape(1, 1);
			conf_shape.push_back(num_classes_);
			conf_gt_.Reshape(conf_shape);
			conf_pred_.Reshape(conf_shape);
			conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
			conf_loss_layer_->SetUp(conf_bottom_vec_, conf_top_vec_);
		}
		/*****************************************************************************/
		else if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_FocalLoss) {
			CHECK_GE(background_label_id_, 0)
				<< "background_label_id should be within [0, num_classes) for Softmax.";
			CHECK_LT(background_label_id_, num_classes_)
				<< "background_label_id should be within [0, num_classes) for Softmax.";
			LayerParameter layer_param;
			layer_param.set_name(this->layer_param_.name() + "_focal_conf");
			layer_param.set_type("FocalLoss");
			layer_param.add_loss_weight(Dtype(1.));
			layer_param.mutable_loss_param()->set_normalization(
				LossParameter_NormalizationMode_NONE);
			layer_param.mutable_loss_param()->set_ignore_label(-1);

			FocalLossParameter *focal_loss_param = layer_param.mutable_focal_loss_param();
			focal_loss_param->set_alpha(multibox_loss_param.fl_alpha());
			focal_loss_param->set_gamma(multibox_loss_param.fl_gamma());
			focal_loss_param->set_beta(multibox_loss_param.fl_beta());

			// Fake reshape.
			vector<int> conf_shape(1, 1);
			conf_gt_.Reshape(conf_shape);
			conf_shape.push_back(num_classes_);
			conf_pred_.Reshape(conf_shape);
			conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
			conf_loss_layer_->SetUp(conf_bottom_vec_, conf_top_vec_);
		}
		/*****************************************************************************/
		else {
			LOG(FATAL) << "Unknown confidence loss type.";
		}
	}

	template <typename Dtype>
	void MultiBoxLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::Reshape(bottom, top);
		num_ = bottom[0]->num(); // batch size
		num_priors_ = bottom[2]->height() / 4;
		num_gt_ = bottom[3]->height();
		CHECK_EQ(bottom[0]->num(), bottom[1]->num());
		CHECK_EQ(num_priors_ * loc_classes_ * 4, bottom[0]->channels())
			<< "Number of priors must match number of location predictions.";
		CHECK_EQ(num_priors_ * num_classes_, bottom[1]->channels())
			<< "Number of priors must match number of confidence predictions.";
	}

	template <typename Dtype>
	void MultiBoxLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* loc_data = bottom[0]->cpu_data();
		const Dtype* conf_data = bottom[1]->cpu_data();
		const Dtype* prior_data = bottom[2]->cpu_data();
		const Dtype* gt_data = bottom[3]->cpu_data();

		// Retrieve all ground truth.
		map<int, vector<NormalizedBBox> > all_gt_bboxes;
		GetGroundTruth(gt_data, num_gt_, background_label_id_, use_difficult_gt_,
			&all_gt_bboxes);

		// Retrieve all prior bboxes. It is same within a batch since we assume all
		// images in a batch are of same dimension.
		vector<NormalizedBBox> prior_bboxes;
		vector<vector<float> > prior_variances;
		GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);

		// Retrieve all predictions.
		vector<LabelBBox> all_loc_preds;
		GetLocPredictions(loc_data, num_, num_priors_, loc_classes_, share_location_,
			&all_loc_preds);


		/************************************************************************/
		// ��ֹ����ʱû�з����������
		if (all_match_indices_.size())
		{
			all_match_indices_.clear();
			all_neg_indices_.clear();
		}
		/************************************************************************/


		// Find matches between source bboxes and ground truth bboxes.
		vector<map<int, vector<float> > > all_match_overlaps;
		FindMatches(all_loc_preds, all_gt_bboxes, prior_bboxes, prior_variances,
			multibox_loss_param_, &all_match_overlaps, &all_match_indices_);

#ifdef BOX_LIST
		/**********************************************************************************************/
		const char* model = "all"; // ѡ��ֻͳ��IOU<0.5��bbox����ͳ��ȫ��ƥ��
		const float overlap_threshold = 0.5; // IOUͳ����ֵ
		int num = all_loc_preds.size(); // batch_size
		ofstream  outfile;
		if (_access("temp.txt", 0) != -1) // �����ʱ�ļ����ڣ�ɾ����
			remove("temp.txt");
		/**********************************************************************************************
		* ����ÿ�������������е�prior boxes��ƥ����(<prior box���, ƥ���ground_truth_box���>)�У�
		* ͳ��ÿ��ground_truth_box��ƥ����������ƥ����
		***********************************************************************************************/
		for (int i = 0; i < num; ++i) { // ����ÿ������
			if (all_gt_bboxes.find(i) == all_gt_bboxes.end()){
				continue;// �������������gt_box��������
			}
			map<int, int> match_box_index;
			int num_matches = 0;
			const map<int, vector<int> >& match_indices = all_match_indices_[i];
			const map<int, vector<float> >& match_overlaps = all_match_overlaps[i];
			map<int, vector<int> >::const_iterator it;
			map<int, vector<float> >::const_iterator it_overlaps;
			for (it = match_indices.begin(), it_overlaps = match_overlaps.begin();
				it != match_indices.end(); ++it) {
				const vector<int>& match_index = it->second;
				const vector<float>& match_overlap = it_overlaps->second;
				for (int m = 0; m < match_index.size(); ++m) { //����ÿһ��pripor boxƥ����
					if (match_index[m] <= -1)
					{
						continue;
					}
					const int gt_idx = match_index[m];//ƥ���ground box���

					/**********************************************************************************************
					* ����ÿһ��ƥ�䣬���ƥ���С����ֵ�����ӡ����������λ�ã�
					* <status> <prior_with_gt_IOU> <proir_box_num> <prior_box_cordinate> <num_of_one_batch> <gt_box_num>
					***********************************************************************************************/
					if (strcmp(model, "small") == 0)  // ֻͳ��IOU<0.5��bbox
					{
						outfile.open("temp.txt", ios::out | ios::app);
						if (match_overlap[m] < overlap_threshold)
						{
							/*const NormalizedBBox& gt_bbox = all_gt_bboxes.find(i)->second[gt_idx];*/
							outfile << "small " << match_overlap[m] << " "
								<< m << " "
								<< prior_bboxes[m].xmin() << " "
								<< prior_bboxes[m].ymin() << " "
								<< prior_bboxes[m].xmax() << " "
								<< prior_bboxes[m].ymax() << " "
								<< i << " " << gt_idx << endl;
						}
						outfile.close();
					}
					else if (strcmp(model, "all") == 0) // ͳ������ƥ���bbox
					{
						outfile.open("temp.txt", ios::out | ios::app);
						if (match_overlap[m] < overlap_threshold)
						{
							/*const NormalizedBBox& gt_bbox = all_gt_bboxes.find(i)->second[gt_idx];*/
							outfile << "small " << match_overlap[m] << " "
								<< m << " "
								<< prior_bboxes[m].xmin() << " "
								<< prior_bboxes[m].ymin() << " "
								<< prior_bboxes[m].xmax() << " "
								<< prior_bboxes[m].ymax() << " "
								<< i << " " << gt_idx << endl;
						}
						else
						{
							/*const NormalizedBBox& gt_bbox = all_gt_bboxes.find(i)->second[gt_idx];*/
							outfile << "biger " << match_overlap[m] << " "
								<< m << " "
								<< prior_bboxes[m].xmin() << " "
								<< prior_bboxes[m].ymin() << " "
								<< prior_bboxes[m].xmax() << " "
								<< prior_bboxes[m].ymax() << " "
								<< i << " " << gt_idx << endl;
						}
						outfile.close();
					}
					match_box_index[gt_idx]++;
				}
			}
			/**********************************************************************************************
			* ���ڰ���gt_box�������е�ÿһ��gt_box�����δ��ƥ�䣬���ӡ����������λ�ã�
			* ����ƥ��̶ȡ�������һ��batch�����ź͸�ground_truth_box����������ź�λ��
			***********************************************************************************************/
			for (int j = 0; j < all_gt_bboxes.find(i)->second.size(); j++)
			{
				if (match_box_index[j] == 0)
				{
					const NormalizedBBox& not_match_gt_bbox = all_gt_bboxes.find(i)->second[j];
					cout << "null" << " 0 " << i << " " << j << " "
						<< not_match_gt_bbox.xmin() << " "
						<< not_match_gt_bbox.ymin() << " "
						<< not_match_gt_bbox.xmax() << " "
						<< not_match_gt_bbox.ymax() << " "
						<< not_match_gt_bbox.size() << endl;
				}
			}
		}
		/**********************************************************************************************/
#endif // BOX_LIST

		num_matches_ = 0;
		int num_negs = 0;
		// Sample hard negative (and positive) examples based on mining type.
		MineHardExamples(*bottom[1], all_loc_preds, all_gt_bboxes, prior_bboxes,
			prior_variances, all_match_overlaps, multibox_loss_param_,
			&num_matches_, &num_negs, &all_match_indices_,
			&all_neg_indices_);

		if (top.size() > 1) {
			vector<vector<Dtype>> all_conf_preds;
#ifdef CPU_ONLY
			GetConfPredictions(bottom[1]->cpu_data(), num_, num_priors_, multibox_loss_param_.num_classes(),
				multibox_loss_param_.background_label_id(), multibox_loss_param_.mining_type(),
				all_match_indices_, all_gt_bboxes, &all_conf_preds);
#else
			GetConfPredictionsGPU(*bottom[1], num_, num_priors_, multibox_loss_param_.num_classes(),
				multibox_loss_param_.background_label_id(), multibox_loss_param_.mining_type(),
				all_match_indices_, all_gt_bboxes, &all_conf_preds);
#endif
			top[1]->Reshape(num_*num_priors_, 1, 1, 1);//score
			top[2]->Reshape(num_*num_priors_, 2, 1, 1);//match info
			//top[3]->Reshape(num_*num_priors_, 1, 1, 1);//match IOU
			for (int batch_index = 0; batch_index < num_; batch_index++) {
				int match_begin = batch_index * num_priors_;
				for (int priors_index = 0; priors_index < num_priors_; priors_index++) {
					top[1]->mutable_cpu_data()[match_begin + priors_index] = all_conf_preds[batch_index][priors_index];//score
					top[2]->mutable_cpu_data()[match_begin * 2 + priors_index * 2] = all_match_indices_[batch_index][-1][priors_index];//gt index
					top[2]->mutable_cpu_data()[match_begin * 2 + priors_index * 2 + 1] = all_match_overlaps[batch_index][-1][priors_index];//IOU
				}
			}
		}
		if (num_matches_ >= 1) {
			// Form data to pass on to loc_loss_layer_.
			vector<int> loc_shape(2);
			loc_shape[0] = 1;
			loc_shape[1] = num_matches_ * 4;
			loc_pred_.Reshape(loc_shape);
			loc_gt_.Reshape(loc_shape);
			Dtype* loc_pred_data = loc_pred_.mutable_cpu_data();
			Dtype* loc_gt_data = loc_gt_.mutable_cpu_data();
			EncodeLocPrediction(all_loc_preds, all_gt_bboxes, all_match_indices_,
				prior_bboxes, prior_variances, multibox_loss_param_,
				loc_pred_data, loc_gt_data);
			loc_loss_layer_->Reshape(loc_bottom_vec_, loc_top_vec_);
			loc_loss_layer_->Forward(loc_bottom_vec_, loc_top_vec_);
		}
		else {
			loc_loss_.mutable_cpu_data()[0] = 0;
		}

		// Form data to pass on to conf_loss_layer_.
		if (do_neg_mining_) {
			num_conf_ = num_matches_ + num_negs;
		}
		else {
			num_conf_ = num_ * num_priors_;
		}
		if (num_conf_ >= 1) {
			// Reshape the confidence data.
			vector<int> conf_shape;
			if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
				conf_shape.push_back(num_conf_);
				conf_gt_.Reshape(conf_shape);
				conf_shape.push_back(num_classes_);
				conf_pred_.Reshape(conf_shape);
			}
			else if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
				conf_shape.push_back(1);
				conf_shape.push_back(num_conf_);
				conf_shape.push_back(num_classes_);
				conf_gt_.Reshape(conf_shape);
				conf_pred_.Reshape(conf_shape);
			}
			/*****************************************************************************/
			else if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_FocalLoss) {
				conf_shape.push_back(num_conf_);
				conf_gt_.Reshape(conf_shape);
				conf_shape.push_back(num_classes_);
				conf_pred_.Reshape(conf_shape);
			}
			/*****************************************************************************/
			else {
				LOG(FATAL) << "Unknown confidence loss type.";
			}
			if (!do_neg_mining_) {
				// Consider all scores.
				// Share data and diff with bottom[1].
				CHECK_EQ(conf_pred_.count(), bottom[1]->count());
				conf_pred_.ShareData(*(bottom[1]));
			}
			Dtype* conf_pred_data = conf_pred_.mutable_cpu_data();
			Dtype* conf_gt_data = conf_gt_.mutable_cpu_data();
			caffe_set(conf_gt_.count(), Dtype(background_label_id_), conf_gt_data);
			EncodeConfPrediction(conf_data, num_, num_priors_, multibox_loss_param_,
				all_match_indices_, all_neg_indices_, all_gt_bboxes,
				conf_pred_data, conf_gt_data);
			conf_loss_layer_->Reshape(conf_bottom_vec_, conf_top_vec_);
			conf_loss_layer_->Forward(conf_bottom_vec_, conf_top_vec_);
		}
		else {
			conf_loss_.mutable_cpu_data()[0] = 0;
		}

		top[0]->mutable_cpu_data()[0] = 0;
		if (this->layer_param_.propagate_down(0)) {
			Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
				normalization_, num_, num_priors_, num_matches_);
			top[0]->mutable_cpu_data()[0] +=
				loc_weight_ * loc_loss_.cpu_data()[0] / normalizer;
		}
		if (this->layer_param_.propagate_down(1)) {
			Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
				normalization_, num_, num_priors_, num_matches_);
			top[0]->mutable_cpu_data()[0] += conf_loss_.cpu_data()[0] / normalizer;
		}
	}

	template <typename Dtype>
	void MultiBoxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {

		if (propagate_down[2]) {
			LOG(FATAL) << this->type()
				<< " Layer cannot backpropagate to prior inputs.";
		}
		if (propagate_down[3]) {
			LOG(FATAL) << this->type()
				<< " Layer cannot backpropagate to label inputs.";
		}

		// Back propagate on location prediction.
		if (propagate_down[0]) {
			Dtype* loc_bottom_diff = bottom[0]->mutable_cpu_diff();
			caffe_set(bottom[0]->count(), Dtype(0), loc_bottom_diff);
			if (num_matches_ >= 1) {
				vector<bool> loc_propagate_down;
				// Only back propagate on prediction, not ground truth.
				loc_propagate_down.push_back(true);
				loc_propagate_down.push_back(false);
				loc_loss_layer_->Backward(loc_top_vec_, loc_propagate_down,
					loc_bottom_vec_);
				// Scale gradient.
				Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
					normalization_, num_, num_priors_, num_matches_);
				Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
				caffe_scal(loc_pred_.count(), loss_weight, loc_pred_.mutable_cpu_diff());
				// Copy gradient back to bottom[0].
				const Dtype* loc_pred_diff = loc_pred_.cpu_diff();
				int count = 0;
				for (int i = 0; i < num_; ++i) {
					for (map<int, vector<int> >::iterator it =
						all_match_indices_[i].begin();
						it != all_match_indices_[i].end(); ++it) {
						const int label = share_location_ ? 0 : it->first;
						const vector<int>& match_index = it->second;
						for (int j = 0; j < match_index.size(); ++j) {
							if (match_index[j] <= -1) {
								continue;
							}
							// Copy the diff to the right place.
							int start_idx = loc_classes_ * 4 * j + label * 4;
							caffe_copy<Dtype>(4, loc_pred_diff + count * 4,
								loc_bottom_diff + start_idx);
							++count;
						}
					}
					loc_bottom_diff += bottom[0]->offset(1);
				}
			}
		}

		// Back propagate on confidence prediction.
		if (propagate_down[1]) {
			Dtype* conf_bottom_diff = bottom[1]->mutable_cpu_diff();
			caffe_set(bottom[1]->count(), Dtype(0), conf_bottom_diff);
			if (num_conf_ >= 1) {
				vector<bool> conf_propagate_down;
				// Only back propagate on prediction, not ground truth.
				conf_propagate_down.push_back(true);
				conf_propagate_down.push_back(false);
				conf_loss_layer_->Backward(conf_top_vec_, conf_propagate_down,
					conf_bottom_vec_);
				// Scale gradient.
				Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
					normalization_, num_, num_priors_, num_matches_);
				Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
				caffe_scal(conf_pred_.count(), loss_weight,
					conf_pred_.mutable_cpu_diff());
				// Copy gradient back to bottom[1].
				const Dtype* conf_pred_diff = conf_pred_.cpu_diff();
				if (do_neg_mining_) {
					int count = 0;
					for (int i = 0; i < num_; ++i) {
						// Copy matched (positive) bboxes scores' diff.
						const map<int, vector<int> >& match_indices = all_match_indices_[i];
						for (map<int, vector<int> >::const_iterator it =
							match_indices.begin(); it != match_indices.end(); ++it) {
							const vector<int>& match_index = it->second;
							CHECK_EQ(match_index.size(), num_priors_);
							for (int j = 0; j < num_priors_; ++j) {
								if (match_index[j] <= -1) {
									continue;
								}
								// Copy the diff to the right place.
								caffe_copy<Dtype>(num_classes_,
									conf_pred_diff + count * num_classes_,
									conf_bottom_diff + j * num_classes_);
								++count;
							}
						}
						// Copy negative bboxes scores' diff.
						for (int n = 0; n < all_neg_indices_[i].size(); ++n) {
							int j = all_neg_indices_[i][n];
							CHECK_LT(j, num_priors_);
							caffe_copy<Dtype>(num_classes_,
								conf_pred_diff + count * num_classes_,
								conf_bottom_diff + j * num_classes_);
							++count;
						}
						conf_bottom_diff += bottom[1]->offset(1);
					}
				}
				else {
					// The diff is already computed and stored.
					bottom[1]->ShareDiff(conf_pred_);
				}
			}
		}

		// After backward, remove match statistics.
		all_match_indices_.clear();
		all_neg_indices_.clear();
	}

	INSTANTIATE_CLASS(MultiBoxLossLayer);
	REGISTER_LAYER_CLASS(MultiBoxLoss);

}  // namespace caffe
