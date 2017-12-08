// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------
#include <thrust/system/cuda/detail/cub/cub.cuh>
#include <iomanip>
#include <io.h>
#include "caffe/FRCNN/frcnn_proposal_layer.hpp"
#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/FRCNN/util/frcnn_helper.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"  
#include "caffe/FRCNN/util/frcnn_gpu_nms.hpp"  
#include <iostream>
namespace caffe {

	namespace Frcnn {

		using std::vector;

		__global__ void GetIndex(const int n, int *indices){
			CUDA_KERNEL_LOOP(index, n){
				indices[index] = index;
			}
		}

		template <typename Dtype>
		__global__ void BBoxTransformInv(const int nthreads, const Dtype* const bottom_rpn_bbox,
			const float im_height, const float im_width, const int* sorted_indices,
			const Dtype* const prior_bboxes, const Dtype* const prior_variances, float* const transform_bbox,
			const Dtype* const match_info) {
			CUDA_KERNEL_LOOP(index, nthreads) {
				const int score_idx = sorted_indices[index];
				int match_gt_index = match_info[2 * score_idx];
				float *box = transform_bbox + index * 7;
				if (match_gt_index < -1){ //丢弃的负样本
					box[0] = 0;
					box[1] = 0;
					box[2] = 0;
					box[3] = 0;
					box[4] = score_idx;
					box[5] = match_info[score_idx * 2];
					box[6] = match_info[score_idx * 2 + 1];
				}
				else
				{
					box[0] = prior_bboxes[score_idx * 4 + 0];
					box[1] = prior_bboxes[score_idx * 4 + 1];
					box[2] = prior_bboxes[score_idx * 4 + 2];
					box[3] = prior_bboxes[score_idx * 4 + 3];
					box[4] = score_idx;
					box[5] = match_info[score_idx * 2];
					box[6] = match_info[score_idx * 2 + 1];
					if (match_gt_index < 0){ // 保留的负样本
						box[0] = max(0.0f, min(box[0] * im_width, im_width)); // im_width - 1.0
						box[1] = max(0.0f, min(box[1] * im_height, im_height)); // im_height - 1.0
						box[2] = max(0.0f, min(box[2] * im_width, im_width)); // im_width - 1.0
						box[3] = max(0.0f, min(box[3] * im_height, im_height)); // im_height - 1.0
					}
					else
					{
						Dtype det[4] = {
							det[0] = bottom_rpn_bbox[score_idx * 4 + 0],
							det[1] = bottom_rpn_bbox[score_idx * 4 + 1],
							det[2] = bottom_rpn_bbox[score_idx * 4 + 2],
							det[3] = bottom_rpn_bbox[score_idx * 4 + 3]
						};
						float src_w = box[2] - box[0];// + 1 / im_width;
						float src_h = box[3] - box[1];// + 1 / im_height;
						float src_ctr_x = box[0] + 0.5 * src_w;
						float src_ctr_y = box[1] + 0.5 * src_h;
						float pred_ctr_x = prior_variances[score_idx * 4] * det[0] * src_w + src_ctr_x;
						float pred_ctr_y = prior_variances[score_idx * 4 + 1] * det[1] * src_h + src_ctr_y;
						float pred_w = exp(prior_variances[score_idx * 4 + 2] * det[2]) * src_w;
						float pred_h = exp(prior_variances[score_idx * 4 + 3] * det[3]) * src_h;
						box[0] = (pred_ctr_x - 0.5 * pred_w)*im_width;;
						box[1] = (pred_ctr_y - 0.5 * pred_h)*im_height;;
						box[2] = (pred_ctr_x + 0.5 * pred_w)*im_width;;
						box[3] = (pred_ctr_y + 0.5 * pred_h)*im_height;;
						box[0] = max(0.0f, min(box[0], im_width)); // im_width - 1.0
						box[1] = max(0.0f, min(box[1], im_height)); // im_height - 1.0
						box[2] = max(0.0f, min(box[2], im_width)); // im_width - 1.0
						box[3] = max(0.0f, min(box[3], im_height)); // im_height - 1.0
					}
				}
			}
		}

		__global__ void SelectBox(const int nthreads, const float *box, float min_size,
			int *flags) {
			CUDA_KERNEL_LOOP(index, nthreads) {
				if ((box[index * 7 + 2] - box[index * 7 + 0] < min_size) ||
					(box[index * 7 + 3] - box[index * 7 + 1] < min_size)) {
					flags[index] = 0;
				}
				else {
					flags[index] = 1;
				}
			}
		}

		template <typename Dtype>
		__global__ void SelectBoxByIndices(const int nthreads, const float *in_box, int *selected_indices,
			float *out_box, const Dtype *in_score, Dtype *out_score) {
			CUDA_KERNEL_LOOP(index, nthreads) {
				if ((index == 0 && selected_indices[index] == 1) ||
					(index > 0 && selected_indices[index] == selected_indices[index - 1] + 1)) {
					out_box[(selected_indices[index] - 1) * 7 + 0] = in_box[index * 7 + 0];
					out_box[(selected_indices[index] - 1) * 7 + 1] = in_box[index * 7 + 1];
					out_box[(selected_indices[index] - 1) * 7 + 2] = in_box[index * 7 + 2];
					out_box[(selected_indices[index] - 1) * 7 + 3] = in_box[index * 7 + 3];
					out_box[(selected_indices[index] - 1) * 7 + 4] = in_box[index * 7 + 4];
					out_box[(selected_indices[index] - 1) * 7 + 5] = in_box[index * 7 + 5];
					out_box[(selected_indices[index] - 1) * 7 + 6] = in_box[index * 7 + 6];
					if (in_score != NULL && out_score != NULL) {
						out_score[selected_indices[index] - 1] = in_score[index];
					}
				}
			}
		}

		template <typename Dtype>
		__global__ void SelectBoxAftNMS(int batch_index, int box_begin, const int nthreads, const float *in_box, int *keep_indices,
			Dtype *top_data, const Dtype *in_score, Dtype* top_info) {
			CUDA_KERNEL_LOOP(index, nthreads) {
				int keep_idx = keep_indices[index];
				top_data[box_begin * 5 + index * 5] = batch_index;// batch_index
				for (int j = 1; j < 5; ++j) {
					top_data[box_begin * 5 + index * 5 + j] = in_box[keep_idx * 7 + j - 1];
				}
				if (top_info != NULL) {
					top_info[box_begin * 2 + index * 2] = in_box[keep_idx * 7 + 4];
					top_info[box_begin * 2 + index * 2 + 1] = in_box[keep_idx * 7 + 5];
					top_info[box_begin * 2 + index * 2 + 2] = in_box[keep_idx * 7 + 6];
				}
			}
		}

		template <typename Dtype>
		void FrcnnProposalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
			const vector<Blob<Dtype> *> &top) {

			DLOG(ERROR) << "========== enter proposal layer";
			const Dtype *bottom_rpn_score = bottom[0]->gpu_data();
			const Dtype *bottom_rpn_bbox = bottom[1]->gpu_data();
			const Dtype *prior_data = bottom[2]->gpu_data();    // prior box
			const Dtype* match_info = bottom[3]->gpu_data(); // match_info
			//const Dtype* match_iou = bottom[4]->gpu_data(); // match_iou_info
			// bottom data comes from host memory
			/*Dtype bottom_im_info[3];
			CHECK_EQ(bottom[2]->count(), 3);
			CUDA_CHECK(cudaMemcpy(bottom_im_info, bottom[2]->gpu_data(), sizeof(Dtype) * 3, cudaMemcpyDeviceToHost));*/

			const int num = bottom[1]->num();// batch size
			const int channes = bottom[1]->channels();
			const int height = bottom[1]->height();
			const int width = bottom[1]->width();
			/*-------------------------改写-------------------------*/
			CHECK(channes % 4 == 0) << "rpn bbox pred channels should be divided by 4";

			const float im_height = FrcnnParam::im_height;
			const float im_width = FrcnnParam::im_width;
			/*-------------------------改写-------------------------*/

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
				rpn_min_size = FrcnnParam::test_rpn_min_size;
			}
			LOG_IF(ERROR, rpn_pre_nms_top_n <= 0) << "rpn_pre_nms_top_n : " << rpn_pre_nms_top_n;
			LOG_IF(ERROR, rpn_post_nms_top_n <= 0) << "rpn_post_nms_top_n : " << rpn_post_nms_top_n;
			if (rpn_pre_nms_top_n <= 0 || rpn_post_nms_top_n <= 0) return;

			const int config_n_anchors = FrcnnParam::anchors.size() / 4;
			const int total_anchor_num = bottom[2]->height() / 4;
			/*-------------------------改写-------------------------*/
			vector<int> batch_keep_num;
			vector<float*> batch_transform_bbox_;
			vector<int *>batch_gpu_keep_indices_;
			vector<Dtype *>batch_bbox_score_;
			vector<int*> batch_match_gt_;// add output

			for (int batch_index = 0; batch_index < num; batch_index++) {
				//Step 1. -------------------------------Sort the rpn result----------------------
				// the first half of rpn_score is the bg score
				// Note that the sorting operator will change the order fg_scores (bottom_rpn_score)
				const int fg_begin = batch_index * total_anchor_num;
				const int bbox_begin = (4 * batch_index)*total_anchor_num;
				const int transform_bbox_begin = 7 * batch_index * rpn_pre_nms_top_n;//
				const int selected_flags_begin = batch_index * rpn_pre_nms_top_n;
				const int gpu_keep_indices_begin = batch_index * rpn_post_nms_top_n;

				Dtype *fg_scores = (Dtype*)(&bottom_rpn_score[fg_begin]);
				Dtype *rpn_bbox = (Dtype*)(&bottom_rpn_bbox[bbox_begin]);
				Dtype *fg_info = (Dtype*)(&match_info[fg_begin]); //匹配的信息

				Dtype *sorted_scores = NULL;
				CUDA_CHECK(cudaMalloc((void**)&sorted_scores, sizeof(Dtype) * total_anchor_num));
				cub::DoubleBuffer<Dtype> d_keys(fg_scores, sorted_scores);

				int *indices = NULL;
				CUDA_CHECK(cudaMalloc((void**)&indices, sizeof(int) * total_anchor_num));
				GetIndex << <caffe::CAFFE_GET_BLOCKS(total_anchor_num), caffe::CAFFE_CUDA_NUM_THREADS >> >(
					total_anchor_num, indices);
				cudaDeviceSynchronize();

				int *sorted_indices = NULL;
				CUDA_CHECK(cudaMalloc((void**)&sorted_indices, sizeof(int) * total_anchor_num));
				cub::DoubleBuffer<int> d_values(indices, sorted_indices);

				void *sort_temp_storage_ = NULL;
				size_t sort_temp_storage_bytes_ = 0;
				// calculate the temp_storage_bytes
				cub::DeviceRadixSort::SortPairsDescending(sort_temp_storage_, sort_temp_storage_bytes_,
					d_keys, d_values, total_anchor_num);
				DLOG(ERROR) << "sort_temp_storage_bytes_ : " << sort_temp_storage_bytes_;
				CUDA_CHECK(cudaMalloc(&sort_temp_storage_, sort_temp_storage_bytes_));
				// sorting
				cub::DeviceRadixSort::SortPairsDescending(sort_temp_storage_, sort_temp_storage_bytes_,
					d_keys, d_values, total_anchor_num);
				cudaDeviceSynchronize();

				//Step 2. ---------------------------bbox transform----------------------------
				const int retained_anchor_num = std::min(total_anchor_num, rpn_pre_nms_top_n);//3000?
				// 这里将匹配gt=-2的box置0，如此filter out small box时会过滤掉
				BBoxTransformInv<Dtype> << <caffe::CAFFE_GET_BLOCKS(retained_anchor_num), caffe::CAFFE_CUDA_NUM_THREADS >> >(
					retained_anchor_num, rpn_bbox, im_height, im_width, sorted_indices,
					prior_data, &prior_data[total_anchor_num * 4], &transform_bbox_[transform_bbox_begin],
					fg_info);// 只存rpn_pre_nms_top_n个，这rpn_pre_nms_top_n个里面还有小于rpn_min_size的
				cudaDeviceSynchronize();
				/*-------------------------验证代码-------------------------*/
				//if (retained_anchor_num > 0) {
				//	std::ofstream  outfile;
				//	if (_access("anchor_before_nmsGPU.txt", 0) != -1) // 如果临时文件存在，删除！
				//		remove("anchor_before_nmsGPU.txt");
				//	outfile.open("anchor_before_nmsGPU.txt", ios::out | ios::app);
				//	vector<float> boxes(rpn_pre_nms_top_n * 7);
				//	vector<Dtype> scores(total_anchor_num);
				//	/*vector<int> indices(total_anchor_num);*/
				//	cudaMemcpy(&boxes[0], &transform_bbox_[transform_bbox_begin], 7 * rpn_pre_nms_top_n * sizeof(float), cudaMemcpyDeviceToHost);
				//	cudaMemcpy(&scores[0], sorted_scores, total_anchor_num * sizeof(Dtype), cudaMemcpyDeviceToHost);
				//	/*cudaMemcpy(&indices[0], sorted_indices, total_anchor_num * sizeof(int), cudaMemcpyDeviceToHost);*/
				//	outfile << retained_anchor_num << std::endl;
				//	for (int i = 0; i < retained_anchor_num; i++) {
				//		outfile << scores[i] << " ";
				//		outfile << boxes[7 * i] << " "
				//			<< boxes[7 * i + 1] << " "
				//			<< boxes[7 * i + 2] << " "
				//			<< boxes[7 * i + 3] << " "
				//			<< boxes[7 * i + 4] << " "
				//      << boxes[7 * i + 5] << " "
				//			<< boxes[7 * i + 6] << std::endl;
				//	}
				//	outfile.close();
				//}
				/*-------------------------验证代码-------------------------*/

				//Step 3. -------------------------filter out small box-----------------------
				SelectBox << <caffe::CAFFE_GET_BLOCKS(retained_anchor_num), caffe::CAFFE_CUDA_NUM_THREADS >> >(
					retained_anchor_num, &transform_bbox_[transform_bbox_begin], rpn_min_size, &selected_flags_[selected_flags_begin]);
				cudaDeviceSynchronize();
				/*-------------------------验证代码-------------------------*/
				//if (selected_flags_ != NULL)
				//{
				//	vector<int> flags(retained_anchor_num);
				//	cudaMemcpy(&flags[0], &selected_flags_[selected_flags_begin], retained_anchor_num * sizeof(int), cudaMemcpyDeviceToHost);
				//	std::ofstream  outfile;
				//	if (_access("flags_GPU.txt", 0) != -1) // 如果临时文件存在，删除！
				//		remove("flags_GPU.txt");
				//	outfile.open("flags_GPU.txt", ios::out | ios::app);
				//	for (int i = 0; i < retained_anchor_num; i++) {
				//		outfile << flags[i] << std::endl;
				//	}
				//	outfile.close();
				//}
				/*-------------------------验证代码-------------------------*/

				// cumulative sum up the flags to get the copy index
				int *selected_indices_ = NULL;
				CUDA_CHECK(cudaMalloc((void**)&selected_indices_, sizeof(int) * retained_anchor_num));
				void *cumsum_temp_storage_ = NULL;
				size_t cumsum_temp_storage_bytes_ = 0;
				cub::DeviceScan::InclusiveSum(cumsum_temp_storage_, cumsum_temp_storage_bytes_,
					&selected_flags_[selected_flags_begin], selected_indices_, retained_anchor_num);
				DLOG(ERROR) << "cumsum_temp_storage_bytes : " << cumsum_temp_storage_bytes_;
				CUDA_CHECK(cudaMalloc(&cumsum_temp_storage_, cumsum_temp_storage_bytes_));
				cub::DeviceScan::InclusiveSum(sort_temp_storage_, cumsum_temp_storage_bytes_,
					&selected_flags_[selected_flags_begin], selected_indices_, retained_anchor_num);

				/*-------------------------验证代码-------------------------*/
				//if (selected_indices_ != NULL)
				//{
				//	vector<int> indices_(retained_anchor_num);
				//	cudaMemcpy(&indices_[0], &selected_indices_[selected_flags_begin], retained_anchor_num * sizeof(int), cudaMemcpyDeviceToHost);
				//	std::ofstream  outfile;
				//	if (_access("indices_.txt", 0) != -1) // 如果临时文件存在，删除！
				//		remove("indices_.txt");
				//	outfile.open("indices_.txt", ios::out | ios::app);
				//	for (int i = 0; i < retained_anchor_num; i++) {
				//		outfile << indices_[i] << std::endl;
				//	}
				//	outfile.close();
				//}
				/*-------------------------验证代码-------------------------*/

				int selected_num = -1;
				cudaMemcpy(&selected_num, &selected_indices_[retained_anchor_num - 1], sizeof(int), cudaMemcpyDeviceToHost);
				CHECK_GT(selected_num, 0);

				float* tmp_transform_bbox = NULL;
				CUDA_CHECK(cudaMalloc(&tmp_transform_bbox, 7 * sizeof(Dtype) * rpn_pre_nms_top_n));//修改retained_anchor_num
				cudaMemcpy(tmp_transform_bbox, &transform_bbox_[transform_bbox_begin], rpn_pre_nms_top_n * sizeof(Dtype) * 7, cudaMemcpyDeviceToDevice);

				Dtype *bbox_score_ = NULL;
				if (top.size() > 1)
				{
					CUDA_CHECK(cudaMalloc(&bbox_score_, sizeof(Dtype) * rpn_pre_nms_top_n));//修改retained_anchor_num
				}
				SelectBoxByIndices << <caffe::CAFFE_GET_BLOCKS(retained_anchor_num), caffe::CAFFE_CUDA_NUM_THREADS >> >(
					retained_anchor_num, &tmp_transform_bbox[transform_bbox_begin], &selected_indices_[selected_flags_begin], &transform_bbox_[transform_bbox_begin], sorted_scores, bbox_score_);
				cudaDeviceSynchronize();

				/*-------------------------验证代码-------------------------*/
				if (selected_num > 0) {
					std::ofstream  outfile;
					if (_access("anchor_before_nmsGPU.txt", 0) != -1) // 如果临时文件存在，删除！
						remove("anchor_before_nmsGPU.txt");
					outfile.open("anchor_before_nmsGPU.txt", ios::out | ios::app);
					vector<float> boxes(retained_anchor_num * 7);
					vector<float> scores(retained_anchor_num);
					/*vector<int> flags(retained_anchor_num);
					cudaMemcpy(&flags[0], &selected_flags_[selected_flags_begin], retained_anchor_num * sizeof(int), cudaMemcpyDeviceToHost);*/
					cudaMemcpy(&boxes[0], &transform_bbox_[transform_bbox_begin], 7 * retained_anchor_num * sizeof(float), cudaMemcpyDeviceToHost);
					cudaMemcpy(&scores[0], bbox_score_, retained_anchor_num * sizeof(float), cudaMemcpyDeviceToHost);
					/*vector<float>::iterator itBox;
					vector<float>::iterator itScore;
					vector<int>::iterator itFlag;
					for (itFlag = flags.begin(), itBox = boxes.begin(), itScore = scores.begin(); itFlag != flags.end();)
					{
						if ((*itFlag) == 0)
						{
							itFlag = flags.erase(itFlag);
							itBox = boxes.erase(itBox);
							itBox = boxes.erase(itBox);
							itBox = boxes.erase(itBox);
							itBox = boxes.erase(itBox);
							itBox = boxes.erase(itBox);
							itBox = boxes.erase(itBox);
							itBox = boxes.erase(itBox);
							itScore = scores.erase(itScore);
						}
						else
						{
							itFlag++;
							itBox = itBox + 7;
							itScore++;
						}
					}
*/
					outfile << selected_num << std::endl;
					for (int i = 0; i < selected_num; i++) {
						outfile << scores[i] << " ";
						outfile << boxes[7 * i] << " "
							<< boxes[7 * i + 1] << " "
							<< boxes[7 * i + 2] << " "
							<< boxes[7 * i + 3] << " "
							<< boxes[7 * i + 4] << " "
							<< boxes[7 * i + 5] << " "
							<< boxes[7 * i + 6] << std::endl;
					}
					outfile.close();
				}
				/*-------------------------验证代码-------------------------*/

				//Step 4. -----------------------------apply nms-------------------------------
				DLOG(ERROR) << "========== apply nms with rpn_nms_thresh : " << rpn_nms_thresh;
				vector<int> keep_indices(selected_num);
				int keep_num = -1;
				gpu_nms(&keep_indices[0], &keep_num, &transform_bbox_[transform_bbox_begin], selected_num, 4, rpn_nms_thresh);
				DLOG(ERROR) << "rpn num after gpu nms: " << keep_num;

				keep_num = std::min(keep_num, rpn_post_nms_top_n);
				DLOG(ERROR) << "========== copy to top";
				cudaMemcpy(&gpu_keep_indices_[gpu_keep_indices_begin], &keep_indices[0], sizeof(int) * keep_num, cudaMemcpyHostToDevice);

				////////////////////////////////////
				// do not forget to free the malloc memory
				CUDA_CHECK(cudaFree(sorted_scores));
				CUDA_CHECK(cudaFree(indices));
				CUDA_CHECK(cudaFree(sorted_indices));
				CUDA_CHECK(cudaFree(sort_temp_storage_));
				CUDA_CHECK(cudaFree(cumsum_temp_storage_));
				CUDA_CHECK(cudaFree(selected_indices_));

				batch_keep_num.push_back(keep_num);
				batch_bbox_score_.push_back(bbox_score_);
			}

			int total_boxes = 0;
			for (size_t batch_index = 0; batch_index < batch_keep_num.size(); batch_index++) {
				total_boxes += batch_keep_num[batch_index];
			}

			top[0]->Reshape(total_boxes, 5, 1, 1);
			Dtype *top_data = top[0]->mutable_gpu_data();
			Dtype *top_info = NULL;
			if (top.size() > 1) {
				top[1]->Reshape(total_boxes, 3, 1, 1);
				top_info = top[1]->mutable_gpu_data();
			}
			int box_begin = 0;
			/*-------------------------验证代码-------------------------*/
			std::ofstream  outfile;
			if (_access("frcnn_proposal_layer_outputGPU.txt", 0) != -1) // 如果临时文件存在，删除！
				remove("frcnn_proposal_layer_outputGPU.txt");
			outfile.open("frcnn_proposal_layer_outputGPU.txt", ios::out | ios::app);
			/*-------------------------验证代码-------------------------*/
			for (size_t batch_index = 0; batch_index < batch_keep_num.size(); batch_index++) {
				const int keep_num = batch_keep_num[batch_index];
				SelectBoxAftNMS << <caffe::CAFFE_GET_BLOCKS(keep_num), caffe::CAFFE_CUDA_NUM_THREADS >> >(
					batch_index, box_begin, keep_num,
					&transform_bbox_[rpn_pre_nms_top_n * batch_index * 7],
					&gpu_keep_indices_[rpn_post_nms_top_n * batch_index],
					top_data, batch_bbox_score_[batch_index], top_info);
				/*-------------------------验证代码-------------------------*/
				vector<float> boxes(rpn_pre_nms_top_n * 7);
				vector<Dtype> scores(rpn_pre_nms_top_n);
				vector<int> indexes(keep_num);
				cudaMemcpy(&boxes[0], &transform_bbox_[rpn_pre_nms_top_n * batch_index * 7], 7 * rpn_pre_nms_top_n * sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(&scores[0], batch_bbox_score_[batch_index], rpn_pre_nms_top_n * sizeof(Dtype), cudaMemcpyDeviceToHost);
				cudaMemcpy(&indexes[0], &gpu_keep_indices_[rpn_post_nms_top_n * batch_index], keep_num * sizeof(int), cudaMemcpyDeviceToHost);
				outfile << "batch index : " << batch_index << " " << keep_num << std::endl;
				for (int i = 0; i < keep_num; i++) {
					outfile << scores[indexes[i]] << " ";
					outfile << boxes[7 * indexes[i]] << " "
						<< boxes[7 * indexes[i] + 1] << " "
						<< boxes[7 * indexes[i] + 2] << " "
						<< boxes[7 * indexes[i] + 3] << " "
						<< boxes[7 * indexes[i] + 4] << " "
						<< boxes[7 * indexes[i] + 5] << " "
						<< boxes[7 * indexes[i] + 6] << std::endl;
				}
				/*-------------------------验证代码-------------------------*/
				box_begin += keep_num;
			}
			outfile.close();
			DLOG(ERROR) << "========== exit proposal layer";
			////////////////////////////////////
			// do not forget to free the malloc memory
			for (size_t batch_index = 0; batch_index < batch_keep_num.size(); batch_index++) {
				if (batch_bbox_score_[batch_index] != NULL)
					CUDA_CHECK(cudaFree(batch_bbox_score_[batch_index]));
			}

		}

		template <typename Dtype>
		void FrcnnProposalLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
			const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
			for (int i = 0; i < propagate_down.size(); ++i) {
				if (propagate_down[i]) {
					NOT_IMPLEMENTED;
				}
			}
		}

		INSTANTIATE_LAYER_GPU_FUNCS(FrcnnProposalLayer);

	} // namespace frcnn

} // namespace caffe
