#include "stdafx.h"
#include "Faster_rcnn.h"
#include <algorithm>
cv::Scalar colortable[20] = { cv::Scalar(0, 0, 0), cv::Scalar(0, 0, 125),
cv::Scalar(0, 125, 125), cv::Scalar(125, 125, 125), cv::Scalar(125, 0, 0), cv::Scalar(125, 125, 0), cv::Scalar(0, 125, 0), cv::Scalar(125, 0, 125),
cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 255), cv::Scalar(255, 255, 255), cv::Scalar(255, 0, 0), cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0),
cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 100), cv::Scalar(0, 0, 100),
cv::Scalar(255, 0, 100), cv::Scalar(255, 255, 100), cv::Scalar(100, 100, 100) };
string classname[20] = { "aeroplane", "bike", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
"diningtable", "dog", "horse", "motobike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };
Faster_rcnn::Faster_rcnn()
{

}


Faster_rcnn::~Faster_rcnn()
{
//	if (boxs != NULL)	delete boxs;
}
bool Faster_rcnn::init()
{
	//init
	Caffe::SetDevice(0);
	Caffe::set_mode(Caffe::GPU);
	//Caffe::set_mode(Caffe::CPU);
	loadnet(); // 导入模型网络
	return true;
}
bool Faster_rcnn::loadnet()
{
	//load net
	rpn_net.reset(new Net<float>("faster_rcnn_VOC0712_ZF\\proposal_test.prototxt", caffe::TEST));
	rpn_net->CopyTrainedLayersFrom("faster_rcnn_VOC0712_ZF\\proposal_final");
	faster_rcnn_net.reset(new Net<float>("faster_rcnn_VOC0712_ZF\\detection_test.prototxt", caffe::TEST));
	faster_rcnn_net->CopyTrainedLayersFrom("faster_rcnn_VOC0712_ZF\\detection_final");
	return true;
}
bool Faster_rcnn::imgToBlob()
{
	Mat sample_float;
	m_src.convertTo(sample_float, CV_32FC3);
	cv::Scalar channel_mean = cv::mean(sample_float);
	Mat mean = cv::Mat(m_src.rows, m_src.cols, sample_float.type(), channel_mean);
	Mat sample_normalized;
	subtract(sample_float, mean, sample_normalized);
	prep_im_size();
	resize(sample_normalized, sample_normalized, input_geometry_);
	Blob<float>* input_layer = rpn_net->input_blobs()[0];
	input_layer->Reshape(1, sample_normalized.channels(), sample_normalized.rows, sample_normalized.cols);
	rpn_net->Reshape();
	float* input_data = input_layer->mutable_cpu_data();
	vector<cv::Mat> input_channels;
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(sample_normalized.rows, sample_normalized.cols, CV_32FC1, input_data);
		input_channels.push_back(channel);
		input_data += sample_normalized.rows * sample_normalized.cols;
	}
	cv::split(sample_normalized, input_channels);

	CHECK(reinterpret_cast<float*>(input_channels.at(0).data)
		== rpn_net->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
	return true;
}
bool aboxcomp(abox&b1, abox&b2)
{
	return b1.score > b2.score;
}
vector<abox> Faster_rcnn::forward()
{
	//forward
	const vector<Blob<float>*>& result = rpn_net->Forward();
	Blob<float>* resule0 = result[0];
	Blob<float>* resule1 = result[1];
	Mat boxs_delta(resule0->num()*resule0->channels()*resule0->width()*resule0->height() / 4, 4, CV_32FC1);
	float* p = resule0->mutable_cpu_data();
	int num = 0;
	for (int i = 0; i < resule0->num()*resule0->channels()*resule0->width()*resule0->height() / 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			boxs_delta.at<float>(i, j) = resule0->data_at(0, num%resule0->channels(), 
				(num - num / resule0->channels() / resule0->height() * resule0->channels() * resule0->height()) / resule0->height(), 
				num / resule0->channels() / resule0->height());
			num++;
			//int order = j + i * 4;
			//boxs_delta.at<float>(i, j) = resule0->data_at(0, (order % (resule0->height()*resule0->channels())) % resule0->channels(), (order % (resule0->height()*resule0->channels())) / resule0->channels(), order / (resule0->height()*resule0->channels()));
		}
	}
	//create anchors
	feature_map_size = Size(resule0->width(), resule0->height());
	//prep_im_size();
	Mat anchors = proposal_local_anchor();
	Mat pre_box = bbox_tranform_inv(anchors, boxs_delta, "rpn");
	//Mat score(resule0->width(), resule0->height(), CV_32FC1);
	Mat score = get_rpn_score(resule1, resule0->width(), resule0->height());
	vector<abox> aboxes;
	filter_boxs(pre_box,score,aboxes);
	std::sort(aboxes.begin(), aboxes.end(), aboxcomp);
	//m_sort(pre_box,score);
	vector<int> vPick(aboxes.size());
	int nPick;
	/////////////有cuda版,待加入,此处为cpu版///////
	nms(aboxes,conf.overlap, vPick, nPick);
	vector<abox> aboxes_;
	boxes_filter(aboxes_, nPick, aboxes, vPick);
	return aboxes_;
}
void Faster_rcnn::nms(vector<abox> input_boxes, double overlap, vector<int> &vPick, int &nPick)
{
	int nSample = min(int(input_boxes.size()),conf.per_nms_topN);

	vector<double> vArea(nSample);
	for (int i = 0; i < nSample; ++i)
	{
		vArea[i] = double(input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
			* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
	}

	std::multimap<double, int> scores;
	for (int i = 0; i < nSample; ++i)
		scores.insert(std::pair<double, int>(input_boxes.at(i).score, i));

	nPick = 0;

	do
	{
		int last = scores.rbegin()->second;
		vPick[nPick] = last;
		nPick += 1;

		for ( std::multimap<double, int>::iterator it = scores.begin(); it != scores.end();)
		{
			int it_idx = it->second;
			double xx1 = max(input_boxes.at(last).x1, input_boxes.at(it_idx).x1);
			double yy1 = max(input_boxes.at(last).y1, input_boxes.at(it_idx).y1);
			double xx2 = min(input_boxes.at(last).x2, input_boxes.at(it_idx).x2);
			double yy2 = min(input_boxes.at(last).y2, input_boxes.at(it_idx).y2);

			double w = max(double(0.0), xx2 - xx1 + 1), h = max(double(0.0), yy2 - yy1 + 1);

			double ov = w*h / (vArea[last] + vArea[it_idx] - w*h);

			if (ov > overlap)
			{
				it = scores.erase(it);
			}
			else
			{
				it++;
			}
		}

	} while (scores.size() != 0);
}
void Faster_rcnn::boxes_filter(vector<abox>& aboxes, int nPick, vector<abox> row, vector<int> vPick)
{
	int n = min(nPick, conf.after_nms_topN);
	for (int i = 0; i < n; i++)
	{
		aboxes.push_back(row[vPick[i]]);
	}
}
void Faster_rcnn::filter_boxs(Mat& pre_box, Mat& score, vector<abox>& aboxes)
{
	aboxes.clear();
	for (int i = 0; i < pre_box.rows; i++)
	{
		int widths = pre_box.at<float>(i, 2) - pre_box.at<float>(i, 0) + 1;
		int heights = pre_box.at<float>(i, 3) - pre_box.at<float>(i, 1) + 1;
		if (widths < conf.test_min_box_size || heights < conf.test_min_box_size)
		{
			pre_box.at<float>(i, 0) = 0;
			pre_box.at<float>(i, 1) = 0;
			pre_box.at<float>(i, 2) = 0;
			pre_box.at<float>(i, 3) = 0;
			score.at<float>(i, 0) = 0;
		}
		abox tmp;
		tmp.x1 = pre_box.at<float>(i, 0);
		tmp.y1 = pre_box.at<float>(i, 1);
		tmp.x2 = pre_box.at<float>(i, 2);
		tmp.y2 = pre_box.at<float>(i, 3);
		tmp.score = score.at<float>(i, 0);
		aboxes.push_back(tmp);
	}
}
void Faster_rcnn::m_sort(Mat& pre_box, Mat& score)
{
	for (int i = 0; i < pre_box.rows - 1; i++)
	{
		for (int j = i + 1; j < pre_box.rows; j++)
		{
			if (score.at<float>(i, 0) < score.at<float>(j, 0))
			{
				float tmp = score.at<float>(j, 0);
				score.at<float>(j, 0) = score.at<float>(i, 0);
				score.at<float>(i, 0) = tmp;

				float tmp0 = pre_box.at<float>(j, 0);
				float tmp1 = pre_box.at<float>(j, 1);
				float tmp2 = pre_box.at<float>(j, 2);
				float tmp3 = pre_box.at<float>(j, 3);
				pre_box.at<float>(j, 0) = pre_box.at<float>(i, 0);
				pre_box.at<float>(j, 1) = pre_box.at<float>(i, 1);
				pre_box.at<float>(j, 2) = pre_box.at<float>(i, 2);
				pre_box.at<float>(j, 3) = pre_box.at<float>(i, 3);
				pre_box.at<float>(i, 0) = tmp0; 
				pre_box.at<float>(i, 1) = tmp1;
				pre_box.at<float>(i, 2) = tmp2;
				pre_box.at<float>(i, 3) = tmp3;
			}

		}
	}
}
Mat Faster_rcnn::get_rpn_score(Blob<float>* resule1,int w,int h)
{
	//Blob<float> tmp;
	int channel = resule1->width()*resule1->height() / (w * h);
	Mat score(resule1->width()*resule1->height(), 1, CV_32FC1);
	//tmp.Reshape(1, resule1->width()*resule1->height() / (w * h),h,w);
	//float* p = tmp.mutable_cpu_data;
	int num = 0;
	for (int i = 0; i < w; i++)
	{
		for (int j = 0; j < h; j++)
		{
			for (int k = 0; k < channel; k++)
			{
				score.at<float>(num, 0) = resule1->data_at(0, 1, h*k + j, i);
				num++;
			}
		}
	}
	return score;
}
Mat Faster_rcnn::bbox_tranform_inv(Mat anchors, Mat boxs_delta,string type)
{
	if (type == "rpn")
	{
		Mat pre_box(anchors.rows, anchors.cols, CV_32FC1);
		for (int i = 0; i < anchors.rows; i++)
		{
			double pred_ctr_x, pred_ctr_y, src_ctr_x, src_ctr_y;
			double dst_ctr_x, dst_ctr_y, dst_scl_x, dst_scl_y;
			double src_w, src_h, pred_w, pred_h;
			src_w = anchors.at<float>(i, 2) - anchors.at<float>(i, 0) + 1;
			src_h = anchors.at<float>(i, 3) - anchors.at<float>(i, 1) + 1;
			src_ctr_x = anchors.at<float>(i, 0) + 0.5 * (src_w - 1);
			src_ctr_y = anchors.at<float>(i, 1) + 0.5 * (src_h - 1);

			dst_ctr_x = boxs_delta.at<float>(i, 0);
			dst_ctr_y = boxs_delta.at<float>(i, 1);
			dst_scl_x = boxs_delta.at<float>(i, 2);
			dst_scl_y = boxs_delta.at<float>(i, 3);
			pred_ctr_x = dst_ctr_x*src_w + src_ctr_x;
			pred_ctr_y = dst_ctr_y*src_h + src_ctr_y;
			pred_w = exp(dst_scl_x) * src_w;
			pred_h = exp(dst_scl_y) * src_h;

			pre_box.at<float>(i, 0) = ((pred_ctr_x - 0.5*(pred_w - 1)) - 1) * (float)(m_src.cols) / (im.cols - 1) + 1;
			pre_box.at<float>(i, 1) = ((pred_ctr_y - 0.5*(pred_h - 1)) - 1) * (float)(m_src.rows) / (im.rows - 1) + 1;
			pre_box.at<float>(i, 2) = ((pred_ctr_x + 0.5*(pred_w - 1)) - 1) * (float)(m_src.cols) / (im.cols - 1) + 1;
			pre_box.at<float>(i, 3) = ((pred_ctr_y + 0.5*(pred_h - 1)) - 1) * (float)(m_src.rows) / (im.rows - 1) + 1;
			if (pre_box.at<float>(i, 0) < 0)	pre_box.at<float>(i, 0) = 0;
			if (pre_box.at<float>(i, 0) > (m_src.cols - 1))	pre_box.at<float>(i, 0) = m_src.cols - 1;
			if (pre_box.at<float>(i, 2) < 0)	pre_box.at<float>(i, 2) = 0;
			if (pre_box.at<float>(i, 2) > (m_src.cols - 1))	pre_box.at<float>(i, 2) = m_src.cols - 1;

			if (pre_box.at<float>(i, 1) < 0)	pre_box.at<float>(i, 1) = 0;
			if (pre_box.at<float>(i, 1) > (m_src.rows - 1))	pre_box.at<float>(i, 1) = m_src.rows - 1;
			if (pre_box.at<float>(i, 3) < 0)	pre_box.at<float>(i, 3) = 0;
			if (pre_box.at<float>(i, 3) > (m_src.rows - 1))	pre_box.at<float>(i, 3) = m_src.rows - 1;
		}
		return pre_box;
	}
	if (type == "rcnn")
	{
		Mat pre_box(boxs_delta.rows, boxs_delta.cols, CV_32FC1);
		for (int i = 0; i < boxs_delta.rows; i++)
		{
			for (int j = 1; j < boxs_delta.cols/4; j++)
			{
				double pred_ctr_x, pred_ctr_y, src_ctr_x, src_ctr_y;
				double dst_ctr_x, dst_ctr_y, dst_scl_x, dst_scl_y;
				double src_w, src_h, pred_w, pred_h;
				src_w = anchors.at<float>(i, 2) - anchors.at<float>(i, 0) + 1;
				src_h = anchors.at<float>(i, 3) - anchors.at<float>(i, 1) + 1;
				src_ctr_x = anchors.at<float>(i, 0) + 0.5 * (src_w - 1);
				src_ctr_y = anchors.at<float>(i, 1) + 0.5 * (src_h - 1);

				dst_ctr_x = boxs_delta.at<float>(i, 4 * j + 0);
				dst_ctr_y = boxs_delta.at<float>(i, 4 * j + 1);
				dst_scl_x = boxs_delta.at<float>(i, 4 * j + 2);
				dst_scl_y = boxs_delta.at<float>(i, 4 * j + 3);
				pred_ctr_x = dst_ctr_x*src_w + src_ctr_x;
				pred_ctr_y = dst_ctr_y*src_h + src_ctr_y;
				pred_w = exp(dst_scl_x) * src_w;
				pred_h = exp(dst_scl_y) * src_h;

				pre_box.at<float>(i, 4 * (j - 1) + 0) = ((pred_ctr_x - 0.5*(pred_w - 1)) - 1);
				pre_box.at<float>(i, 4 * (j - 1) + 1) = ((pred_ctr_y - 0.5*(pred_h - 1)) - 1);
				pre_box.at<float>(i, 4 * (j - 1) + 2) = ((pred_ctr_x + 0.5*(pred_w - 1)) - 1);
				pre_box.at<float>(i, 4 * (j - 1) + 3) = ((pred_ctr_y + 0.5*(pred_h - 1)) - 1);
				if (pre_box.at<float>(i, 4 * (j - 1) + 0) < 0)	pre_box.at<float>(i, 4 * (j - 1) + 0) = 0;
				if (pre_box.at<float>(i, 4 * (j - 1) + 0) > (m_src.cols - 1))	pre_box.at<float>(i, 4 * (j - 1) + 0) = m_src.cols - 1;
				if (pre_box.at<float>(i, 4 * (j - 1) + 2) < 0)	pre_box.at<float>(i, 4 * (j - 1) + 2) = 0;
				if (pre_box.at<float>(i, 4 * (j - 1) + 2) > (m_src.cols - 1))	pre_box.at<float>(i, 4 * (j - 1) + 2) = m_src.cols - 1;

				if (pre_box.at<float>(i, 4 * (j - 1) + 1) < 0)	pre_box.at<float>(i, 4 * (j - 1) + 1) = 0;
				if (pre_box.at<float>(i, 4 * (j - 1) + 1) > (m_src.rows - 1))	pre_box.at<float>(i, 4 * (j - 1) + 1) = m_src.rows - 1;
				if (pre_box.at<float>(i, 4 * (j - 1) + 3) < 0)	pre_box.at<float>(i, 4 * (j - 1) + 3) = 0;
				if (pre_box.at<float>(i, 4 * (j - 1) + 3) > (m_src.rows - 1))	pre_box.at<float>(i, 4 * (j - 1) + 3) = m_src.rows - 1;
			}
			
		}
		return pre_box;
	}

	
}
Mat Faster_rcnn::proposal_local_anchor()
{
	Blob<float> shift;
	Mat shitf_x(feature_map_size.height, feature_map_size.width, CV_32SC1);
	Mat shitf_y(feature_map_size.height, feature_map_size.width, CV_32SC1);
	for (size_t i = 0; i < feature_map_size.width; i++)
	{
		for (size_t j = 0; j < feature_map_size.height; j++)
		{
			shitf_x.at<int>(j, i) = i * conf.feat_stride;
			shitf_y.at<int>(j, i) = j * conf.feat_stride;
		}
	}
	shift.Reshape(9, feature_map_size.width*feature_map_size.height,4,1);
	float *p = shift.mutable_cpu_diff(), *a = shift.mutable_cpu_data();
	for (int i = 0; i < feature_map_size.width*feature_map_size.height; i++)
	{
		for (int j = 0; j < 9; j++)
		{
			{
				size_t num = i * 4 + j * 4 * feature_map_size.width*feature_map_size.height;
				p[num + 0] = -shitf_x.at<int>(i % shitf_x.rows, i / shitf_x.rows);
				p[num + 2] = -shitf_x.at<int>(i % shitf_x.rows, i / shitf_x.rows);
				p[num + 1] = -shitf_y.at<int>(i % shitf_y.rows, i / shitf_y.rows);
				p[num + 3] = -shitf_y.at<int>(i % shitf_y.rows, i / shitf_y.rows);
				a[num + 0] = conf.anchor[j][0];
				a[num + 1] = conf.anchor[j][1];
				a[num + 2] = conf.anchor[j][2];
				a[num + 3] = conf.anchor[j][3];
			}
		}
	}
	shift.Update();
	Mat anchors(9 * feature_map_size.width*feature_map_size.height, 4, CV_32FC1);
	size_t num = 0;
	for (int i = 0; i < anchors.cols; i++)
	{
		for (int j = 0; j < anchors.rows; j++)
		{
			anchors.at<float>(j, i) = shift.data_at(num%shift.num(), 
				(num - num / (shift.num() * shift.channels())*shift.num() * shift.channels()) / shift.num(),
				num / (shift.num() * shift.channels()), 0);
			num++;
		}
	}
	/*for (int i = 0; i < 4; i++)
	{
		for (int k = 0; k < feature_map_size.width*feature_map_size.height; k++)
		{
			for (int j = 0; j < 9; j++)
			{
				anchors.at<float>(num%anchors.rows, num / anchors.rows) = shift.data_at(j, k, i, 0);
				num++;
			}
		}
	}*/
	return anchors;
}
void Faster_rcnn::prep_im_size()
{
	int im_size_min = min(m_src.cols, m_src.rows);
	int im_size_max = max(m_src.cols, m_src.rows);
	im_scale = double(conf.target_size)/im_size_min;
	if (round(im_scale*im_size_max) > conf.maxsize)
	{
		im_scale = double(conf.maxsize) / im_size_max;
	}
	input_geometry_ = Size(round(m_src.cols * im_scale), round(m_src.rows * im_scale));
	resize(m_src, im, input_geometry_);
}
bool Faster_rcnn::rpn_converttoboxs()
{
	return true;
}
Mat Faster_rcnn::getTarget(Mat src)
{
	m_src = src;
	//img=>blob
	imgToBlob(); // 输入层转换
	vector<abox> abox = forward(); // 前行传播
	testDetection(abox); // 运用非极大抑制
	return m_src;
}
void Faster_rcnn::testDetection(vector<abox>& aboxes)
{
	float scales = im_scale;
	Mat anchors(aboxes.size(),4,CV_32FC1);
	//int a= faster_rcnn_net->blob_by_name("data")->count();
	//vector<string> a = rpn_net->blob_names();
//	int b = rpn_net->blob_by_name("conv5")->count();
	faster_rcnn_net->blob_by_name("data")->CopyFrom(*rpn_net->blob_by_name("conv5").get(),false,true);
	Blob<float>* input_layer = faster_rcnn_net->input_blobs()[1];
	int sub_blob_num = aboxes.size();
	input_layer->Reshape(sub_blob_num, 5, 1, 1);
	float* input_data = input_layer->mutable_cpu_data();
	int num = 0;
	for (auto ite = aboxes.begin(); ite != aboxes.end(); ite++)
	{
		anchors.at<float>(num, 0) = ite->x1;
		anchors.at<float>(num, 1) = ite->y1;
		anchors.at<float>(num, 2) = ite->x2;
		anchors.at<float>(num, 3) = ite->y2;
		ite->score = 0;
		ite->x1 = (ite->x1 - 1) * scales + 1 - 1;
		ite->y1 = (ite->y1 - 1) * scales + 1 - 1;
		ite->x2 = (ite->x2 - 1) * scales + 1 - 1;
		ite->y2 = (ite->y2 - 1) * scales + 1 - 1;
		input_data[num * 5 + 0] = ite->score;
		input_data[num * 5 + 1] = ite->x1;
		input_data[num * 5 + 2] = ite->y1;
		input_data[num * 5 + 3] = ite->x2;
		input_data[num * 5 + 4] = ite->y2;
		num++;
	}
	const vector<Blob<float>*>& result = faster_rcnn_net->Forward();
	Blob<float>* scores = result[1];
	Blob<float>* box_deltas = result[0];
	box_deltas->data_at(0,3,0,0);
	Mat boxs_delta(box_deltas->num(), box_deltas->channels(), CV_32FC1);
	for (int i = 0; i < box_deltas->num(); i++)
	{
		for (int j = 0; j < box_deltas->channels(); j++)
		{
			boxs_delta.at<float>(i, j) = box_deltas->data_at(i, j, 0, 0);
		}
	}
	Mat score(scores->num(), scores->channels()-1, CV_32FC1);
	for (int i = 0; i < scores->num(); i++)
	{
		for (int j = 0+1; j < scores->channels(); j++)
		{
			score.at<float>(i, j-1) = scores->data_at(i, j, 0, 0);
		}
	}
	Mat pred_boxes = bbox_tranform_inv(anchors, boxs_delta,"rcnn");
	std::map<int, vector<Rect>> classer;
	////////20为类别数///////
	for (int i = 0; i < 20; i++)
	{
		vector<Rect> r;
		vector<abox> aboxes;
		for (int j = 0; j < pred_boxes.rows; j++)
		{
			abox tmp;
			tmp.x1 = pred_boxes.at<float>(j, i * 4);
			tmp.y1 = pred_boxes.at<float>(j, i * 4 + 1);
			tmp.x2 = pred_boxes.at<float>(j, i * 4 + 2);
			tmp.y2 = pred_boxes.at<float>(j, i * 4 + 3);
			tmp.score = score.at<float>(j, i);
			aboxes.push_back(tmp);
		}
		vector<int> vPick(aboxes.size());
		int nPick;
		nms(aboxes, 0.3, vPick,nPick);
		for (int i = 0; i < nPick; i++)
		{
			if (aboxes[vPick[i]].score > 0.6)
			{
				Rect rt;
				rt.x = aboxes[vPick[i]].x1;
				rt.y = aboxes[vPick[i]].y1;
				rt.width = aboxes[vPick[i]].x2 - aboxes[vPick[i]].x1 + 1;
				rt.height = aboxes[vPick[i]].y2 - aboxes[vPick[i]].y1 + 1;
				r.push_back(rt);
			}
		}
		classer.insert(std::pair<int, vector<Rect>>(i,r));
	}
	for (int i = 0; i < 20; i++)
	{
		if (!classer[i].empty())
		{
			//printf("%d\t%d\n", i, classer[i].size());
			for (auto ite = classer[i].begin(); ite != classer[i].end(); ite++)
			{
				rectangle(m_src, *ite, colortable[i], 3,8,0);
				putText(m_src, classname[i], Point(ite->x, ite->y), CV_FONT_HERSHEY_SIMPLEX,0.5, colortable[i], 2, 2);
			}
		}
	}
}