#pragma once

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"
#include "mdatatype.h"
#include <opencv2/opencv.hpp>
//using namespace std;
using namespace cv;
using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;
class Faster_rcnn
{
public:
	Faster_rcnn();
	~Faster_rcnn();
	bool init();
	Mat getTarget(Mat);
public:
	config conf;
private:
	Mat im, m_src;
	Size input_geometry_;
	shared_ptr<Net<float> > rpn_net, faster_rcnn_net;
	double im_scale;
	Size feature_map_size;

private:
	bool loadnet();
	bool imgToBlob();
	vector<abox> forward();
	bool rpn_converttoboxs();
	void prep_im_size();
	Mat proposal_local_anchor();
	Mat bbox_tranform_inv(Mat, Mat,string);
	Mat get_rpn_score(Blob<float>*, int w, int h);
	void m_sort(Mat&, Mat&);
	//bool aboxcomp(abox&, abox&);
	void boxes_filter(vector<abox>&, int, vector<abox>, vector<int>);
	void filter_boxs(Mat&, Mat&, vector<abox>&);
	void nms(vector<abox>, double overlap, vector<int> &vPick, int &nPick);
	void testDetection(vector<abox>&);
		//float*  boxs;
};

