#include "caffe/FRCNN/util/frcnn_utils.hpp"

namespace caffe {

namespace Frcnn {

INSTANTIATE_CLASS(Point4f);
INSTANTIATE_CLASS(BBox);

template <typename Dtype>
Dtype get_iou(const Point4f<Dtype> &A, const Point4f<Dtype> &B) {
  const Dtype xx1 = std::max(A[0], B[0]);
  const Dtype yy1 = std::max(A[1], B[1]);
  const Dtype xx2 = std::min(A[2], B[2]);
  const Dtype yy2 = std::min(A[3], B[3]);
  Dtype inter = std::max(Dtype(0), xx2 - xx1 + 1) * std::max(Dtype(0), yy2 - yy1 + 1);
  Dtype areaA = (A[2] - A[0] + 1) * (A[3] - A[1] + 1);
  Dtype areaB = (B[2] - B[0] + 1) * (B[3] - B[1] + 1);
  return inter / (areaA + areaB - inter);
}
template float get_iou(const Point4f<float> &A, const Point4f<float> &B);
template double get_iou(const Point4f<double> &A, const Point4f<double> &B);

template <typename Dtype>
vector<vector<Dtype> > get_ious(const vector<Point4f<Dtype> > &A, const vector<Point4f<Dtype> > &B) {
  vector<vector<Dtype> >ious;
  for (size_t i = 0; i < A.size(); i++) {
    ious.push_back(get_ious(A[i], B));
  }
  return ious;
}
template vector<vector<float> > get_ious(const vector<Point4f<float> > &A, const vector<Point4f<float> > &B);
template vector<vector<double> > get_ious(const vector<Point4f<double> > &A, const vector<Point4f<double> > &B);

template <typename Dtype>
vector<Dtype> get_ious(const Point4f<Dtype> &A, const vector<Point4f<Dtype> > &B) {
  vector<Dtype> ious;
  for (size_t i = 0; i < B.size(); i++) {
    ious.push_back(get_iou(A, B[i]));
  }
  return ious;
}

template vector<float> get_ious(const Point4f<float> &A, const vector<Point4f<float> > &B);
template vector<double> get_ious(const Point4f<double> &A, const vector<Point4f<double> > &B);

float get_scale_factor(int width, int height, int short_size, int max_long_size) {
  float im_size_min = std::min(width, height);
  float im_size_max = std::max(width, height);

  float scale_factor = static_cast<float>(short_size) / im_size_min;
  // Prevent the biggest axis from being more than max_size
  if (scale_factor * im_size_max > max_long_size) {
    scale_factor = static_cast<float>(max_long_size) / im_size_max;
  }
  return scale_factor;
}

/***********************************************************************
* function: ªÒ»° anchors
***********************************************************************/
template <typename Dtype>
void GetAnchors(const Dtype* prior_data, const int num_priors,
	vector < Point4f<Dtype> >* anchors,
	vector<vector<float> >* prior_variances,
	const float& im_height, const float& im_width) {
	anchors->clear();
	prior_variances->clear();
	for (int i = 0; i < num_priors; ++i) {
		int start_idx = i * 4;
		Point4f<Dtype> anchor(prior_data[start_idx],
			prior_data[start_idx + 1],
			prior_data[start_idx + 2],
			prior_data[start_idx + 3]);
		anchors->push_back(anchor);
	}

	for (int i = 0; i < num_priors; ++i) {
		int start_idx = (num_priors + i) * 4;
		vector<float> var;
		for (int j = 0; j < 4; ++j) {
			var.push_back(prior_data[start_idx + j]);
		}
		prior_variances->push_back(var);
	}
}

// Explicit initialization.
template void GetAnchors(const float* prior_data, const int num_priors,
	vector<Point4f<float>>* anchors,
	vector<vector<float> >* prior_variances,
	const float& im_height, const float& im_width);
template void GetAnchors(const double* prior_data, const int num_priors,
	vector<Point4f<double>>* anchors,
	vector<vector<float> >* prior_variances,
	const float& im_height, const float& im_width);

} // namespace frcnn

} // namespace caffe
