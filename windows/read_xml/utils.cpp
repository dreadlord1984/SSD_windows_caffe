#include "xml.h"

using namespace std;
using namespace cv;
using namespace caffe;
using namespace boost::property_tree;

void ReadandResizeImage(string filename, int& img_width, int& img_height, Mat& cv_img_resize)
{
	cv::Mat cv_img_origin = cv::imread(filename, 1);
	img_height = cv_img_origin.rows;
	img_width = cv_img_origin.cols;
	cv::resize(cv_img_origin, cv_img_resize, cv::Size(resize_width, resize_height));
}

void ReadXMLandResizebox(std::string labelname, std::map<std::string, int>& name_to_label, const int& img_width, const int& img_height, std::vector<caffe::NormalizedBBox> &Boxes)
{
	ptree pt;
	read_xml(labelname, pt);
	// Parse annotation.
	int width = 0, height = 0;
	try {
		height = pt.get<int>("annotation.size.height");
		width = pt.get<int>("annotation.size.width");
	}
	catch (const ptree_error &e) {
		LOG(WARNING) << "When parsing " << labelname << ": " << e.what();
		height = img_height;
		width = img_width;
	}

	LOG_IF(WARNING, height != img_height) << labelname <<
		" inconsistent image height.";
	LOG_IF(WARNING, width != img_width) << labelname <<
		" inconsistent image width.";
	CHECK(width != 0 && height != 0) << labelname <<
		" no valid image width/height.";
	int instance_id = 0;
	int num = 0;
	BOOST_FOREACH(ptree::value_type &v1, pt.get_child("annotation")) {
		ptree pt1 = v1.second;
		if (v1.first == "object") {
			bool difficult = false;
			ptree object = v1.second;
			BOOST_FOREACH(ptree::value_type &v2, object.get_child("")) {
				ptree pt2 = v2.second;
				if (v2.first == "name") {
					string name = pt2.data();
					if (name_to_label.find(name) == name_to_label.end()) {
						LOG(FATAL) << "Unknown name: " << name;
					}
					int label = name_to_label.find(name)->second;
				}
				else if (v2.first == "difficult") {
					difficult = pt2.data() == "1";
				}
				else if (v2.first == "bndbox") {

					// Get bounding box.
					float xmin = pt2.get<float>("xmin");
					float ymin = pt2.get<float>("ymin");
					float xmax = pt2.get<float>("xmax");
					float ymax = pt2.get<float>("ymax");

					// Store the normalized bounding box.
					NormalizedBBox bbox;
					bbox.set_xmin(static_cast<float>(xmin) / width);
					bbox.set_ymin(static_cast<float>(ymin) / height);
					bbox.set_xmax(static_cast<float>(xmax) / width);
					bbox.set_ymax(static_cast<float>(ymax) / height);
					bbox.set_difficult(difficult);
				
					/*char temp[20];
					sprintf(temp, "%d", num);
					cv::putText(cv_img_origin, temp, cv::Point(bbox.xmin() + 15, bbox.ymin() + 15), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2, 2);*/
					Boxes.push_back(bbox);
				}
			}
			num += 1;
		}
	}
}