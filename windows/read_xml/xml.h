#ifndef XML_H
#define XML_H

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "boost/variant.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include <opencv2/opencv.hpp>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

extern int resize_width;
extern int resize_height;

void ReadandResizeImage(std::string filename, int& ori_width, int& ori_height, cv::Mat& cv_img_resize);


void ReadXMLandResizebox(std::string labelname, std::map<std::string, int>& name_to_label, const int& img_width, const int& img_height, std::vector<caffe::NormalizedBBox> &Boxes);
#endif
