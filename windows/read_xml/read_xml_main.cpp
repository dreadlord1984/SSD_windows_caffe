#include "xml.h"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;
using namespace boost::property_tree;


int resize_width = 384;
int resize_height = 256;

int main(int argc, char** argv)
{
	std::string root_folder = "\\\\192.168.1.186/PedestrianData/Data_0728/";
	string label_map_file = "labelmap_VehicleFull.prototxt";
	std::map<std::string, int> name_to_label;
	const bool check_label = true;
	int label;
	
	string list_name = argv[1];
	string list_file = "\\\\192.168.1.186/PedestrianData/Data_0728/testxml/" + list_name;
	std::ifstream infile(list_file);
	std::string filename;
	std::string labelname;
	std::vector<std::pair<std::string, boost::variant<int, std::string> > > lines;

	label_map_file = root_folder + label_map_file;
	LabelMap label_map;
	CHECK(ReadProtoFromTextFile(label_map_file, &label_map))
		<< "Failed to read label map file.";
	CHECK(MapNameToLabel(label_map, check_label, &name_to_label))
		<< "Failed to convert name to label.";

	std::string djStrLine;
	while (infile)
	{
		if (getline(infile, djStrLine))
		{
			int dwPos = djStrLine.rfind(".jpg ");
			filename = djStrLine.substr(0, dwPos + 4);
			int dwEnd = djStrLine.rfind(".xml");
			labelname = djStrLine.substr(dwPos + 5, (dwEnd + 4 - (dwPos + 5)));
			lines.push_back(std::make_pair(filename, labelname));
		}
	}


	LOG(INFO) << "A total of " << lines.size() << " images.";


	cv::namedWindow("result",1);

	// 对于每个样本，获取 bounding box.
   	for (int line_id = 0; line_id < lines.size(); ++line_id) {
		bool status = true;
		filename = root_folder + lines[line_id].first;
		std::cout << filename << std::endl;
		labelname = root_folder + boost::get<std::string>(lines[line_id].second);

		int ori_width, ori_height;
		cv::Mat cv_img_resize;
		ReadandResizeImage(filename, ori_width, ori_height, cv_img_resize);

		vector<NormalizedBBox> Boxes;
		ReadXMLandResizebox(labelname, name_to_label, ori_width, ori_height, Boxes);


		for (int i = 0; i < Boxes.size(); i++){
			cv::Rect pos(Boxes[i].xmin() * resize_width, Boxes[i].ymin() * resize_height,
				Boxes[i].xmax() * resize_width - Boxes[i].xmin()* resize_width,
				Boxes[i].ymax() * resize_height - Boxes[i].ymin() * resize_height);
			cv::rectangle(cv_img_resize, pos, cv::Scalar(0, 0, 255), 2, 8, 0);
		}
		
		cv::imshow("result", cv_img_resize);
		cv::waitKey(0);
	}
	return 0;
}