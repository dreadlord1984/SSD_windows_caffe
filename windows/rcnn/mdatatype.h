#include <vector>
#include <map>
struct config{
	int maxsize;
	int target_size;
	int feat_stride;
	int anchor[9][4];
	int test_min_box_size;
	int per_nms_topN;
	int after_nms_topN;
	float overlap;
	config(){
		maxsize = 1000;
		target_size = 600;
		feat_stride = 16;
		int tmp[9][4] = {
				{ -83, -39, 100, 56 },
				{ -175, -87, 192, 104 },
				{ -359, -183, 376, 200 },
				{ -55, -55, 72, 72 },
				{ -119, -119, 136, 136 },
				{ -247, -247, 264, 264 },
				{ -35, -79, 52, 96 },
				{ -79, -167, 96, 184 },
				{ -167, -343, 184, 360 }
		};
		memcpy(anchor, tmp,9*4*sizeof(int));
		test_min_box_size = 16;
		per_nms_topN = 6000;
		after_nms_topN = 300;
		overlap = 0.7;
	}
};
struct abox
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
};
