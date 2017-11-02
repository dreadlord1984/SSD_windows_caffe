// fasterrcnn_vs2013.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
#include "Faster_rcnn.h"
#include "head.h"
int main(int argc, char** argv[])
{
 	//interface
	IplImage* pFrame = NULL;

	//获取摄像头
	CvCapture* pCapture = cvCreateCameraCapture(0);
	pFrame = cvQueryFrame(pCapture);
	Mat im = imread("2030.jpg"); // E:/caffe-master/windows/rcnn/10001.jpg
//	Mat im(pFrame);
	Faster_rcnn detect;
	detect.init();
	//while (1)
	{
		//Mat im = imread("C:\\000456.jpg");
		//pFrame = cvQueryFrame(pCapture);
		//Mat im(pFrame);
		Mat img = detect.getTarget(im);
		namedWindow("", 0);
		imshow("", img);
		cvWaitKey(0);
		
		
	}

	return 0;
}

