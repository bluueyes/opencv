#include "quickopencv.h"

void QuickDemo::colorSpace_Demo(Mat& image) {

	Mat gray, hsv;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	cvtColor(image, gray, COLOR_BGR2GRAY);

	// 创建可调整大小的窗口
	cv::namedWindow("HSV", cv::WINDOW_NORMAL);
	//H 0~180 , S(0~255) , V	(H,S是对颜色调整) V(是亮度)
	cv::namedWindow("灰度", cv::WINDOW_NORMAL);

	imshow("HSV", hsv);
	imshow("灰度", gray);
	imwrite("D:/hsv.png", hsv);
	imwrite("D:/gray.png", gray);

}