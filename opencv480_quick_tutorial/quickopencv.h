#pragma once
#include <opencv2\opencv.hpp>

using namespace cv;

class QuickDemo {
public:
	void colorSpace_Demo(const Mat& image);
	void mat_creation_demo(const Mat& image);
	void pixel_visit_demo(Mat& image);
	void operator_demo(Mat& image);
	void check_bar_demo(Mat& image);


};