#include <iostream>
#include <opencv2/opencv.hpp>
#include "quickopencv.h"

int main(int argc,char** argv) {
	//图像读取为 B(blue) G(green) R(red)
	cv::Mat src = cv::imread("D:/Project/Opencv/OpencvLearn/image/head_1.jpg");
	if (src.empty()) {
		{
			printf("could not load image");
			return -1;
		}
	}
	cv::Mat resized;
	cv::resize(src, resized, cv::Size(src.cols / 2, src.rows / 2));
	// 创建可调整大小的窗口
	cv::namedWindow("输入图像", cv::WINDOW_NORMAL);
	cv::imshow("输入图像", resized);
	

	QuickDemo qd;
	qd.colorSpace_Demo(resized);

	cv::waitKey(0);

	cv::destroyAllWindows();
	return 0;
}