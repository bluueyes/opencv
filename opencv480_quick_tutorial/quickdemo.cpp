#include "quickopencv.h"

void QuickDemo::colorSpace_Demo(Mat& image) {

	Mat gray, hsv;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	cvtColor(image, gray, COLOR_BGR2GRAY);

	// �����ɵ�����С�Ĵ���
	cv::namedWindow("HSV", cv::WINDOW_NORMAL);
	//H 0~180 , S(0~255) , V	(H,S�Ƕ���ɫ����) V(������)
	cv::namedWindow("�Ҷ�", cv::WINDOW_NORMAL);

	imshow("HSV", hsv);
	imshow("�Ҷ�", gray);
	imwrite("D:/hsv.png", hsv);
	imwrite("D:/gray.png", gray);

}