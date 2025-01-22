#include "quickopencv.h"
#include <functional>

void QuickDemo::colorSpace_Demo(const Mat& image) {

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

void QuickDemo::mat_creation_demo(const Mat& image) {
	Mat m2;
	Mat m1 = image.clone();		//���
	image.copyTo(m2);	//��� ������Ҫһ��Ŀ�������Ϊ����,���Զ����ݶ�Ŀ�����������е���
	if (image.data == m2.data) {
		std::cout << "m2==Image"<<std::endl;
	}
	else {
		std::cout << " m2 �� data �ĸ���" << std::endl;
	}
	if (image.data == m1.data) {
		std::cout << "m1==Image" << std::endl;
	}
	else {
		std::cout << " m1 �� data �ĸ���" << std::endl;
	}

	//�����հ�ͼ��
	Mat m3 = Mat::zeros(Size(8, 8), CV_8UC3);
	m3 = Scalar(255, 0, 0);
	std::cout << "width: "<< m3.cols <<" height: "<<m3.rows<<" channels: "<<m3.channels() << std::endl;

	std::cout << m3 << std::endl;

	Mat m4 = m3;	//Mat�Ŀ���������ǳ����
	m4 = Scalar(0, 0, 255);

	namedWindow("ͼ��", WINDOW_NORMAL);
	imshow("ͼ��",m3);
}


void QuickDemo::pixel_visit_demo(Mat& image) {
	int r = image.rows;
	int c = image.cols;
	int dims = image.channels();

	//for (int row = 0; row < r; row++) {
	//	for (int clo = 0; clo < c; clo++) {
	//		if (dims == 1) {
	//			int pv = image.at<uchar>(row, clo);
	//			image.at<uchar>(row, clo) = 255 - pv;
	//		}
	//		if (dims == 3) {
	//			Vec3b bgr = image.at<Vec3b>(row, clo);
	//			image.at<Vec3b>(row, clo)[0] = 255 - bgr[0];
	//			image.at<Vec3b>(row, clo)[1] = 255 - bgr[1];
	//			image.at<Vec3b>(row, clo)[2] = 255 - bgr[2];
	//		}
	//	}
	//}

	for (int row = 0; row < r; row++) {
		uchar* current_row = image.ptr<uchar>(row);
		for (int clo = 0; clo < c; clo++) {
			if (dims == 1) {
				*current_row++ = 255 - *current_row;
			}
			if (dims == 3) {
			 ;
				*current_row++ = 255 - *current_row;
				*current_row++ = 255 - *current_row;
				*current_row++ = 255 - *current_row;
			}
		}
	}


	cv::namedWindow("���ض�д��ʾ", cv::WINDOW_NORMAL);
	imshow("���ض�д��ʾ",image);
}

void QuickDemo::operator_demo(Mat& image)
{
	Mat dst;
	dst = image + Scalar(50, 50, 50);
	cv::namedWindow("�ӷ�����", WINDOW_NORMAL);
	imshow("�ӷ�����", dst);

	multiply(image, Scalar(2, 2, 2), image);
	cv::namedWindow("�˷�����", WINDOW_NORMAL);
	imshow("�˷�����", image);

	//Mat m = Mat::ones(image.size(), image.type());
	//m = Scalar(50, 50, 50);
	//int r = image.rows;
	//int c = image.cols;
	//int dims = image.channels();
	//for (int row = 0; row < r; row++) {
	//	for (int clo = 0; clo < c; clo++) {
	//		Vec3b bgr = image.at<Vec3b>(row, clo);
	//		Vec3b bgr2 = m.at<Vec3b>(row, clo);
	//		image.at<Vec3b>(row, clo)[0] = saturate_cast<uchar>(bgr[0] + bgr2[0]);	//saturate_cast<uchar> ���޶�0~255֮��
	//		image.at<Vec3b>(row, clo)[1] = saturate_cast<uchar>(bgr[1] + bgr2[1]);
	//		image.at<Vec3b>(row, clo)[2] = saturate_cast<uchar>(bgr[2] + bgr2[2]);
	//	}
	//}
	//cv::namedWindow("�ӷ�����ʵ��", WINDOW_NORMAL);
	//imshow("�ӷ�����ʵ��", dst);

	//����
	divide(image, Scalar(10, 10, 10),image);
	cv::namedWindow("��������", WINDOW_NORMAL);
	imshow("��������", image);

}

struct Data {
	Mat src, dst, m;
	double alpha, beta;
	Data() :alpha(0),beta(0) {}
};



static void on_track(int pos, void* d)
{
	
	Data* data = static_cast<Data*>(d);
	data->m = Scalar(pos,pos,pos);
	addWeighted(data->src, data->alpha, data->m, data->beta, pos, data->dst);

	imshow("������Աȶȵ���", data->dst);
}

static void on_constrast(int pos, void* d)
{

	Data* data = static_cast<Data*>(d);
	data->m = Scalar(pos, pos, pos);

	double contrast = pos/ 100.0;
	addWeighted(data->src, contrast, data->m, data->beta, 0, data->dst);

	imshow("������Աȶȵ���", data->dst);
}

static int max_value = 100;

void QuickDemo::check_bar_demo(Mat& image)
{
	namedWindow("������Աȶȵ���",WINDOW_NORMAL);
	
	Data *data=new Data();
	data->dst = Mat::zeros(image.size(), image.type());
	data->m = Mat::zeros(image.size(), image.type());
	data->src = image;
	data->alpha = 1.0;
	data->beta = 0;

	int beginValue = 50;
	int contrastValue = 100;
    createTrackbar("Value Bar", "������Աȶȵ���", &beginValue, max_value,on_track,(void*)data);

	createTrackbar("Contrase Bar", "������Աȶȵ���", &contrastValue, max_value,on_constrast, (void*)data);



}

void QuickDemo::key_demo(Mat& image)
{
	Mat dst=Mat::zeros(image.size(),image.type());
	Mat temp=Mat::zeros(image.size(), image.type());;
	temp = Scalar(50, 50, 50);
	Mat original = image.clone(); // ����ԭʼͼ��

	while (true) {
		char c = waitKey(10);
		if (c == 27) break;

		if (c == 49) {
			std::cout << "you enter key #1" << std::endl;
			cvtColor(image, dst, COLOR_BGR2GRAY);
		}

		if (c == 50) {
			std::cout << "you enter key #2" << std::endl;
			cvtColor(image, dst, COLOR_BGR2HSV);
		}

		if (c == 51) {
			std::cout << "you enter key #3" << std::endl;;
			add(image, temp, dst);
			image += temp;
		}

		if (c == 52) {
			std::cout << "you enter key #4" << std::endl;;
			subtract(image, temp, dst);
			image -= temp;
		}

		imshow("������Ӧ", dst);
	}

}

void QuickDemo::color_style_demo(Mat& image)
{
	Mat grayimage;
	cvtColor(image, grayimage, COLOR_BGR2GRAY);;
	
	for (int i = 0; i <= 21; i++) {
		Mat colorImage;
		applyColorMap(grayimage, colorImage, i);

		std::string windowName = "Color Map ";
		namedWindow(windowName, WINDOW_NORMAL);
		imshow(windowName, colorImage);
		waitKey(500);
	}

}

void QuickDemo::bitwise_demo(Mat& image)
{
	Mat m1 = Mat::zeros(Size(256, 256), CV_8UC3);
	Mat m2 = Mat::zeros(Size(256, 256), CV_8UC3);

	rectangle(m1, Rect(100, 100, 80, 80), Scalar(255, 255, 0), -1, LINE_8, 0);
	rectangle(m2, Rect(150, 150, 80, 80), Scalar(0, 255, 255), -1, LINE_8, 0);
	imshow("m1", m1);
	imshow("m2", m2);

	Mat dst;
	bitwise_xor(m1, m2, dst);
	imshow("����λ����", dst);
}

void QuickDemo::channels_demo(Mat& image)
{
	std::vector<Mat> mv;
	split(image, mv);

	imshow("��ɫ", mv[0]);
	imshow("��ɫ", mv[1]);
	imshow("��ɫ", mv[2]);

	Mat dst;
	mv[0] = 0;
	mv[1] = 0;

	merge(mv,dst);
	imshow("��ɫ", dst);

	int from_to[] = { 0,2,1,1,2,0 }; //0->2 1->1 2->0
	mixChannels(&image,1 ,&dst, 1,from_to,3);
	imshow("ͨ�����", dst);
}

void QuickDemo::inrange_demo(Mat& image)
{
	//gbrͨ��������ɫ�ֲ�̫�������ȡ׼ȷ�ķ�Χ����hsvֻ��һ����ɫͨ����������ȡ
	Mat hsv;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	Mat mask;
	inRange(hsv, Scalar(0, 0, 200), Scalar(180, 50, 255), mask);
	imshow("mask",mask);

	Mat redback=Mat::zeros(image.size(),image.type());
	redback = Scalar(40, 40, 200);
	bitwise_not(mask, mask);
	imshow("mask", mask);

	image.copyTo(redback, mask);
	imshow("roi������ȡ", redback);


}


