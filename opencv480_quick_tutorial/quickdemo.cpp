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


