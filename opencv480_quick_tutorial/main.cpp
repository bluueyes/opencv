#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc,char** argv) {

	cv::Mat src = cv::imread("D:/Project/Opencv/OpencvLearn/image/head_1.jpg"
								,cv::IMREAD_GRAYSCALE);
	if (src.empty()) {
		{
			printf("could not load image");
			return -1;
		}
	}
	std::cout << src.depth() << std::endl;
	cv::Mat resized;
	cv::resize(src, resized, cv::Size(src.cols / 2, src.rows / 2));
	// �����ɵ�����С�Ĵ���
	cv::namedWindow("����ͼ��", cv::WINDOW_NORMAL);
	cv::imshow("����ͼ��", resized);
	cv::waitKey(0);

	cv::destroyAllWindows();
	return 0;
}