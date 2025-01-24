#include <iostream>
#include <opencv2/opencv.hpp>
#include "quickopencv.h"

int main(int argc,char** argv) {
	////图像读取为 B(blue) G(green) R(red)
	//cv::Mat src = cv::imread("D:/Project/Opencv/OpencvLearn/image/head_1.jpg");
	//if (src.empty()) {
	//	{
	//		printf("could not load image");
	//		return -1;
	//	}
	//}
	//cv::Mat resized;
	//cv::resize(src, resized, cv::Size(src.cols / 10, src.rows / 10));
	//// 创建可调整大小的窗口
	//cv::namedWindow("输入图像", cv::WINDOW_AUTOSIZE);
	//cv::imshow("输入图像", resized);
	//

	//QuickDemo qd;
	//qd.bifilter_demo(resized);

	//cv::waitKey(0);
	//cv::destroyAllWindows();


	//人脸识别
	std::string pb_file_path = "D:/Project/Opencv/OpencvLearn/opencv480_quick_tutorial/src/opencv_face_detector_uint8.pb";
	std::string pbtxt_file_path = "D:/Project/Opencv/OpencvLearn/opencv480_quick_tutorial/src/opencv_face_detector.pbtxt";
	cv::dnn::Net net = cv::dnn::readNetFromTensorflow(pb_file_path, pbtxt_file_path);
	VideoCapture cap(0);
	cv::Mat frame;
	while (true) {
		cap.read(frame);
		if (frame.empty()) {
			break;
		}
		cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, Size(300, 300), cv::Scalar(104, 177, 123), false, false);
		net.setInput(blob);
		cv::Mat probs = net.forward();
		
		cv::Mat detectMat(probs.size[2], probs.size[3], CV_32F, probs.ptr<float>());
		for (int row = 0; row < detectMat.rows; row++) {
			float conf = detectMat.at<float>(row, 2);
			if (conf > 0.5) {
				float x1 = detectMat.at<float>(row, 3)*frame.cols;
				float y1 = detectMat.at<float>(row, 4)*frame.rows;
				float x2 = detectMat.at<float>(row, 5)*frame.cols;
				float y2 = detectMat.at<float>(row, 6) * frame.rows;
				cv::Rect box(x1, y1, x2 - x1, y2 - y1);
				cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 2, 8);
			}
		}
		cv::imshow("人脸检测", frame);
		char c = waitKey(1);
		if (c == 27) {
			break;
		}

	}
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}