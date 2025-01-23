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
	//Scalar��һ�����ڱ�ʾ��ͨ������ֵ���ࡣ
	// ��ͨ�����ڱ�ʾ��ɫ��������ͨ������
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
	//addWeighted���������ڽ�����ͼ��ָ��Ȩ�ؽ��м�Ȩ�͡�
	// ��ͨ������ͼ���ϡ�ͼ����ӵȲ�����
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
	//cvtColor������OpenCV�е�һ�������������ڲ�ͬ��ɫ�ռ�֮��ת��ͼ��
	// �����Խ�ͼ���һ����ɫ�ռ�ת��Ϊ��һ����ɫ�ռ䣬
	// �����BGRת��Ϊ�Ҷ�ͼ�񡢴�BGRת��ΪHSVͼ���
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
	//bitwise_xor ���ڰ�λ��� 
	bitwise_xor(m1, m2, dst);
	imshow("����λ����", dst);
}

void QuickDemo::channels_demo(Mat& image)
{
	std::vector<Mat> mv;
	split(image, mv);	//split�������ڽ���ͨ��ͼ�����ɵ�ͨ��ͼ��

	imshow("��ɫ", mv[0]);
	imshow("��ɫ", mv[1]);
	imshow("��ɫ", mv[2]);

	Mat dst;
	mv[0] = 0;
	mv[1] = 0;

	//merge�������ڽ������ͨ��ͼ��ϲ���һ����ͨ��ͼ��
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
	//inRange������OpenCV�е�һ�����������ڼ������Ԫ���Ƿ���ָ����Χ�ڡ�
	// ��ͨ�����ڴ�����ֵ����ͼ�������������������ر�����Ϊ��ɫ��255����
	// ���������������ر�����Ϊ��ɫ��0���������������ɫ�ָ�Ͷ������зǳ����á�
	inRange(hsv, Scalar(0, 0, 200), Scalar(180, 50, 255), mask);
	imshow("mask",mask);

	Mat redback=Mat::zeros(image.size(),image.type());
	redback = Scalar(40, 40, 200);
	bitwise_not(mask, mask);
	imshow("mask", mask);

	image.copyTo(redback, mask);
	imshow("roi������ȡ", redback);


}

void QuickDemo::pixel_static_demo(Mat& image)
{
	double minv, maxv;
	Point minLoc, maxLoc;
	std::vector<Mat> vec;
	split(image, vec);
	int i = 0;
	//minMaxLoc���������ڲ��������ͼ����Ԫ�ص���Сֵ�����ֵ����λ��
	for (auto& a : vec) {
		i++;
		minMaxLoc(a, &minv, &maxv, &minLoc, &maxLoc, Mat());
		std::cout << "Channel " << i << ": minvalue" << minv << " maxvalue" << maxv<<std::endl;
	}
	Mat mean, stddev;
	//meanStdDev���������ڼ��������ͼ��ľ�ֵ�ͱ�׼�
	// ���������ڵ�ͨ�����ͨ��ͼ�񣬲�����ÿ��ͨ���ľ�ֵ�ͱ�׼��
	meanStdDev(image, mean, stddev);
	for (int i = 0; i < 3; i++) {
		std::cout << "means: "<<i+1 <<" = " << mean.at<double>(i,0) << std::endl;
	}

	for (int i = 0; i < 3; i++) {
		std::cout << "stddev: " << i + 1 << " = " << stddev.at<double>(i,0) << std::endl;
	}
}

void QuickDemo::draw_demo(Mat& image)
{
	Rect rect;
	rect.x = 100;
	rect.y = 100;
	rect.width = 50;
	rect.height = 50;
	Mat bg = Mat::zeros(image.size(), image.type());
	rectangle(bg, rect, Scalar(0, 0, 255), -1, 8, 0);
	circle(bg, Point(150, 150), 10, Scalar(0, 255, 0), -1, 8, 0);
	Mat dest;
	addWeighted(image, 0.7, bg, 0.3, 0, dest);
	imshow("������ʾ", dest);
}

void QuickDemo::random_drawing_demo(Mat& image)
{
	Mat canvas = Mat::zeros(Size(512, 512), CV_8UC3);
	int w = canvas.cols;
	int h = canvas.rows;
	RNG rng(12345); //���������
	while (true) {
		char c = waitKey(10);
		if (c == 27) break;

		int x1 = rng.uniform(0, w);
		int y1 = rng.uniform(0, h);
		int x2 = rng.uniform(0, w);
		int y2 = rng.uniform(0, h);
		int colorb = rng.uniform(0, 255);
		int colorg = rng.uniform(0, 255);
		int colorr = rng.uniform(0, 255);
		canvas = Scalar(0, 0, 0);
		line(canvas, Point(x1, y1), Point(x2, y2), Scalar(colorb,colorg,colorr), 1, LINE_AA, 0);

		imshow("չʾ", canvas);

	}

}

void QuickDemo::polyline_draw_demo(Mat& image)
{
	Mat canvas = Mat::zeros(Size(512, 512), CV_8UC3);
	Point p1(100, 100);
	Point p2(350, 100);
	Point p3(450, 280);
	Point p4(200, 130);
	Point p5(180, 310);

	std::vector<Point> pts = { p1,p2,p3,p4,p5 };

	/*polylines(canvas, pts, true, Scalar(0, 0, 255), 1, 8, 0);
	fillPoly(canvas, pts, Scalar(255, 255, 0),8,0);*/
	std::vector<std::vector<Point>> contours = { pts };

	drawContours(canvas, contours, -1, Scalar(255, 0, 0), -1);
	imshow("����λ���",canvas);


}

Point posbegin(-1,-1);
Point posend(-1,-1);
static Mat temp;
static void on_draw(int event, int x, int y, int flags, void* userdata) {

	Mat* image = static_cast<Mat*>(userdata);

	if (event == EVENT_LBUTTONDOWN) {
		posbegin.x = x;
		posbegin.y = y;
		std::cout << "pox x:" << posbegin.x << " y:" << posbegin.y << std::endl;
	}
	else if (event == EVENT_LBUTTONUP) {
		posend.x = x;
		posend.y = y;
		if (x > 0 && y > 0) {
			int maxs = std::max(std::abs(posbegin.x - posend.x), std::abs(posbegin.y - posend.y));
			
			// ��ͼ���ϻ���Բ
			Point center((posbegin.x + posend.x) / 2, (posbegin.y + posend.y) / 2);
			int radius = maxs / 2;

		
			circle(*image, center, radius, Scalar(0, 255, 0), 1, 8, 0);
			imshow("������", *image);
			// ��ʾROI����
			if (posend.x > image->cols || posend.y > image->rows) return;
			Rect roi(center.x - radius, center.y - radius, radius * 2, radius * 2);
			Mat roiImage = temp(roi);
			imshow("ROI����", roiImage);
		}
		posend.x = -1;
		posend.y = -1;
	}
	else if (event == EVENT_MOUSEMOVE && flags== EVENT_FLAG_LBUTTON) {
		posend.x = x;
		posend.y = y;
		temp.copyTo(*image);
		if (x > 0 && y > 0) {
			int maxs = std::max(std::abs(posbegin.x - posend.x),std::abs(posbegin.y-posend.y));
			circle(*image, Point((posbegin.x + posend.x) / 2, (posbegin.y + posend.y) / 2),maxs/2 , Scalar(0, 255, 0), 1, 8, 0);
			imshow("������", *image);
		}
	}
	
}

void QuickDemo::mouse_drawing_demo(Mat& image)
{
	namedWindow("������", WINDOW_AUTOSIZE);
	setMouseCallback("������", on_draw,&image);
	temp = image.clone();
	imshow("������",image);
}

void QuickDemo::norm_demo(Mat& image)
{
	Mat dst;
	std::cout << image.type() << std::endl;

	image.convertTo(image, CV_32FC3);
	std::cout << image.type() << std::endl;

	normalize(image, dst, 1.0, 0, NORM_MINMAX);
	std::cout << dst.type() << std::endl;
	imshow("ͼ�����ݹ�һ��", dst);
}

void QuickDemo::resize_demo(Mat& image)
{
	Mat zoomin, zoomout;
	int h = image.rows;
	int w = image.cols;
	resize(image, zoomin, Size(w / 2, h / 2), 0, 0 , INTER_LINEAR);
	imshow("zoomin", zoomin);
	resize(image, zoomout, Size(w * 1.5, h * 1.5), 0, 0, INTER_LINEAR);
	imshow("zoomout", zoomout);
}

void QuickDemo::flip_demo(Mat& image)
{
	Mat dst;
	//flip(image, dst, 0);	//���·�ת
	//flip(image, dst, 1);	//���ҷ�ת
	flip(image, dst, -1);	//180����ת
	imshow("ͼ��ת", dst);
}

void QuickDemo::rotate_demo(Mat& image)
{
	Mat dst, M;
	int w = image.cols;
	int h = image.rows;
	//getRotationMatrix2D������OpenCV�е�һ��������
	// ��������һ������ͼ����ת��2x3����任����
	// �þ������������תͼ��ͬʱ����ָ����ת���ġ���ת�ǶȺ����ű�����
	M = getRotationMatrix2D(Point2f(w / 2, h / 2), 45, 1.0);
	double cos = abs(M.at<double>(0, 0));
	double sin = abs(M.at<double>(0, 1));

	int nw = cos * w + sin * h;
	int nh = cos * h + sin * w;

	M.at<double>(0, 2) += (nw / 2 - w / 2);
	M.at<double>(1, 2) += (nh / 2 - h / 2);
	//warpAffine������OpenCV�е�һ�����������ڶ�ͼ����з���任��
	// ����任��һ�����Ա任��������ת�����š�ƽ�ƺͼ��еȲ�����
	// warpAffine�������Խ�����ͼ��Ӧ��ָ���ķ���任���󣬲�����任���ͼ��
	warpAffine(image, dst, M, Size(nw, nh), INTER_LINEAR, 0, Scalar(255, 255, 0));
	imshow("��ת��ʾ", dst);
}

void QuickDemo::video_demo(Mat& iamge)
{
	VideoCapture capture("D:/cc.mp4");
	int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);
	int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);
	int count = capture.get(CAP_PROP_FRAME_COUNT);
	double fps = capture.get(CAP_PROP_FPS);
	std::cout << "frame width:" << frame_width << std::endl;
	std::cout << "frame height:" << frame_height << std::endl;
	std::cout << "FPS:" << fps << std::endl;
	std::cout << "Number of frame:" << count << std::endl;
	VideoWriter write("D:/test.mp4", capture.get(CAP_PROP_FOURCC),fps,Size(frame_width,frame_height),true);
	Mat frame;
	while (true) {
		capture.read(frame);
		flip(frame,frame, 1);
		if (frame.empty()) {
			break;
		}
		imshow("frame", frame);
		write.write(frame);

		int c = waitKey(25);
		if (c == 27) {	//�˳�
			break;
		}
	}
	capture.release();
}

void QuickDemo::showHistogranm_demo(Mat& image)
{
	//����ͨ������
	std::vector<Mat> bgr_plane;
	split(image, bgr_plane);

	//�����������
	const int channels[1] = { 0 };
	const int bins[1] = { 256 };
	float hranges[2] = { 0,255 };
	const float* ranges[1] = { hranges };

	Mat b_hist;
	Mat g_hist;
	Mat r_hist;

	//����b,g,rͨ��ֱ��ͼ
	calcHist(&bgr_plane[0], 1, 0, Mat(), b_hist, 1, bins, ranges);
	calcHist(&bgr_plane[1], 1, 0, Mat(), g_hist, 1, bins, ranges);
	calcHist(&bgr_plane[2], 1, 0, Mat(), r_hist, 1, bins, ranges);

	// ��һ��ֱ��ͼ
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / bins[0]);
	Mat histImage(hist_h, hist_w, CV_8UC3);

	//��һ��
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	//����ֱ��ͼ����
	for (int i = 1; i < bins[0]; i++) {
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i-1),hist_h-cvRound(b_hist.at<float>(i))),Scalar(255,0,0),2,8,0);
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255), 2, 8, 0);
	}

	// ��ʾֱ��ͼ
	namedWindow("ֱ��ͼ", WINDOW_NORMAL);
	imshow("ֱ��ͼ", histImage);
}

void QuickDemo::Histogranm2D_demo(Mat& image)
{
	// ��ͼ��ת��ΪHSV��ɫ�ռ�
	// ��ͼ��ת��ΪHSV��ɫ�ռ�
	Mat hsv,hs_hist;
	cvtColor(image, hsv, COLOR_BGR2HSV);

	int hbins = 30, sbins = 32;
	int hist_bins[] = { hbins,sbins };
	float h_range[] = { 0,180 };
	float s_range[] = { 0,255 };
	const float* hs_ranges[] = { h_range,s_range };

	int hs_channels[] = { 0,1 };
	calcHist(&hsv, 1, hs_channels, Mat(), hs_hist, 2,hist_bins, hs_ranges, true, false);

	double maxVal = 0;
	minMaxLoc(hs_hist, 0, &maxVal, 0,0);
	int scale = 10;
	Mat hist2d_image = Mat::zeros(sbins * scale, hbins * scale, CV_8UC3);
	for (int h = 0; h < hbins; h++) {
		for (int s = 0; s < sbins; s++) {
			float binVal = hs_hist.at<float>(h, s);
			int intensity = cvRound(binVal * 255 / maxVal);
			rectangle(hist2d_image, Point(h * scale, s * scale), Point((h + 1) * scale-1, (s + 1) * scale-1), Scalar::all(intensity),-1);
		}
	}

	// ��ʾ��άֱ��ͼ
	namedWindow("��άֱ��ͼ", WINDOW_NORMAL);
	imshow("��άֱ��ͼ", hist2d_image);



}


