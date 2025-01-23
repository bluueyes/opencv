#include "quickopencv.h"
#include <functional>

void QuickDemo::colorSpace_Demo(const Mat& image) {

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

void QuickDemo::mat_creation_demo(const Mat& image) {
	Mat m2;
	Mat m1 = image.clone();		//深拷贝
	image.copyTo(m2);	//深拷贝 但是需要一个目标矩阵作为参数,会自动根据对目标矩阵进行行列调整
	if (image.data == m2.data) {
		std::cout << "m2==Image"<<std::endl;
	}
	else {
		std::cout << " m2 是 data 的副本" << std::endl;
	}
	if (image.data == m1.data) {
		std::cout << "m1==Image" << std::endl;
	}
	else {
		std::cout << " m1 是 data 的副本" << std::endl;
	}

	//创建空白图像
	Mat m3 = Mat::zeros(Size(8, 8), CV_8UC3);
	m3 = Scalar(255, 0, 0);
	std::cout << "width: "<< m3.cols <<" height: "<<m3.rows<<" channels: "<<m3.channels() << std::endl;

	std::cout << m3 << std::endl;

	Mat m4 = m3;	//Mat的拷贝构造是浅拷贝
	m4 = Scalar(0, 0, 255);

	namedWindow("图像", WINDOW_NORMAL);
	imshow("图像",m3);
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


	cv::namedWindow("像素读写演示", cv::WINDOW_NORMAL);
	imshow("像素读写演示",image);
}

void QuickDemo::operator_demo(Mat& image)
{
	Mat dst;
	//Scalar是一个用于表示多通道像素值的类。
	// 它通常用于表示颜色或其他多通道数据
	dst = image + Scalar(50, 50, 50);
	cv::namedWindow("加法操作", WINDOW_NORMAL);
	imshow("加法操作", dst);

	multiply(image, Scalar(2, 2, 2), image);
	cv::namedWindow("乘法操作", WINDOW_NORMAL);
	imshow("乘法操作", image);

	//Mat m = Mat::ones(image.size(), image.type());
	//m = Scalar(50, 50, 50);
	//int r = image.rows;
	//int c = image.cols;
	//int dims = image.channels();
	//for (int row = 0; row < r; row++) {
	//	for (int clo = 0; clo < c; clo++) {
	//		Vec3b bgr = image.at<Vec3b>(row, clo);
	//		Vec3b bgr2 = m.at<Vec3b>(row, clo);
	//		image.at<Vec3b>(row, clo)[0] = saturate_cast<uchar>(bgr[0] + bgr2[0]);	//saturate_cast<uchar> 能限定0~255之间
	//		image.at<Vec3b>(row, clo)[1] = saturate_cast<uchar>(bgr[1] + bgr2[1]);
	//		image.at<Vec3b>(row, clo)[2] = saturate_cast<uchar>(bgr[2] + bgr2[2]);
	//	}
	//}
	//cv::namedWindow("加法操作实现", WINDOW_NORMAL);
	//imshow("加法操作实现", dst);

	//除法
	divide(image, Scalar(10, 10, 10),image);
	cv::namedWindow("除法操作", WINDOW_NORMAL);
	imshow("除法操作", image);

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
	//addWeighted函数，用于将两幅图像按指定权重进行加权和。
	// 它通常用于图像混合、图像叠加等操作。
	addWeighted(data->src, data->alpha, data->m, data->beta, pos, data->dst);

	imshow("亮度与对比度调整", data->dst);
}

static void on_constrast(int pos, void* d)
{

	Data* data = static_cast<Data*>(d);
	data->m = Scalar(pos, pos, pos);

	double contrast = pos/ 100.0;
	addWeighted(data->src, contrast, data->m, data->beta, 0, data->dst);

	imshow("亮度与对比度调整", data->dst);
}

static int max_value = 100;

void QuickDemo::check_bar_demo(Mat& image)
{
	namedWindow("亮度与对比度调整",WINDOW_NORMAL);
	
	Data *data=new Data();
	data->dst = Mat::zeros(image.size(), image.type());
	data->m = Mat::zeros(image.size(), image.type());
	data->src = image;
	data->alpha = 1.0;
	data->beta = 0;

	int beginValue = 50;
	int contrastValue = 100;
    createTrackbar("Value Bar", "亮度与对比度调整", &beginValue, max_value,on_track,(void*)data);

	createTrackbar("Contrase Bar", "亮度与对比度调整", &contrastValue, max_value,on_constrast, (void*)data);



}

void QuickDemo::key_demo(Mat& image)
{
	Mat dst=Mat::zeros(image.size(),image.type());
	Mat temp=Mat::zeros(image.size(), image.type());;
	temp = Scalar(50, 50, 50);
	Mat original = image.clone(); // 保存原始图像

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

		imshow("键盘响应", dst);
	}

}

void QuickDemo::color_style_demo(Mat& image)
{
	Mat grayimage;
	//cvtColor函数是OpenCV中的一个函数，用于在不同颜色空间之间转换图像。
	// 它可以将图像从一种颜色空间转换为另一种颜色空间，
	// 例如从BGR转换为灰度图像、从BGR转换为HSV图像等
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
	//bitwise_xor 用于按位异或 
	bitwise_xor(m1, m2, dst);
	imshow("像素位操作", dst);
}

void QuickDemo::channels_demo(Mat& image)
{
	std::vector<Mat> mv;
	split(image, mv);	//split函数用于将多通道图像分离成单通道图像

	imshow("蓝色", mv[0]);
	imshow("绿色", mv[1]);
	imshow("红色", mv[2]);

	Mat dst;
	mv[0] = 0;
	mv[1] = 0;

	//merge函数用于将多个单通道图像合并成一个多通道图像
	merge(mv,dst);
	imshow("红色", dst);

	int from_to[] = { 0,2,1,1,2,0 }; //0->2 1->1 2->0
	mixChannels(&image,1 ,&dst, 1,from_to,3);
	imshow("通道混合", dst);
}

void QuickDemo::inrange_demo(Mat& image)
{
	//gbr通道由于颜色分布太广很难提取准确的范围，而hsv只有一个颜色通道更容易提取
	Mat hsv;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	Mat mask;
	//inRange函数是OpenCV中的一个函数，用于检查数组元素是否在指定范围内。
	// 它通常用于创建二值掩码图像，其中满足条件的像素被设置为白色（255），
	// 不满足条件的像素被设置为黑色（0）。这个函数在颜色分割和对象检测中非常有用。
	inRange(hsv, Scalar(0, 0, 200), Scalar(180, 50, 255), mask);
	imshow("mask",mask);

	Mat redback=Mat::zeros(image.size(),image.type());
	redback = Scalar(40, 40, 200);
	bitwise_not(mask, mask);
	imshow("mask", mask);

	image.copyTo(redback, mask);
	imshow("roi区域提取", redback);


}

void QuickDemo::pixel_static_demo(Mat& image)
{
	double minv, maxv;
	Point minLoc, maxLoc;
	std::vector<Mat> vec;
	split(image, vec);
	int i = 0;
	//minMaxLoc函数，用于查找数组或图像中元素的最小值和最大值及其位置
	for (auto& a : vec) {
		i++;
		minMaxLoc(a, &minv, &maxv, &minLoc, &maxLoc, Mat());
		std::cout << "Channel " << i << ": minvalue" << minv << " maxvalue" << maxv<<std::endl;
	}
	Mat mean, stddev;
	//meanStdDev函数，用于计算数组或图像的均值和标准差。
	// 它可以用于单通道或多通道图像，并返回每个通道的均值和标准差
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
	imshow("绘制演示", dest);
}

void QuickDemo::random_drawing_demo(Mat& image)
{
	Mat canvas = Mat::zeros(Size(512, 512), CV_8UC3);
	int w = canvas.cols;
	int h = canvas.rows;
	RNG rng(12345); //产生随机数
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

		imshow("展示", canvas);

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
	imshow("多边形绘制",canvas);


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
			
			// 在图像上绘制圆
			Point center((posbegin.x + posend.x) / 2, (posbegin.y + posend.y) / 2);
			int radius = maxs / 2;

		
			circle(*image, center, radius, Scalar(0, 255, 0), 1, 8, 0);
			imshow("鼠标绘制", *image);
			// 显示ROI区域
			if (posend.x > image->cols || posend.y > image->rows) return;
			Rect roi(center.x - radius, center.y - radius, radius * 2, radius * 2);
			Mat roiImage = temp(roi);
			imshow("ROI区域", roiImage);
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
			imshow("鼠标绘制", *image);
		}
	}
	
}

void QuickDemo::mouse_drawing_demo(Mat& image)
{
	namedWindow("鼠标绘制", WINDOW_AUTOSIZE);
	setMouseCallback("鼠标绘制", on_draw,&image);
	temp = image.clone();
	imshow("鼠标绘制",image);
}

void QuickDemo::norm_demo(Mat& image)
{
	Mat dst;
	std::cout << image.type() << std::endl;

	image.convertTo(image, CV_32FC3);
	std::cout << image.type() << std::endl;

	normalize(image, dst, 1.0, 0, NORM_MINMAX);
	std::cout << dst.type() << std::endl;
	imshow("图像数据归一化", dst);
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
	//flip(image, dst, 0);	//上下翻转
	//flip(image, dst, 1);	//左右翻转
	flip(image, dst, -1);	//180°旋转
	imshow("图像翻转", dst);
}

void QuickDemo::rotate_demo(Mat& image)
{
	Mat dst, M;
	int w = image.cols;
	int h = image.rows;
	//getRotationMatrix2D函数是OpenCV中的一个函数，
	// 用于生成一个用于图像旋转的2x3仿射变换矩阵。
	// 该矩阵可以用于旋转图像，同时可以指定旋转中心、旋转角度和缩放比例。
	M = getRotationMatrix2D(Point2f(w / 2, h / 2), 45, 1.0);
	double cos = abs(M.at<double>(0, 0));
	double sin = abs(M.at<double>(0, 1));

	int nw = cos * w + sin * h;
	int nh = cos * h + sin * w;

	M.at<double>(0, 2) += (nw / 2 - w / 2);
	M.at<double>(1, 2) += (nh / 2 - h / 2);
	//warpAffine函数是OpenCV中的一个函数，用于对图像进行仿射变换。
	// 仿射变换是一种线性变换，包括旋转、缩放、平移和剪切等操作。
	// warpAffine函数可以将输入图像应用指定的仿射变换矩阵，并输出变换后的图像。
	warpAffine(image, dst, M, Size(nw, nh), INTER_LINEAR, 0, Scalar(255, 255, 0));
	imshow("旋转演示", dst);
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
		if (c == 27) {	//退出
			break;
		}
	}
	capture.release();
}

void QuickDemo::showHistogranm_demo(Mat& image)
{
	//三个通道分离
	std::vector<Mat> bgr_plane;
	split(image, bgr_plane);

	//定义参数变量
	const int channels[1] = { 0 };
	const int bins[1] = { 256 };
	float hranges[2] = { 0,255 };
	const float* ranges[1] = { hranges };

	Mat b_hist;
	Mat g_hist;
	Mat r_hist;

	//计算b,g,r通道直方图
	calcHist(&bgr_plane[0], 1, 0, Mat(), b_hist, 1, bins, ranges);
	calcHist(&bgr_plane[1], 1, 0, Mat(), g_hist, 1, bins, ranges);
	calcHist(&bgr_plane[2], 1, 0, Mat(), r_hist, 1, bins, ranges);

	// 归一化直方图
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / bins[0]);
	Mat histImage(hist_h, hist_w, CV_8UC3);

	//归一化
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	//绘制直方图曲线
	for (int i = 1; i < bins[0]; i++) {
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i-1),hist_h-cvRound(b_hist.at<float>(i))),Scalar(255,0,0),2,8,0);
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255), 2, 8, 0);
	}

	// 显示直方图
	namedWindow("直方图", WINDOW_NORMAL);
	imshow("直方图", histImage);
}

void QuickDemo::Histogranm2D_demo(Mat& image)
{
	// 将图像转换为HSV颜色空间
	// 将图像转换为HSV颜色空间
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

	// 显示二维直方图
	namedWindow("二维直方图", WINDOW_NORMAL);
	imshow("二维直方图", hist2d_image);



}


