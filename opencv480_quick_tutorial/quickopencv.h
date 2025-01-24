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
	void key_demo(Mat& image);
	void color_style_demo(Mat& image);
	void bitwise_demo(Mat& image);
	void channels_demo(Mat& image);
	void inrange_demo(Mat& image);
	void pixel_static_demo(Mat& image);
	void draw_demo(Mat& image);
	void random_drawing_demo(Mat& image);
	void polyline_draw_demo(Mat& image);
	void mouse_drawing_demo(Mat& image);
	void norm_demo(Mat& image);
	void resize_demo(Mat& image);
	void flip_demo(Mat& image);
	void rotate_demo(Mat& image);
	void video_demo(Mat& iamge);
	void showHistogranm_demo(Mat& image);
	void Histogranm2D_demo(Mat& image);
	void histogram_eq_demo(Mat& image);
	void histogram_eq_color_demo(Mat& iamge);
	void blur_demo(Mat& image);
	void gaussian_blur_demo(Mat& image);
	void bifilter_demo(Mat& image);
};