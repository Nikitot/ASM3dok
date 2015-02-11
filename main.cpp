#include <cv.h>
#include <stdio.h>
#include <stdlib.h>
#include "highgui.h"
#include "cxcore.h"
#include <string>
#include <sstream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>
#include <cxcore.h>
#include <gl/glew.h>
#include <gl/freeglut.h>

using namespace std;
using namespace cv;

struct opt_flow_parametrs{
	Size win_size = Size(15, 15);
	int max_level = 10;
	TermCriteria term_crit = TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
	int deriv_lamda = 0;
	int lk_flags = 0;
	double min_eig_threshold = 0.01;
};

struct feature_detect_parametrs{
	Size win_size = Size(15, 15);
	int max_ñorners = INT_MAX;
	double quality_level = 0.001;
	double min_distance = 15;
	int block_size = 3;
	double k = 0.05;
};

void draw_opt_flow_map(Mat flow, Mat &frame, int step)
{
	for (int y = 0; y < flow.rows / step; y++){
		for (int x = 0; x < flow.cols / step; x++)
		{
			const Point2f& fxy = flow.at<Point2f>(y*step, x*step);

			Point p1(x*step, y*step);
			Point p2(round(x*step + fxy.x), round(y*step + fxy.y));

			//line(frame, Point(p1.x * 2, p1.y * 2), Point(p2.x * 2, p2.y * 2), CV_RGB(255, 255, 255));

			float length_fxy = pow(pow(fxy.x, 2) + pow(fxy.y, 2), 0.5);
			float length_xy = pow(pow(step, 2) + pow(step, 2), 0.5);


			float color = length_fxy * 20;
			circle(frame, Point(p1.x * 2, p1.y * 2), 2, Scalar(int(color), int(color), int(color)));
		}
	}
}

void imposition_opt_flow(Mat &frame, vector<Point> &faceKeyPoints, Mat &gray, Mat &prevgray){
	Mat flow, cflow;
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	resize(gray, gray, Size(frame.cols / 2, frame.rows / 2));

	if (prevgray.data)
	{
		calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 7, 10, 3, 7, 1.5, OPTFLOW_FARNEBACK_GAUSSIAN);
		draw_opt_flow_map(flow, frame, 2);
	}
	swap(prevgray, gray);
}

void imposition_opt_flow_LK(vector<Point2f> &prev_features, vector<Point2f> &found_features, Mat &prevgray, Mat &gray, vector<float> &error, vector<uchar> &status, opt_flow_parametrs opf_parametrs){

	if (prevgray.data)
	{
		calcOpticalFlowPyrLK(prevgray, gray, prev_features, found_features, status, error
			, opf_parametrs.win_size, opf_parametrs.max_level, opf_parametrs.term_crit, opf_parametrs.lk_flags, opf_parametrs.min_eig_threshold);

	}
	swap(prevgray, gray);
}

//Not used
void calculation_simple_Z(Mat &img1, Mat &img2, vector <Point2f> found_opfl_points, vector <Point2f> prev_opfl_points)
{

	//double focal_length = 30;		//mm
	//double pixel_length = 0.2646;	//mm

	//points_3d = vector<Point3f>();

	double f = 30;
	double B = 1;
	int w = img1.cols;
	int h = img1.rows;

	float X = 0;
	float Y = 0;
	float Z = 0;

	for (unsigned int i = 0; i < found_opfl_points.size(); i++)
	{
		float delta_x = abs(found_opfl_points.at(i).x - prev_opfl_points.at(i).x);
		float delta_y = abs(found_opfl_points.at(i).y - prev_opfl_points.at(i).y);
		double delta = pow(pow(delta_x, 2) + pow(delta_y, 2), 0.5);

		Z = (B * f) / delta;
		X = (prev_opfl_points.at(i).x + prev_opfl_points.at(i).y) / 2;
		Y = (prev_opfl_points.at(i).y + prev_opfl_points.at(i).y) / 2;
		//points_3d.push_back(Point3f(X, Y, Z));
	}

	//glutDisplayFunc(draw_point_cloud);
	//glutReshapeFunc(reshape);
	//Set the function for the animation.
	//glutIdleFunc(animation);
	//glutMainLoop();

}

void good_features_init(Mat frame, Rect rect_face, vector<Point2f> &prev_opfl_points)
{
	feature_detect_parametrs fd_parametrs;
	Mat gray_face, face;
	Mat image_roi = frame(rect_face);
	image_roi.copyTo(face);
	cvtColor(face, gray_face, COLOR_BGR2GRAY);

	prev_opfl_points = vector<Point2f>();
	goodFeaturesToTrack(gray_face, prev_opfl_points
		, fd_parametrs.max_ñorners, fd_parametrs.quality_level, fd_parametrs.min_distance, Mat(), fd_parametrs.block_size, 0, fd_parametrs.k);
	for (unsigned int i = 0; i < prev_opfl_points.size(); i++)
	{
		prev_opfl_points.at(i) = Point2f(prev_opfl_points.at(i).x + rect_face.x, prev_opfl_points.at(i).y + rect_face.y);
	}
}

int main(int argc, char* argv[])
{
	vector <Point2f> prev_opfl_points;
	Mat frame, prev_gray_frame, gray_frame, face;
	opt_flow_parametrs opf_parametrs;

	CascadeClassifier face_cascade;
	face_cascade.load("E:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml");

	VideoCapture cap(0);
	if (!cap.isOpened()){
		printf("Camera could not be opened");
		return -1;
	}

	cap >> frame;

	VideoWriter writer("./video_result.avi", 0, 15, frame.size(), true);
	if (!writer.isOpened())
	{
		printf("Output video could not be opened");
		return -1;
	}

	bool second_frame = false;

	while (1){
		if (waitKey(33) == 27)	break;
		cap >> frame;

		vector<Rect> faces;
		face_cascade.detectMultiScale(frame, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(150, 150));

		if (faces.size() != 0)
		{
			vector<float> error;
			vector<uchar> status;

			if (second_frame){


				cvtColor(frame, gray_frame, COLOR_BGR2GRAY);
				vector <Point2f> found_opfl_points;
				imposition_opt_flow_LK(prev_opfl_points, found_opfl_points, prev_gray_frame, gray_frame, error, status, opf_parametrs);


				for (unsigned int i = 0; i < found_opfl_points.size(); i++){
					if (error.at(i) == 0){
						float delta_x = abs(found_opfl_points.at(i).x - prev_opfl_points.at(i).x);
						float delta_y = abs(found_opfl_points.at(i).y - prev_opfl_points.at(i).y);
						double delta = pow(pow(delta_x, 2) + pow(delta_y, 2), 0.5);
						
						Point2f frame_coord = Point2f(found_opfl_points.at(i).x + faces[0].x);

						if (delta < frame.cols / 10 && status.at(i) == '\0'){

							circle(frame, found_opfl_points.at(i), 1, CV_RGB(128, 128, 255), 2, 8, 0);
							line(frame, prev_opfl_points.at(i), found_opfl_points.at(i), CV_RGB(64, 64, 128));
							
							//prev_opfl_points.at(i) = found_opfl_points.at(i);
						}
						else
						{
							circle(frame, found_opfl_points.at(i), 1, CV_RGB(200, 0, 0), 2, 8, 0);
						}

					}
				}

				imshow("frame", frame);

				//calculation_SFM(frame, found_opfl_points, prev_opfl_points);
				//calculation_simple_Z(prevgray, gray, found_opfl_points, prev_opfl_points);


				good_features_init(frame, faces[0], prev_opfl_points);
			}
			else
			{
				if (waitKey(33) == 27)	break;
				
				good_features_init(frame, faces[0], prev_opfl_points);
				
				if (prev_opfl_points.size() != 0)
					second_frame = true;
			}
		}
		
	}

	return 0;
}