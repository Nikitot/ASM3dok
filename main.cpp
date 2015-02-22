#include <cv.h>
#include <stdio.h>
#include <stdlib.h>
#include <highgui.h>
#include <string>
#include <sstream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <cxcore.h>
#include <gl/glew.h>
#include <gl/freeglut.h>
#include <math.h>
#include "stereoscopy.h"

#define _USE_MATH_DEFINES
#define FOCAL_LENGTH 1000


using namespace std;
using namespace cv;

struct opt_flow_parametrs{
	Size win_size = Size(15, 15);
	int max_level = 10;
	TermCriteria term_crit = TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
	int deriv_lamda = 2;
	int lk_flags = 0;
	double min_eig_threshold = 0.01;
};

struct feature_detect_parametrs{
	Size win_size = Size(15, 15);
	int max_ñorners = INT_MAX;
	double quality_level = 0.01;
	double min_distance = 15;
	int block_size = 3;
	double k = 0.05;
};

struct posit_orientation{
	double roll = 0;
	double pitch = 0;
	double yaw = 0;
};

opt_flow_parametrs opf_parametrs;
static const double pi = 3.14159265358979323846;

double deg2rad(double deg)
{
	return (pi * deg / 180.0);
}

double rad2deg(double rad)
{
	return (180.0 * rad / (pi));
}

Point3f projectPointsOnCylinder(cv::Point2f point, float r) {
	cv::Point3f p;
	p.x = point.x;
	p.y = point.y;
	p.z = -(float)(sqrt(r*r - point.x*point.x));
	return p;
}

//Not used
void calculation_simple_Z(Mat &img1, Mat &img2, vector <Point2f> found_opfl_points, vector <Point2f> prev_opfl_points)
{
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
	}
}

void imposition_opt_flow_LK(vector<Point2f> &prev_opfl_points, vector<Point2f> &found_opfl_points, Mat &prevgray, Mat &frame
	, int &good_points_count, Rect face_rect,  Scalar color){

	Mat gray_frame;

	vector<Point2f> found_opfl_points_good;
	vector<Point2f> prev_opfl_points_good;
	vector<float> error;
	vector<uchar> status;
		cvtColor(frame, gray_frame, COLOR_BGR2GRAY);
	if (prevgray.data)
	{
		calcOpticalFlowPyrLK(prevgray, gray_frame, prev_opfl_points, found_opfl_points, status
			, error, opf_parametrs.win_size, opf_parametrs.max_level, opf_parametrs.term_crit
			, opf_parametrs.lk_flags, opf_parametrs.min_eig_threshold);
	}

	good_points_count = 0;
	for (unsigned int i = 0; i < found_opfl_points.size(); i++){
		float delta_x = abs(found_opfl_points.at(i).x - prev_opfl_points.at(i).x);
		float delta_y = abs(found_opfl_points.at(i).y - prev_opfl_points.at(i).y);
		double delta = pow(pow(delta_x, 2) + pow(delta_y, 2), 0.5);

		Point2f frame_coord = Point2f(found_opfl_points.at(i).x + face_rect.x);

		if (delta < gray_frame.cols / 15 && status.at(i) == '\0' ){
			circle(frame, found_opfl_points.at(i), 1, color, 2, 8, 0);
			line(frame, prev_opfl_points.at(i), found_opfl_points.at(i), color);
			found_opfl_points_good.push_back(found_opfl_points.at(i));
			prev_opfl_points_good.push_back(prev_opfl_points.at(i));
			good_points_count++;
		}
		else
		{
			circle(frame, found_opfl_points.at(i), 1, CV_RGB(200, 0, 0), 2, 8, 0);
		}

	}
	prev_opfl_points.clear();
	found_opfl_points.clear();

	prev_opfl_points = prev_opfl_points_good;
	found_opfl_points = found_opfl_points_good;

	swap(prevgray, gray_frame);
}

void calculation_POSIT(vector<CvPoint3D32f> &modelPoints, vector<Point2f> &found_opfl_points, Size half_size, posit_orientation &orientation, int good_points_count)
{
	if (good_points_count > 6){
		vector<Point3f> objectPoints;
		vector<CvPoint2D32f> points;

		CvMatr32f rotation_matrix = new float[9];
		CvVect32f translation_vector = new float[3];

		CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 100, 1.0e-4f);
		CvPOSITObject *positObject = cvCreatePOSITObject(&modelPoints[0], (int)modelPoints.size());
		points.clear();
		for (int i = 0; i < found_opfl_points.size(); i++) {
			points.push_back(cvPoint2D32f(half_size.width - found_opfl_points.at(i).x, half_size.width - found_opfl_points.at(i).y));
		}

		cvPOSIT(positObject, &points[0], FOCAL_LENGTH, criteria, rotation_matrix, translation_vector);

		orientation.roll = atan(rotation_matrix[3] / rotation_matrix[0]);
		orientation.pitch = atan((-rotation_matrix[6]) / (rotation_matrix[0]))*cos(orientation.roll) + rotation_matrix[3] * sin(orientation.roll);
		orientation.yaw = atan(rotation_matrix[7] / rotation_matrix[8]);
	}
}

void goodfeatures_and_cylinder_init(Mat frame, Rect rect_face, vector<Point2f> &prev_opfl_points, vector<CvPoint3D32f> &modelPoints, posit_orientation &orientation, Scalar &color)
{
	orientation.roll = 0;
	orientation.pitch = 0;
	orientation.yaw = 0;

	color = Scalar(rand() % 255 + 1, rand() % 255 + 1, rand() % 255 + 1);

	feature_detect_parametrs fd_parametrs;
	Mat gray_face, face;
	Mat image_roi = frame(rect_face);
	image_roi.copyTo(face);
	cvtColor(face, gray_face, COLOR_BGR2GRAY);

	float half_width = frame.cols / 2.0;
	float half_height = frame.rows / 2.0;
	float cylWidth = (float)rect_face.width*2;

	prev_opfl_points.clear();

	goodFeaturesToTrack(gray_face, prev_opfl_points
		, fd_parametrs.max_ñorners, fd_parametrs.quality_level, fd_parametrs.min_distance, Mat(), fd_parametrs.block_size, 0, fd_parametrs.k);


	modelPoints.clear();

	for (unsigned int i = 0; i < prev_opfl_points.size(); i++)
	{
		prev_opfl_points.at(i) = Point2f(prev_opfl_points.at(i).x + rect_face.x, prev_opfl_points.at(i).y + rect_face.y);
		Point2f centeredPoint = Point2f(half_width - prev_opfl_points.at(i).x, half_height - prev_opfl_points.at(i).y);
		Point3f cylinderPoint = projectPointsOnCylinder(centeredPoint, cylWidth);
		modelPoints.push_back(cvPoint3D32f(cylinderPoint.x, cylinderPoint.y, cylinderPoint.z));
	}
}

int main(int argc, char* argv[])
{
	vector <Point2f> prev_opfl_points;
	vector<CvPoint3D32f> modelPoints;
	Mat frame, prev_gray_frame, gray_frame, face;
	posit_orientation orientation;
	Scalar color;
	bool second_frame = false;

	CascadeClassifier face_cascade;
	face_cascade.load("E:/opencv2410/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml");

	VideoCapture cap(0);
	if (!cap.isOpened()){
		printf("Camera could not be opened");
		return -1;
	}

	cap >> frame;

	Size2f half_size = Size2f(frame.cols / 2.0f, frame.rows / 2.0f);

	VideoWriter writer("./video_result.avi", 0, 15, frame.size(), true);
	if (!writer.isOpened())
	{
		printf("Output video could not be opened");
		return -1;
	}

	while (1){
		if (waitKey(33) == 27)	break;
		cap >> frame;

		vector<Rect> faces;
		face_cascade.detectMultiScale(frame, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(100, 100));

		if (faces.size() != 0)
		{
			vector<float> error;
			vector<uchar> status;

			Rect face_rect = faces[0];

			if (second_frame){
				vector<Point2f> found_opfl_points;
				int good_points_count = 0;

				xyz_coords_t *prev_points,*found_points;

				imposition_opt_flow_LK(prev_opfl_points, found_opfl_points, prev_gray_frame, frame, good_points_count, face_rect, color);


				//calculation_POSIT(modelPoints, found_opfl_points, half_size, orientation, good_points_count);
				string roll1 = "roll: " + to_string(rad2deg(orientation.roll));
				string pitch1 = "pitch: " + to_string(rad2deg(orientation.pitch));
				string yaw1 = "yaw: " + to_string(rad2deg(orientation.yaw));
				putText(frame, roll1, Point(10, 30), 2, 1.0, CV_RGB(255, 255, 255));
				putText(frame, pitch1, Point(10, 60), 2, 1.0, CV_RGB(255, 255, 255));
				putText(frame, yaw1, Point(10, 90), 2, 1.0, CV_RGB(255, 255, 255));

				prev_points = alloc_xyz_coords(good_points_count);
				found_points = alloc_xyz_coords(good_points_count);

				for (int i = 0; i < good_points_count; i++)
				{
					prev_points->x[i] = prev_opfl_points.at(i).x;
					prev_points->y[i] = prev_opfl_points.at(i).y;
					prev_points->z[i] = 0;

					found_points->x[i] = found_opfl_points.at(i).x;
					found_points->y[i] = found_opfl_points.at(i).y;
					found_points->z[i] = 0;
				}

				get_3d_object_coords(prev_points,0.0,found_points,10.0);

				imshow("frame", frame);				
				writer.write(frame);				

				//if (cv::waitKey(33) == 13) {
				//	//ïåðâûé ñïîñîá
				//	ofstream F1,F2;
				//	F1.open("./point1.txt");
				//	F2.open("./point2.txt");
				//	for (unsigned int i = 0; i < found_opfl_points.size(); i++){
				//		F1 << prev_opfl_points.at(i).x << "\t" << prev_opfl_points.at(i).y << endl;
				//		F2 << found_opfl_points.at(i).x << "\t" << found_opfl_points.at(i).y << endl;
				//	}
				//	F1.close();
				//	F2.close();
				//}
				goodfeatures_and_cylinder_init(frame, face_rect, prev_opfl_points, modelPoints, orientation, color);
			}
			else
			{
				if (waitKey(33) == 27)	break;

				goodfeatures_and_cylinder_init(frame, face_rect, prev_opfl_points, modelPoints, orientation, color);
				cvtColor(frame, prev_gray_frame, COLOR_BGR2GRAY);

				if (prev_opfl_points.size() != 0)
					second_frame = true;
			}
		}
	}
	writer.release();

	return 0;
}