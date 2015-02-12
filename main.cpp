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

GLfloat yRotated;
vector<Point3f> points_3d;
Point3f camera_position;

struct opt_flow_parametrs{
	Size win_size = Size(11, 11);
	int max_level = 10;
	TermCriteria term_crit = TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
	int deriv_lamda = 0;
	int lk_flags = 0;
	double min_eig_threshold = 0.01;
};

struct feature_detect_parametrs{
	Size win_size = Size(5, 5);
	int max_ñorners = 1000;
	double quality_level = 0.001;
	double min_distance = 15;
	int block_size = 3;
	double k = 0.05;
};

void drawOptFlowMap(Mat flow, Mat &frame, int step)
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

void impositionOptFlow(Mat &frame, vector<Point> &faceKeyPoints, Mat &gray, Mat &prevgray){
	Mat flow, cflow;
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	resize(gray, gray, Size(frame.cols / 2, frame.rows / 2));

	if (prevgray.data)
	{
		calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 7, 10, 3, 7, 1.5, OPTFLOW_FARNEBACK_GAUSSIAN);
		drawOptFlowMap(flow, frame, 2);
	}
	swap(prevgray, gray);
}

void impositionOptFlowLK(vector<Point2f> &prev_features, vector<Point2f> &found_features, Mat &prevgray, Mat &gray, vector<float> &error, opt_flow_parametrs opf_parametrs){

	vector<uchar> status;
	if (prevgray.data)
	{
		calcOpticalFlowPyrLK(prevgray, gray, prev_features, found_features, status, error
			, opf_parametrs.win_size, opf_parametrs.max_level, opf_parametrs.term_crit, opf_parametrs.lk_flags, opf_parametrs.min_eig_threshold);

	}
	swap(prevgray, gray);
}


void init()
{
	glClearColor(0, 0, 0, 0);
}

void draw_point_cloud()
{

	glMatrixMode(GL_MODELVIEW);
	// clear the drawing buffer.
	glClear(GL_COLOR_BUFFER_BIT);
	glLoadIdentity();
	glTranslatef(camera_position.x, camera_position.y, camera_position.z);
	// rotation about X axis
	glRotatef(0, 1.0, 0.0, 0.0);
	// rotation about Y axis
	glRotatef(yRotated, 0.0, 1.0, 0.0);
	// rotation about Z axis
	glRotatef(0, 0.0, 0.0, 1.0);

	glBegin(GL_POINTS);

	for (unsigned int i = 0; i < points_3d.size(); i++){
		if (points_3d.at(i).x != 0 && points_3d.at(i).y != 0 && points_3d.at(i).z)
			glVertex3f(points_3d.at(i).x, points_3d.at(i).y, points_3d.at(i).z);
	}
	glEnd();
	glFlush();
}

void animation()
{
	yRotated += 0.01;
	draw_point_cloud();
}

void reshape(int x, int y)
{
	if (y == 0 || x == 0) return;  //Nothing is visible then, so return
	//Set a new projection matrix
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//Angle of view:40 degrees
	//Near clipping plane distance: 0.5
	//Far clipping plane distance: 20.0

	gluPerspective(40.0, (GLdouble)x / (GLdouble)y, 0.5, 20.0);
	glMatrixMode(GL_MODELVIEW);
	glViewport(0, 0, x, y);  //Use the whole window for rendering
}


void calculation_SFM(Mat &depth_map, vector <Point2f> found_opfl_points, vector <Point2f> prev_opfl_points){
	if (found_opfl_points.size() != found_opfl_points.size()){
		return;
	}

	double focal_length = 30;		//mm
	double pixel_length = 0.2646;	//mm


	const int size = 300;
	Matx<float, 4, size> W;
	Matx <float, 4, 1> U;
	Matx <float, 4, 4>	D;
	Matx <float, 4, size> Vt;
	points_3d = vector<Point3f>();

	Point2f prev_mass_center = Point2f(0, 0);
	Point2f found_mass_center = Point2f(0, 0);
	for (unsigned int i = 0; i < found_opfl_points.size(); i++){
		prev_mass_center = Point2f(prev_opfl_points.at(i).x + prev_mass_center.x
			, prev_opfl_points.at(i).y + prev_mass_center.y);

		found_mass_center = Point2f(found_opfl_points.at(i).x + found_mass_center.x
			, found_opfl_points.at(i).y + found_mass_center.y);
	}
	prev_mass_center = Point2f(prev_mass_center.x / prev_opfl_points.size(), prev_mass_center.y / prev_opfl_points.size());
	found_mass_center = Point2f(found_mass_center.x / found_opfl_points.size(), found_mass_center.y / found_opfl_points.size());

	for (unsigned int i = 0; i < found_opfl_points.size(); i++){
		found_opfl_points.at(i) = Point2f(found_opfl_points.at(i).x - found_mass_center.x, found_opfl_points.at(i).y - found_mass_center.y);
		prev_opfl_points.at(i) = Point2f(prev_opfl_points.at(i).x - prev_mass_center.x, prev_opfl_points.at(i).y - prev_mass_center.y);
	}



	for (int i = 0; i < size; i++)
	{
		if (i < prev_opfl_points.size()){
			W.val[i] = prev_opfl_points.at(i).x;
			W.val[i + size] = prev_opfl_points.at(i).y;
			W.val[i + size * 2] = found_opfl_points.at(i).x;
			W.val[i + size * 3] = found_opfl_points.at(i).y;
		}
		else
			break;
	}

	SVD::compute(W, U, D, Vt);

	camera_position = Point3f(U.val[0], U.val[1], U.val[2]);

	for (int i = 0; i < size; i++)
	{
		points_3d.push_back(Point3f(Vt.val[i], Vt.val[i + size], Vt.val[i + size * 2]));
	}

	glutDisplayFunc(draw_point_cloud);
	glutReshapeFunc(reshape);
	//Set the function for the animation.
	glutIdleFunc(animation);
	glutMainLoop();

}

void calculation_simple_Z(Mat &img1, Mat &img2, vector <Point2f> found_opfl_points, vector <Point2f> prev_opfl_points)
{

	//double focal_length = 30;		//mm
	//double pixel_length = 0.2646;	//mm

	points_3d = vector<Point3f>();

	double f = 300;
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
		points_3d.push_back(Point3f(X, Y, Z));
	}

	glutDisplayFunc(draw_point_cloud);
	glutReshapeFunc(reshape);
	//Set the function for the animation.
	glutIdleFunc(animation);
	glutMainLoop();

}

int main(int argc, char* argv[])
{
	vector <Point2f> prev_opfl_points;
	Mat frame, prevgray, gray;
	opt_flow_parametrs opf_parametrs;
	feature_detect_parametrs fd_parametrs;

	VideoCapture cap(0);
	if (!cap.isOpened()){
		printf("Camera could not be opened");
		return -1;
	}

	cap >> frame;

	glutInit(&argc, argv);
	//we initizlilze the glut. functions
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowPosition(frame.cols / 2, frame.rows / 2);
	glutCreateWindow("Point Cloud");
	init();

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
		vector<float> error;


		if (second_frame){
			cvtColor(frame, gray, COLOR_BGR2GRAY);

			vector <Point2f> found_opfl_points;
			impositionOptFlowLK(prev_opfl_points, found_opfl_points, prevgray, gray, error, opf_parametrs);


			for (unsigned int i = 0; i < found_opfl_points.size(); i++){
				if (error.at(i) == 0){
					circle(frame, found_opfl_points.at(i), 1, CV_RGB(128, 128, 255), 2, 8, 0);
					line(frame, prev_opfl_points.at(i), found_opfl_points.at(i), CV_RGB(64, 64, 128));
				}
			}

			////////////////////////////////////////////
			imshow("frame", frame);
			//calculation_SFM(frame, found_opfl_points, prev_opfl_points);


			calculation_simple_Z(prevgray, gray, found_opfl_points, prev_opfl_points);
			////////////////////////////////////////////

			prev_opfl_points = vector<Point2f>();
			goodFeaturesToTrack(prevgray, prev_opfl_points
				, fd_parametrs.max_ñorners, fd_parametrs.quality_level, fd_parametrs.min_distance, Mat(), fd_parametrs.block_size, 0, fd_parametrs.k);

		}
		else
		{
			if (waitKey(33) == 27)	break;
			imshow("frame", frame);
			cvtColor(frame, prevgray, COLOR_BGR2GRAY);
			prev_opfl_points = vector<Point2f>();
			goodFeaturesToTrack(prevgray, prev_opfl_points
				, fd_parametrs.max_ñorners, fd_parametrs.quality_level, fd_parametrs.min_distance, Mat(), fd_parametrs.block_size, 0, fd_parametrs.k);
			if (prev_opfl_points.size() != 0)
				second_frame = true;
		}
	}

	return 0;
}