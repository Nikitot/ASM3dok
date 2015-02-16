#include <windows.h>  // for MS Windows
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

//gl
#include <GL/glut.h>  // GLUT, include glu.h and gl.h

#include <windows.h> 
#define _USE_MATH_DEFINES
#include <math.h>
#define FOCAL_LENGTH 1000

//gl
#define RED 1
#define GREEN 2
#define BLUE 3
#define WHITE 4

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

GLfloat angleX = 0, angleY = 0, angleZ = 0;

vector <Point2f> prev_opfl_points;
vector<CvPoint3D32f> modelPoints;
posit_orientation orientation;
Mat frame, prev_gray_frame, gray_frame, face;
opt_flow_parametrs opf_parametrs;
Scalar color;
bool second_frame = false;
VideoCapture cap(0);

CascadeClassifier face_cascade;

double deg2rad(double deg)
{
	return (M_PI * deg / 180.0);
}

double rad2deg(double rad)
{
	return (180.0 * rad / (M_PI));
}

Point3f projectPointsOnCylinder(Point2f point, float r) {
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
	, int &good_points_count, Rect face_rect, vector<float> &error, vector<uchar> &status, opt_flow_parametrs opf_parametrs, Scalar color){

	Mat gray_frame;
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

		if (delta < gray_frame.cols / 15 /*&& status.at(i) == '\0'*/){

			circle(frame, found_opfl_points.at(i), 1, color, 2, 8, 0);
			line(frame, prev_opfl_points.at(i), found_opfl_points.at(i), color);

			prev_opfl_points.at(i) = found_opfl_points.at(i);
			good_points_count++;
		}
		else
		{
			circle(frame, found_opfl_points.at(i), 1, CV_RGB(200, 0, 0), 2, 8, 0);
		}
	}
	swap(prevgray, gray_frame);
}

//!
void calculation_POSIT(vector<CvPoint3D32f> &modelPoints, vector<Point2f> &found_opfl_points, Size half_size, int good_points_count)
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

void goodfeatures_and_cylinder_init(Mat frame, Rect rect_face, vector<Point2f> &prev_opfl_points, vector<CvPoint3D32f> &modelPoints, Scalar &color)
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
	float cylWidth = (float)rect_face.width;

	prev_opfl_points.clear();

	goodFeaturesToTrack(gray_face, prev_opfl_points, 
		fd_parametrs.max_ñorners, fd_parametrs.quality_level, fd_parametrs.min_distance, Mat(), fd_parametrs.block_size, 0, fd_parametrs.k);


	modelPoints.clear();

	for (unsigned int i = 0; i < prev_opfl_points.size(); i++)
	{
		prev_opfl_points.at(i) = Point2f(prev_opfl_points.at(i).x + rect_face.x, prev_opfl_points.at(i).y + rect_face.y);
		Point2f centeredPoint = Point2f(half_width - prev_opfl_points.at(i).x, half_height - prev_opfl_points.at(i).y);
		Point3f cylinderPoint = projectPointsOnCylinder(centeredPoint, cylWidth);
		modelPoints.push_back(cvPoint3D32f(cylinderPoint.x, cylinderPoint.y, cylinderPoint.z));
	}
}

void init()
{
	glClearColor(0, 0, 0, 0);
	//glRotatef(90, 1.0, 0.0, 0.0);
}

void draw_angle( )
{
	//posit_orientation pos_or;

	glMatrixMode(GL_MODELVIEW);
	// clear the drawing buffer.
	glClear(GL_COLOR_BUFFER_BIT);
	glLoadIdentity();

	//Far away from center
	glTranslatef(-1, -1, -11);
	//glRotatef(90, 1.0, 0.0, 0.0);

	//rotation about X axis
	//glRotatef(angleX, 1.0, 0.0, 0.0);

	glRotatef(rad2deg(orientation.roll), 1.0, 0.0, 0.0);

	// rotation about Y axis
	//glRotatef(angleY, 0.0, 1.0, 0.0);
	
	glRotatef(rad2deg(orientation.pitch)-90, 0.0, 1.0, 0.0);

	// rotation about Z axis
	//glRotatef(angleZ, 0.0, 0.0, 1.0);
	glRotatef(rad2deg(orientation.yaw), 0.0, 0.0, 1.0);

	cout << orientation.roll << " " << orientation.pitch << " " << orientation.yaw << endl;

	//X-red
	glBegin(GL_LINE_STRIP);

	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3f(2.00, 0.00, 0.00);
	glColor3f(1.0f, 1.0f, 1.0f);
	glVertex3f(-2.00, 0.00, 0.00);
	glEnd();
	
	//Y-green
	glBegin(GL_LINE_STRIP);

	glColor3f(0.0f, 1.0f, 0.0f);
	glVertex3f(0.00, 2.00, 0.00);
	glColor3f(0.0f, 1.0f, 0.0f);
	glVertex3f(0.00, -2.00, 0.00);

	glEnd();

	//Z-blue
	glBegin(GL_LINE_STRIP);

	glColor3f(0.0f, 0.0f, 1.0f);
	glVertex3f(0.00, 0.00, 2.00);
	glColor3f(0.0f, 0.0f, 1.0f);
	glVertex3f(0.00, 0.00, -2.00);

	glEnd();
	
	glFlush();
}

void animation( )
{

	cap >> frame;

	Size2f half_size = Size2f(frame.cols / 2.0f, frame.rows / 2.0f);

	VideoWriter writer("./video_result.avi", 0, 15, frame.size(), true);
	if (!writer.isOpened())
	{
		printf("Output video could not be opened");
		//return -1;
	}

	/*while (1){
		if (waitKey(33) == 27)	
			break;
	*/
		cap >> frame;

		//imshow("frame1", frame);

		vector<Rect> faces;
		face_cascade.detectMultiScale(frame, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(100, 100));

		if (faces.size() != 0)
		{
			vector<float> error;
			vector<uchar> status;

			Rect face_rect = Rect(
				faces[0].x + faces[0].width / 4
				, faces[0].y + faces[0].height / 3.5
				, faces[0].width - faces[0].width / 2
				, faces[0].height - faces[0].height / 1.3
				);

			if (second_frame){

				vector<Point2f> found_opfl_points;
				int good_points_count = 0;

				imposition_opt_flow_LK(prev_opfl_points, found_opfl_points, prev_gray_frame, frame, good_points_count, face_rect
					, error, status, opf_parametrs, color);

				calculation_POSIT(modelPoints, found_opfl_points, half_size, good_points_count);

				string roll1 = "roll: " + to_string(rad2deg(orientation.roll));
				string pitch1 = "pitch: " + to_string(rad2deg(orientation.pitch));
				string yaw1 = "yaw: " + to_string(rad2deg(orientation.yaw));
				putText(frame, roll1, Point(10, 30), 2, 1.0, CV_RGB(255, 255, 255));
				putText(frame, pitch1, Point(10, 60), 2, 1.0, CV_RGB(255, 255, 255));
				putText(frame, yaw1, Point(10, 90), 2, 1.0, CV_RGB(255, 255, 255));

				imshow("frame", frame);

				writer.write(frame);

				if (waitKey(33) == 13){
					goodfeatures_and_cylinder_init(frame, face_rect, prev_opfl_points, modelPoints, color);
				}
			}

			else
			{
				imshow("frame", frame);
				//if (waitKey(33) == 27)	
					//break;

				goodfeatures_and_cylinder_init(frame, face_rect, prev_opfl_points, modelPoints, color);
				cvtColor(frame, prev_gray_frame, COLOR_BGR2GRAY);

				if (prev_opfl_points.size() != 0)
					second_frame = true;

			}
		}

	//}

	writer.release();

	draw_angle();
	
}

void reshape(int x, int y)
{
	if (y == 0 || x == 0) return; //Nothing is visible then, so return

	//Set a new projection matrix
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//Angle of view:40 degrees
	//Near clipping plane distance: 0.5
	//Far clipping plane distance: 20.0
	gluPerspective(40.0, (GLdouble)x / (GLdouble)y, 0.5, 20.0);
	glMatrixMode(GL_MODELVIEW);
	glViewport(0, 0, x, y); //Use the whole window for rendering
}

int main(int argc, char* argv[])
{
	
	face_cascade.load("C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml");
	
	if (!cap.isOpened()) {
		printf("Camera could not be opened");
		//return -1;
	}

	//gl
	glutInit(&argc, argv);
	// we initizlilze the glut. functions
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(800, 600);
	glutInitWindowPosition(40, 40);
	glutCreateWindow("Feflection of angle rotation.");
	init();

	//glRotatef(90, 1.0, 0.0, 0.0);

	//gl
	glutDisplayFunc(draw_angle);
	glutReshapeFunc(reshape);

	glutIdleFunc(animation);

	glutMainLoop();

	return 0;

}