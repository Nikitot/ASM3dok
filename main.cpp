#include <cv.h>
#include <stdio.h>
#include <stdlib.h>
#include "highgui.h"
#include "stasm_lib.h"
#include <string>
#include <sstream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/core.hpp>
#include <cxcore.h>

using namespace std;
using namespace cv;

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
	int max_ñorners = INT_MAX;
	double quality_level = 0.01;
	double min_distance = 10;
	int block_size = 3;
	double k = 0.05;
};

void shiftImageAndPointsFromBorder(Mat &frame, vector<Point> &allPoints, Point shift){
	Point2f srcTri[3], dstTri[3];
	Mat warp_mat = Mat(2, 3, CV_32FC1);
	Mat dst = frame.clone();

	srcTri[0] = Point2f(0, 0);						//src Top left
	srcTri[1] = Point2f(frame.cols - 1, 0);			//src Top right
	srcTri[2] = Point2f(0, frame.rows - 1);			//src Bottom left

	dstTri[0] = Point2f(0 + shift.x, 0 + shift.y);						//dst Top left
	dstTri[1] = Point2f(frame.cols - 1 + shift.x, 0 + shift.y);			//dst Top right
	dstTri[2] = Point2f(0 + shift.x, frame.rows - 1 + shift.y);			//dst Bottom left

	// âûïîëíÿåì òðàíñôîðìàöèþ
	warpAffine(frame, dst, getAffineTransform(srcTri, dstTri), dst.size());
	frame = dst;

	for (unsigned int i = 0; i < allPoints.size(); i++){
		allPoints.at(i).x += shift.x;
		allPoints.at(i).y += shift.y;
	}
}

void face_points_stabilisation(Mat &frame, vector<Point> &allPoints, Size maxFaceSize){
	float x = (allPoints.at(34).x + allPoints.at(44).x + allPoints.at(67).x) / 3;
	float y = (allPoints.at(34).y + allPoints.at(44).y + allPoints.at(67).y) / 3;
	Point2f thisCenter = Point2f(x, y);
	if (thisCenter.x < 0 || thisCenter.y < 0){
		Mat nullFrame(maxFaceSize.height, maxFaceSize.width, frame.type());
		nullFrame.copyTo(frame);
		return;
	}

	int rect_x = thisCenter.x - maxFaceSize.width / 2;
	int rect_y = thisCenter.y - maxFaceSize.height / 2;
	int rect_width = maxFaceSize.width;
	int rect_height = maxFaceSize.height;

	bool capable_displacement[2] = { true, true };

	if (rect_x + rect_width >= frame.cols){
		shiftImageAndPointsFromBorder(frame, allPoints, Point(-(rect_x + rect_width - frame.cols), 0));
		thisCenter.x += -(rect_x + rect_width - frame.cols);
		capable_displacement[0] = false;
	}
	if (rect_y + rect_height >= frame.rows){
		shiftImageAndPointsFromBorder(frame, allPoints, Point(0, -(rect_y + rect_height - frame.rows)));
		thisCenter.y += -(rect_y + rect_height - frame.rows);
		capable_displacement[1] = false;
	}
	if (rect_x <= 0 && capable_displacement[0])
	{
		shiftImageAndPointsFromBorder(frame, allPoints, Point(-rect_x, 0));
		thisCenter.x += -rect_x;
	}
	if (rect_y <= 0 && capable_displacement[1]){
		shiftImageAndPointsFromBorder(frame, allPoints, Point(0, -rect_y));
		thisCenter.y += -rect_y;
	}


	Mat stableFrame(frame.cols, frame.rows, frame.type());
	float angle = 0;
	float dy = (allPoints.at(44).y - allPoints.at(34).y);
	float dx = (allPoints.at(44).x - allPoints.at(34).x);

	if (dx) angle = atan(dy / dx);

	Mat rot_mat = getRotationMatrix2D(thisCenter, angle*57.3, 1);
	warpAffine(frame, stableFrame, rot_mat, frame.size());



	Rect region_of_interest = Rect(thisCenter.x - maxFaceSize.width / 2, thisCenter.y - maxFaceSize.height / 2, maxFaceSize.width, maxFaceSize.height);
	if (region_of_interest.x >= 0 && region_of_interest.y >= 0){


		Mat image_roi = stableFrame(region_of_interest);
		image_roi.copyTo(frame);

		for (unsigned int i = 0; i < allPoints.size(); i++){
			Point newP(allPoints.at(i).x - thisCenter.x, allPoints.at(i).y - thisCenter.y);
			allPoints.at(i).x = (newP.x * cos(-angle) - newP.y * sin(-angle)) + maxFaceSize.width / 2;
			allPoints.at(i).y = (newP.x * sin(-angle) + newP.y * cos(-angle)) + maxFaceSize.height / 2;
		}
	}
	else
	{
		Mat nullFrame(maxFaceSize.height, maxFaceSize.width, frame.type());
		for (unsigned int i = 0; i < allPoints.size(); i++)
			allPoints.at(i) = Point(-1, -1);
		nullFrame.copyTo(frame);
		return;
	}

}

//return max size for scaling result face
void calculationASM(Mat &frame, vector<Point> &prevFaceKeyPoints){

	Mat_<unsigned char> img;
	vector<Point> faceKeyPoints;

	cvtColor(frame, img, CV_RGB2GRAY);
	int foundface;
	float landmarks[2 * stasm_NLANDMARKS]; // x,y coords (note the 2)

	if (!stasm_search_single(&foundface, landmarks, (const char*)img.data, img.cols, img.rows, NULL, "./data"))
	{
		printf("Error in stasm_search_single: %s\n", stasm_lasterr());
		exit(1);
	}

	Point2f thisCenter(-1, -1);

	if (foundface)
	{
		// draw the landmarks on the image as white dots (image is monochrome)
		//stasm_force_points_into_image(landmarks, img.cols, img.rows);

		for (int i = 0; i < stasm_NLANDMARKS; i++){
			string text = to_string(i);
			Point facePoint(landmarks[i * 2], landmarks[i * 2 + 1]);
			faceKeyPoints.push_back(facePoint);
		}

		int width = pow(pow((faceKeyPoints.at(12).x - faceKeyPoints.at(0).x), 2) + pow((faceKeyPoints.at(12).y - faceKeyPoints.at(0).y), 2), 0.5);
		int height = pow(pow((faceKeyPoints.at(6).x - faceKeyPoints.at(14).x), 2) + pow((faceKeyPoints.at(6).y - faceKeyPoints.at(14).y), 2), 0.5);

		thisCenter = Point2f(
			(float)(faceKeyPoints.at(34).x + faceKeyPoints.at(44).x + faceKeyPoints.at(67).x) / 3.0f
			, (float)(faceKeyPoints.at(34).y + faceKeyPoints.at(44).y + faceKeyPoints.at(67).y) / 3.0f);


		//if (width >= faceFrameInfo.maxSize.width && height >= faceFrameInfo.maxSize.height)
		//	faceFrameInfo.maxSize = Size(width, height);

	}
	else
	{
		for (int i = 0; i < 77; i++)
			faceKeyPoints.push_back(Point(-1, -1));
	}
	//faceFrameInfo.thisCenter = thisCenter;

	//faceFrameInfo.maxSize.width += faceFrameInfo.maxSize.width / 4;
	//faceFrameInfo.maxSize.height += faceFrameInfo.maxSize.height / 4;

	prevFaceKeyPoints = faceKeyPoints;
}

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

void trackingFacePoints(const Mat& flow, Mat& frame, vector<Point> &faceKeyPoints, int step, const Scalar& color)
{
	for (unsigned int i = 0; i < faceKeyPoints.size(); i++){

		int x = faceKeyPoints.at(i).x / (step * 2);
		int y = faceKeyPoints.at(i).y / (step * 2);

		x *= step;
		y *= step;

		const Point2f& fxy = flow.at<Point2f>(y, x);

		Point p1(x, y);
		Point p2(round(x + fxy.x), round(y + fxy.y));

		line(frame, Point(p1.x * 2, p1.y * 2), Point(p2.x * 2, p2.y * 2), color);

		faceKeyPoints.at(i) = Point(p2.x * 2, p2.y * 2);

	}
}

void impositionOptFlow(Mat &frame, vector<Point> &faceKeyPoints, Mat &gray, Mat &prevgray){
	Mat flow, cflow;
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	resize(gray, gray, Size(frame.cols / 2, frame.rows / 2));

	if (prevgray.data)
	{
		calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 7, 10, 3, 7, 1.5, OPTFLOW_FARNEBACK_GAUSSIAN);
		//trackingFacePoints(flow, frame, faceKeyPoints, 2, CV_RGB(0, 0, 255));
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

void getTexture(Mat &frame, vector<Point> allPoints)
{
	vector<Point> countourPoints;
	int pointsOrder[18] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 25, 13, 14, 15, 18 };

	for (int cp = 0; cp < 18; cp++){
		countourPoints.push_back(allPoints.at(pointsOrder[cp]));
	}


	Point faceContourPoints[1][25];
	for (unsigned int i = 0; i < countourPoints.size(); i++)
		faceContourPoints[0][i] = countourPoints.at(i);

	faceContourPoints[0][19] = Point(0, 0);
	faceContourPoints[0][20] = Point(frame.cols, 0);
	faceContourPoints[0][21] = Point(frame.cols, frame.rows);
	faceContourPoints[0][22] = Point(0, frame.rows);
	faceContourPoints[0][23] = Point(0, 0);
	faceContourPoints[0][24] = countourPoints.at(17);

	const Point* ppt[1] = { faceContourPoints[0] };
	int npt[] = { 25 };

	fillPoly(frame, ppt, npt, 1, Scalar::all(0), 8);
}

void asm_stabilisation(Mat &frame, vector<Point> &asm_points){
	calculationASM(frame, asm_points); //ASM - face detection
	if (asm_points.at(0).x >= 0){
		int x = asm_points.at(0).x;
		int y = asm_points.at(14).y;
		int w = asm_points.at(12).x - asm_points.at(0).x;
		int h = asm_points.at(6).y - asm_points.at(14).y;
		x = x + w / 2 - h / 2;
		w = h;
		face_points_stabilisation(frame, asm_points, Size(w, h));

	}
}

void calculation_SFM(vector <Point2f> found_opfl_points, vector <Point2f> prev_opfl_points){
	double focal_length = 30;		//mm
	double pixel_length = 0.2646;	//mm

	for (unsigned int i = 0; i < found_opfl_points.size(); i++){

	}
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

	VideoWriter writer("./video_result.avi", 0, 15, frame.size(), true);
	if (!writer.isOpened())
	{
		printf("Output video could not be opened");
		return -1;
	}

	bool first_frame = false;
	
	while (1){
		if (waitKey(33) == 27)	break;
		cap >> frame;

		vector<float> error;

		if (first_frame){
			cvtColor(frame, gray, COLOR_BGR2GRAY);

			vector <Point2f> found_opfl_points;
			impositionOptFlowLK(prev_opfl_points, found_opfl_points, prevgray, gray, error, opf_parametrs);


			for (unsigned int i = 0; i < found_opfl_points.size(); i++){	
				if (error.at(i) == 0){
					circle(frame, found_opfl_points.at(i), 1, CV_RGB(128, 128, 255), 2, 8, 0);
					line(frame, prev_opfl_points.at(i), found_opfl_points.at(i), CV_RGB(64, 64, 128));
				}
			}
			imshow("frame", frame);
			goodFeaturesToTrack(prevgray, prev_opfl_points
				, fd_parametrs.max_ñorners, fd_parametrs.quality_level, fd_parametrs.min_distance, Mat(), fd_parametrs.block_size, 0, fd_parametrs.k);
		}
		else
		{
			cvtColor(frame, prevgray, COLOR_BGR2GRAY);
			prev_opfl_points = vector<Point2f>();
			goodFeaturesToTrack(prevgray, prev_opfl_points
				, fd_parametrs.max_ñorners, fd_parametrs.quality_level, fd_parametrs.min_distance, Mat(), fd_parametrs.block_size, 0, fd_parametrs.k);

			first_frame++;
		}
	}

	destroyAllWindows();

	return 0;
}