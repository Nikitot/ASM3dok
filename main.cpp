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

struct FaceFramesInfo
{
	Size maxSize = Size(0, 0);
	int anfasFrame = -1;
	vector<Point2f> thisCenter;	
};

//Old triangulation
void draw_subdiv(Mat &img, Subdiv2D& subdiv, Scalar delaunay_color)
{
	int rows = img.rows;
	int cols = img.cols;

	cv::Size s = img.size();
	rows = s.height;
	cols = s.width;

	bool draw;
	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	vector<Point> pt(3);

	for (size_t i = 0; i < triangleList.size(); ++i)
	{
		Vec6f t = triangleList[i];

		pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
		pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
		pt[2] = Point(cvRound(t[4]), cvRound(t[5]));

		draw = true;

		for (int i = 0; i < 3; i++){
			//cout << pt[i].x << " " << pt[i].y << endl;
			if (pt[i].x >= cols || pt[i].y >= rows || pt[i].x <= 0 || pt[i].y <= 0)
				draw = false;
		}
		if (draw){
			line(img, pt[0], pt[1], delaunay_color, 1);
			line(img, pt[1], pt[2], delaunay_color, 1);
			line(img, pt[2], pt[0], delaunay_color, 1);
		}

	}
} 

void writeAndRotateImage(Mat &frame, int angle, vector <Mat> &frames){
	
	Mat writingFrame(frame.cols,frame.rows, frame.type());
	if (angle){
		Point2f src_center;
		if (angle == 90)
			src_center = Point(frame.cols / 2, frame.cols / 2);
		else if (angle == 270)
			src_center = Point(frame.rows / 2, frame.rows / 2);
		else
			src_center = Point(frame.cols / 2, frame.rows / 2);

		Mat rot_mat = getRotationMatrix2D(src_center, angle, 1);
		warpAffine(frame, writingFrame, rot_mat, writingFrame.size());
	}
	else{
		frame.copyTo(writingFrame);
	}
	frames.push_back(writingFrame);

	imshow("writing video", writingFrame);
}

void writeFramesFromVideo(vector <Mat> &frames, vector<int> frameNums,char* path, int warp){

	VideoCapture cap(path);
	//VideoCapture cap(0);
	if (!cap.isOpened())  // check if we succeeded
		return;

	int frameIt = 0;
	while (1){
		Mat frame;
		cap >> frame;	
		if (cvWaitKey(33) == 27 || !cap.read(frame)) break;

		if (frameNums.size() > 0){
			for (int i = 0; i < frameNums.size(); i++)
				if (frameNums.at(i) == frameIt)
					writeAndRotateImage(frame, warp, frames);
		}
		else{
				writeAndRotateImage(frame, warp, frames);		
			}
		frameIt++;
	}
	destroyAllWindows();
	cap.release();
	
}

void facePointsStabilisation(Mat &frame, vector<Point> &allPoints, Size maxFaceSize, Point thisCenter){
	//doesn not work if the ROI is beyond the border of the max size
	
	if (thisCenter.x < 0 || thisCenter.y < 0 ){
		Mat nullFrame(maxFaceSize.height, maxFaceSize.width, frame.type());
		nullFrame.copyTo(frame);
		return;
	}
	Mat stableFrame(frame.cols, frame.rows, frame.type());
	float angle = 0;
	float dy = (allPoints.at(44).y - allPoints.at(34).y);
	float dx = (allPoints.at(44).x - allPoints.at(34).x);

	if (dx)
		angle = atan(dy / dx);

	Mat rot_mat = getRotationMatrix2D(thisCenter, angle*57.3, 1);
	warpAffine(frame, stableFrame, rot_mat, frame.size());	
		
	try{
		Rect region_of_interest = Rect(thisCenter.x - maxFaceSize.width / 2, thisCenter.y - maxFaceSize.height / 2, maxFaceSize.width, maxFaceSize.height);
		Mat image_roi = stableFrame(region_of_interest);
		image_roi.copyTo(frame);
	}
	catch(...){
		Mat nullFrame(maxFaceSize.height, maxFaceSize.width, frame.type());
		nullFrame.copyTo(frame);
		return;
	}
	for (int i = 0; i < allPoints.size(); i++){
		Point newP(allPoints.at(i).x - thisCenter.x, allPoints.at(i).y - thisCenter.y);
		allPoints.at(i).x = (newP.x * cos(-angle) - newP.y * sin(-angle)) + maxFaceSize.width / 2;
		allPoints.at(i).y = (newP.x * sin(-angle) + newP.y * cos(-angle)) + maxFaceSize.height / 2;
	}
}

//return max size for scaling result face
void calculationASM(vector<Mat> &frames, vector<vector<Point>> &facesKeyPoints, FaceFramesInfo &faceFrameInfo){
	Mat_<unsigned char> img = (imread("./FRONT.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	Mat img_grid;

	for (int f = 0; f < frames.size(); f++){
		img_grid = Mat(img.rows, img.cols, CV_LOAD_IMAGE_GRAYSCALE);
		cvtColor(frames[f], img, CV_RGB2GRAY);
		
		int foundface;
		float landmarks[2 * stasm_NLANDMARKS]; // x,y coords (note the 2)

		if (!stasm_search_single(&foundface, landmarks, (const char*)img.data, img.cols, img.rows, "./FRONT.jpg", "./data"))
		{
			printf("Error in stasm_search_single: %s\n", stasm_lasterr());
			exit(1);
		}
		vector<Point> facePoints;
		double percent = ((double)f / (double)frames.size()) * 100;
		Point2f thisCenter(-1,-1);

		if (foundface)
		{
			// draw the landmarks on the image as white dots (image is monochrome)
			stasm_force_points_into_image(landmarks, img.cols, img.rows);

			for (int i = 0; i < stasm_NLANDMARKS; i++){
				string text = to_string(i);
				Point facePoint(landmarks[i * 2], landmarks[i * 2 + 1]);
				facePoints.push_back(facePoint);
			}

			int width = pow(pow((facePoints.at(12).x - facePoints.at(0).x), 2) + pow((facePoints.at(12).y - facePoints.at(0).y), 2), 0.5);
			int height = pow(pow((facePoints.at(6).x - facePoints.at(14).x), 2) + pow((facePoints.at(6).y - facePoints.at(14).y), 2), 0.5);

			thisCenter = Point2f(
				(float)(facePoints.at(34).x + facePoints.at(44).x + facePoints.at(67).x) / 3.0f
				, (float)(facePoints.at(34).y + facePoints.at(44).y + facePoints.at(67).y) / 3.0f);


			if (width >= faceFrameInfo.maxSize.width && height >= faceFrameInfo.maxSize.height)
				faceFrameInfo.maxSize = Size(width, height);
			printf("Calculating: %.0f%%, face found\n", percent);

		}
		else
		{
			for (int i = 0; i < 77; i++)
				facePoints.push_back(Point(-1, -1));
			printf("Calculating: %.0f %%, face not found\n", percent);
		}
		facesKeyPoints.push_back(facePoints);
		faceFrameInfo.thisCenter.push_back(thisCenter);
	}
	
	faceFrameInfo.maxSize.width += faceFrameInfo.maxSize.width / 20;
}

void framePointsÑoloring(Mat &frame, vector <Point> &keyPoints, Point center, int numFrame){
	for (int p = 0; p < keyPoints.size(); p++){
		circle(frame, keyPoints.at(p), 1, Scalar(255, 128, 128), 2);
	}

	line(frame, keyPoints.at(34), keyPoints.at(44), Scalar(0, 0, 255));
	line(frame, keyPoints.at(34), keyPoints.at(67), Scalar(0, 0, 255));
	line(frame, keyPoints.at(44), keyPoints.at(67), Scalar(0, 0, 255));

	line(frame, keyPoints.at(14), keyPoints.at(6), Scalar(255, 0, 0));
	line(frame, keyPoints.at(0), keyPoints.at(12), Scalar(255, 0, 0));

	Point crossLine = ((keyPoints.at(0).x + keyPoints.at(12).x) / 2, (keyPoints.at(14).y + keyPoints.at(6).y) / 2);

	circle(frame, crossLine, 1, Scalar(0,0,255),2);
	circle(frame, center, 1, Scalar(128, 255, 128), 2);

	stringstream text;
	text << numFrame;
	putText(frame, text.str(), Point(frame.cols / 15, frame.rows / 15), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(255), 3, 8);
}

void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, double scale, const Scalar& color)
{
	double xx = 0, yy = 0;
	int x, y;
	for (y = 0; y < cflowmap.rows; y += step)
	for (x = 0; x < cflowmap.cols; x += step)
	{
		const Point2f& fxy = flow.at<Point2f>(y, x);

		Point p1(x, y);
		Point p2(cvRound(x + fxy.x), cvRound(y + fxy.y));

		line(cflowmap, p1, p2,	color);
		
		//if (trackPoint.x >= p1.x && trackPoint.y >= p1.y && trackPoint.x < p1.x + 6 && trackPoint.y < p1.y + 6)
		//{
		//	trackPoint = p2;
		//	line(cflowmap, p1, trackPoint, CV_RGB(255, 0, 0));
		//	circle(cflowmap, trackPoint, 3, CV_RGB(255, 0, 0));
		//}

	}
	xx /= x*y;
	yy /= x*y;
}

void impositionOptFlow(Mat &frame, Mat &prevgray){
	Mat gray, flow, cflow;
	cvtColor(frame, gray, CV_BGR2GRAY);
	resize(gray, gray, Size(frame.cols / 2, frame.rows / 2));

	if (prevgray.data)
	{
		calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
		cvtColor(prevgray, cflow, CV_GRAY2BGR);
		drawOptFlowMap(flow, cflow, 15, 1.5, CV_RGB(0, 0, 255));
		imshow("flow", cflow);
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
	for (int i = 0; i < countourPoints.size(); i++)
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

int main(int argc, char* argv[])
{
	vector <int> frameNums;
	vector <Mat> frames;
	vector <vector <Point>> facesKeyPoints;
	FaceFramesInfo faceFrameInfo;

	if (argc > 1)
	{
		for (int i = 1; i < argc; i++)
			frameNums.push_back(atoi(argv[i]));
	}

	writeFramesFromVideo(frames, frameNums, "./Dim.mp4", 0);
	calculationASM(frames, facesKeyPoints, faceFrameInfo);


	fstream resultCoords("./result.txt");
	VideoWriter writer("./video.avi", 0, 15, faceFrameInfo.maxSize, true);

	if (!writer.isOpened())
	{
		printf("Output video could not be opened");
		return -1;
	}

	for (int i = 0; i < facesKeyPoints.size(); i++)
	{		
		
		framePointsÑoloring(frames[i], facesKeyPoints.at(i), faceFrameInfo.thisCenter[i], i);
		facePointsStabilisation(frames[i], facesKeyPoints.at(i), faceFrameInfo.maxSize, faceFrameInfo.thisCenter[i]);
		//getTexture(frames[i], facesKeyPoints.at(i));

		if (frameNums.size() != 0)
		{			
			ostringstream fileName;
			fileName << "./frame" << frameNums.at(i) << ".jpg";
			imwrite(fileName.str(), frames[i]);

			resultCoords << "frame " << frameNums.at(i) << ":" << endl;
			resultCoords << "center :" << faceFrameInfo.thisCenter[i].x << " " << faceFrameInfo.thisCenter[i].y << endl;

			for (int p = 0; p < facesKeyPoints.at(i).size(); p++) 
			{
				resultCoords << facesKeyPoints.at(i).at(p).x - faceFrameInfo.thisCenter[i].x << " " << facesKeyPoints.at(i).at(p).y - faceFrameInfo.thisCenter[i].y << endl;
			}
			resultCoords << endl;
		}	
				
		imshow("ASM result", frames[i]);
		writer.write(frames[i]);

		if (waitKey(33)){}	
	}

	
	/*
	
	for fan :)
	
	*/
	
	for (int i = frames.size() - 1; i >= 0; i--)
	{
		writer.write(frames[i]);
	}
	resultCoords.close();
	writer.release();
	

	return 0;
}