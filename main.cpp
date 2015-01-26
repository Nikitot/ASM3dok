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

	Mat writingFrame(frame.cols, frame.rows, frame.type());
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

void writeFramesFromVideo(vector <Mat> &frames, vector<int> frameNums, char* path){
	int warp = 0;

	//VideoCapture cap(path);
	VideoCapture cap(0);
	if (!cap.isOpened())  // check if we succeeded
		return;

	int frameIt = 0;
	while (1){
		Mat frame;
		cap >> frame;
		if (cvWaitKey(33) == 27 || !cap.read(frame)) break;

		if (!frameNums.empty()){
			for (size_t i = 0; i < frameNums.size(); i++)
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

void facePointsStabilisation(Mat &frame, vector<Point> &allPoints, Size maxFaceSize, Point2f &thisCenter){
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

	//imshow("goodframe", stableFrame);
	//waitKey(0);


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
void calculationASM(vector<Mat> &frames, vector<vector<Point>> &facesKeyPoints, FaceFramesInfo &faceFrameInfo){
	Mat_<unsigned char> img;

	for (unsigned int f = 0; f < frames.size(); f++){
		cvtColor(frames[f], img, CV_RGB2GRAY);
		int foundface;
		float landmarks[2 * stasm_NLANDMARKS]; // x,y coords (note the 2)

		if (!stasm_search_single(&foundface, landmarks, (const char*)img.data, img.cols, img.rows, NULL, "./data"))
		{
			printf("Error in stasm_search_single: %s\n", stasm_lasterr());
			exit(1);
		}

		vector<Point> facePoints;
		double percent = ((double)f / (double)frames.size()) * 100;
		Point2f thisCenter(-1, -1);

		if (foundface)
		{
			// draw the landmarks on the image as white dots (image is monochrome)
			//stasm_force_points_into_image(landmarks, img.cols, img.rows);

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

	faceFrameInfo.maxSize.width += faceFrameInfo.maxSize.width / 4;
	faceFrameInfo.maxSize.height += faceFrameInfo.maxSize.height / 4;
}

void framePointsÑoloring(Mat &frame, vector <Point> &keyPoints, Point center, int numFrame){
	for (unsigned int p = 0; p < keyPoints.size(); p++){
		circle(frame, keyPoints.at(p), 1, Scalar(255, 128, 128), 2);
	}

	line(frame, keyPoints.at(34), keyPoints.at(44), Scalar(0, 0, 255));
	line(frame, keyPoints.at(34), keyPoints.at(67), Scalar(0, 0, 255));
	line(frame, keyPoints.at(44), keyPoints.at(67), Scalar(0, 0, 255));

	line(frame, keyPoints.at(14), keyPoints.at(6), Scalar(255, 0, 0));
	line(frame, keyPoints.at(0), keyPoints.at(12), Scalar(255, 0, 0));

	Point crossLine = ((keyPoints.at(0).x + keyPoints.at(12).x) / 2, (keyPoints.at(14).y + keyPoints.at(6).y) / 2);

	circle(frame, crossLine, 1, Scalar(0, 0, 255), 2);
	circle(frame, center, 1, Scalar(128, 255, 128), 2);

	stringstream text;
	text << numFrame;
	//putText(frame, text.str(), Point(frame.cols / 15, frame.rows / 15), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(255), 3, 8);
}

void drawOptFlowMap(const Mat& flow, Mat& frame, vector<Point> allPoints, int step, double scale, const Scalar& color)
{
	for (unsigned int i = 0; i < allPoints.size(); i++){

		//float d = (float) allPoints.at(i).x / step

		int x = allPoints.at(i).x / (step * 2); 
		int y = allPoints.at(i).y / (step * 2);

		x *= step;
		y *= step;


		const Point2f& fxy = flow.at<Point2f>(y, x);

		Point p1(x, y);
		Point p2(round(x + fxy.x), round(y + fxy.y));

		line(frame, Point(p1.x * 2, p1.y * 2), Point(p2.x * 2, p2.y * 2), color);
	}

	/*for (int y = 0; y < flow.rows/step; y++ ){
		for (int x = 0; x < flow.cols/step; x++ )
		{
			const Point2f& fxy = flow.at<Point2f>(y*step, x*step);

			Point p1(x*step, y*step);
			Point p2(round(x*step + fxy.x), round(y*step + fxy.y));

			line(frame, Point(p1.x * 2, p1.y * 2), Point(p2.x * 2, p2.y * 2), color);
		}
	}*/
}

void impositionOptFlow(Mat &frame, vector<Point> allPoints, Mat &prevgray, Mat &gray){
	Mat flow, cflow;
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	resize(gray, gray, Size(frame.cols / 2, frame.rows / 2));

	if (prevgray.data)
	{
		calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
		drawOptFlowMap(flow, frame, allPoints, 2, 1.5, CV_RGB(0, 0, 255));
	}
	swap(prevgray, gray);
}

void impositionOptFlowLK(Mat &frame, vector<Point> old_features, Mat &prevgray, Mat &gray, int cornersCount){
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	vector<uchar> status;
	vector<float> error;
	vector<Point2f> found;
	vector<Point2f> frameFeatures;
	Size subPixWinSize(10, 10);
	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, cornersCount, 0.03);

	if (prevgray.data)
	{
		goodFeaturesToTrack(gray, frameFeatures, cornersCount, 0.001, 20);
		cornerSubPix(gray, frameFeatures, subPixWinSize, Size(-1, -1), termcrit);
		calcOpticalFlowPyrLK(prevgray, gray, frameFeatures, found, status, error);

		for (unsigned int i = 0; i < found.size(); i++){
			circle(frame, found.at(i), 1, CV_RGB(0, 255, 0), 3, 8, 0);
		}
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

	writeFramesFromVideo(frames, frameNums, "./Dim.mp4");

	calculationASM(frames, facesKeyPoints, faceFrameInfo);

	fstream resultCoords("./result.txt");
	VideoWriter writer("./video_result.avi", 0, 15, faceFrameInfo.maxSize, true);

	if (!writer.isOpened())
	{
		printf("Output video could not be opened");
		return -1;
	}

	Mat prevgray, gray;

	for (unsigned int i = 0; i < facesKeyPoints.size(); i++)
	{

		facePointsStabilisation(frames[i], facesKeyPoints.at(i), faceFrameInfo.maxSize, faceFrameInfo.thisCenter[i]);

		if (i>1 && facesKeyPoints.at(i - 1).at(0).x > 0){
			impositionOptFlow(frames[i], facesKeyPoints.at(i - 1), prevgray, gray);
		}

		framePointsÑoloring(frames[i], facesKeyPoints.at(i), faceFrameInfo.thisCenter[i], i);
		//getTexture(frames[i], facesKeyPoints.at(i));

		if (frameNums.size() != 0)
		{
			ostringstream fileName;
			fileName << "./frame" << frameNums.at(i) << ".jpg";
			imwrite(fileName.str(), frames[i]);

			resultCoords << "frame " << frameNums.at(i) << ":" << endl;
			resultCoords << "center :" << faceFrameInfo.maxSize.width / 2 << " " << faceFrameInfo.maxSize.height / 2 << endl;

			for (int p = 0; p < facesKeyPoints.at(i).size(); p++)
			{
				
				//if ( p == 34 || p == 44 || p == 52 || p == 59 || p == 65 || p == 1 || p == 11)
					resultCoords << facesKeyPoints.at(i).at(p).x << " " << facesKeyPoints.at(i).at(p).y << endl;
			}
			resultCoords << endl;
		}

		imshow("ASM result", frames[i]);
		writer.write(frames[i]);

		if (waitKey(33)){}
	}

	return 0;
}