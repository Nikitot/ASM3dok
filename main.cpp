// minimal.cpp: Display the landmarks of a face in an image.
// This demonstrates stasm_search_single.

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

using namespace std;
using namespace cv;


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

void writeAndRotateImage(Mat &frame, bool warp, vector <Mat> &frames){
	Mat writingFrame(frame.cols,frame.rows, frame.type());

	if (warp){
		Point2f src_center(frame.rows/2, frame.rows/2);
		Mat rot_mat = getRotationMatrix2D(src_center, 270, 1);
		warpAffine(frame, writingFrame, rot_mat, writingFrame.size());
	}
	else{
		frame.copyTo(writingFrame);
	}

	frames.push_back(writingFrame);

	imshow("writing video", writingFrame);
}

void writeFramesFromVideo(vector <Mat> &frames, vector<int> frameNums,char* path, bool warp){

	VideoCapture cap(path);
	if (!cap.isOpened())  // check if we succeeded
		return;


	int frameIt = 0;
	while (1){
		Mat frame;
		cap >> frame;	

		if (frameNums.size() > 0){
			for (int i = 0; i < frameNums.size(); i++)
				if (frameNums.at(i) == frameIt)
					writeAndRotateImage(frame, warp, frames);
		}
		else
			writeAndRotateImage(frame, warp, frames);

		if (cvWaitKey(33) == 27 || !cap.read(frame)) break;
		frameIt++;
	}
	destroyAllWindows();
	cap.release();
	
}

int calculationASM(vector<Mat> &frames, vector<vector<Point>> &facesKeyPoints){
	Mat_<unsigned char> img = (imread("./FRONT.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	Mat imgc, img_grid;
	double maxDist[3] = { 0, 0, 0 };
	int numFrontalFace = -1;

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
		double dist[3] = { 0, 0, 0 };

		if (foundface)
		{
			// draw the landmarks on the image as white dots (image is monochrome)
			stasm_force_points_into_image(landmarks, img.cols, img.rows);

			for (int i = 0; i < stasm_NLANDMARKS; i++){
				string text = to_string(i);
				Point facePoint(landmarks[i * 2], landmarks[i * 2 + 1]);
				facePoints.push_back(facePoint);
			}
			printf("Calculating: %.0f%%, face found\n", percent);

			dist[0] = abs(facePoints.at(38).x - facePoints.at(39).x);
			dist[1] = abs(facePoints.at(38).x - facePoints.at(67).x);
			dist[2] = abs(facePoints.at(39).x - facePoints.at(67).x);

			if (dist[0] >= maxDist[0] && dist[1] >= maxDist[1] && dist[2] >= maxDist[2]){
				numFrontalFace = f;
				maxDist[0] = dist[0];
				maxDist[1] = dist[1];
				maxDist[2] = dist[2];
			}

		}
		else
		{
			for (int i = 0; i < 77; i++)
				facePoints.push_back(Point(-1, -1));
			printf("Calculating: %.0f %%, face not found\n", percent);
		}

		facesKeyPoints.push_back(facePoints);
	}
	return numFrontalFace;
}

void frameProcessing(Mat &frame, vector <Point> &keyPoints, Point center, int numFrame){
	for (int p = 0; p < keyPoints.size(); p++){
		circle(frame, keyPoints.at(p), 1, Scalar(255, 128, 128), 2);
	}

	line(frame, keyPoints.at(38), keyPoints.at(39), Scalar(0, 0, 255));
	line(frame, keyPoints.at(38), keyPoints.at(67), Scalar(0, 0, 255));
	line(frame, keyPoints.at(39), keyPoints.at(67), Scalar(0, 0, 255));

	line(frame, keyPoints.at(14), keyPoints.at(6), Scalar(255, 0, 0));
	line(frame, keyPoints.at(0), keyPoints.at(12), Scalar(255, 0, 0));

	Point crossLine = ((keyPoints.at(0).x + keyPoints.at(12).x) / 2, (keyPoints.at(14).y + keyPoints.at(6).y) / 2);

	circle(frame, crossLine, 1, Scalar(0,0,255),2);
	circle(frame, center, 1, Scalar(128, 255, 128), 2);

	stringstream text;
	text << numFrame;
	putText(frame, text.str(), Point(frame.cols / 20, frame.rows / 20), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(255), 3, 8);
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

	fillPoly(frame, ppt, npt, 1, Scalar(0, 0, 0), 8);
}

int main(int argc, char* argv[])
{
	vector<int> frameNums;
	if (argc > 1)
	{
		for (int i = 1; i < argc; i++)
			frameNums.push_back(atoi(argv[i]));
	}

	vector <Mat> frames;
	vector <vector <Point>> facesKeyPoints;

	writeFramesFromVideo(frames, frameNums, "C:/Users/Nikitot/Desktop/vc10/VID_20150115_205545.mp4",true);

	int numfrontalface = calculationASM(frames, facesKeyPoints);

	if (numfrontalface >= 0)
		imwrite("./FRONT.jpg", frames.at(numfrontalface));
	
	fstream resultCoords;
	resultCoords.open("./result.txt");

	VideoWriter writer("./video.avi", 0, 15, frames[0].size(), true);

	if (!writer.isOpened())
	{
		printf("Output video could not be opened");
		return -1;
	}

	for (int i = 0; i < facesKeyPoints.size(); i++)
	{		
		getTexture(frames[i], facesKeyPoints.at(i));
		float xMaxCenter = (float)(facesKeyPoints.at(i).at(34).x + facesKeyPoints.at(i).at(44).x + facesKeyPoints.at(i).at(67).x)/3.0f;
		float yMaxCenter = (float)(facesKeyPoints.at(i).at(34).y + facesKeyPoints.at(i).at(44).y + facesKeyPoints.at(i).at(67).y)/3.0f;

		Point2f thisCenter = Point2f(xMaxCenter, yMaxCenter);

		frameProcessing(frames[i], facesKeyPoints.at(i), thisCenter, i);
		

		if (frameNums.size() != 0)
		{			
			ostringstream fileName;
			fileName << "./frame" << frameNums.at(i) << ".jpg";
			imwrite(fileName.str(), frames[i]);

			resultCoords << "frame " << frameNums.at(i) << ":" << endl;
			resultCoords << "center :" << thisCenter.x << " " << thisCenter.y << endl;

			for (int p = 0; p < facesKeyPoints.at(i).size(); p++) {
				resultCoords << facesKeyPoints.at(i).at(p).x - thisCenter.x << " " << facesKeyPoints.at(i).at(p).y - thisCenter.y << endl;
			}
			resultCoords << endl;
		}		
		
		imshow("ASM", frames[i]);
		writer.write(frames[i]);

		if (waitKey(33)){}
		//_sleep(1000);
		
	}
	resultCoords.close();
	writer.release();

	return 0;
}