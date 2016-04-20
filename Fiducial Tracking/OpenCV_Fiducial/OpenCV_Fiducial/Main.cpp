/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.

License Agreement
For Open Source Computer Vision Library
(3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

* Neither the names of the copyright holders nor the names of the contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

void createFiducial();
void detectMarker(bool checkWrite);
void writeToFile(double mean, double stdDev);
void doStatistics(vector<double> rvecs, void(*wf)(double mean, double stdDev));

int main(int argc, char *argv[]) {

	int choice;
	bool flag = true;

	while (flag){
		cout << "1. Create fiducial marker" << endl;
		cout << "2. Detect fiducial marker" << endl;
		cout << "3. Detect fiducial and record statistics (std dev & mean) 50 samples" << endl;
		cout << "3. Exit the program      " << endl;
		cout << "Enter your choice: " << endl;
		cin >> choice;

		switch (choice){
		case 1:  createFiducial();
			break;
		case 2:
			detectMarker(false);
			break;
		case 3: 
			detectMarker(true);
			break;
		case 4:
			flag = false;
			cout << "Now exiting..." << endl;
			break;
		default:
			cout << "Bad Choice" << endl;
		}
	}


	return 0;
}

void createFiducial()
{
	auto dictionaryId = 10;
	auto markerId = 23;
	auto borderBits = 1;
	auto markerSize = 150;
	auto showImage = false;
	String out = "fiducial.png";

	Ptr<aruco::Dictionary> dictionary =
		aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

	Mat markerImg;
	aruco::drawMarker(dictionary, markerId, markerSize, markerImg, borderBits);

	if (showImage) {
		imshow("marker", markerImg);
		waitKey(0);
	}

	imwrite(out, markerImg);
}

static bool readCameraParameters(string filename, Mat &camMatrix, Mat &distCoeffs) {
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	fs["camera_matrix"] >> camMatrix;
	fs["distortion_coefficients"] >> distCoeffs;
	return true;
}

static bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters> &params) {
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
	fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
	fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
	fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
	fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
	fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
	fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
	fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
	fs["minDistanceToBorder"] >> params->minDistanceToBorder;
	fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
	fs["doCornerRefinement"] >> params->doCornerRefinement;
	fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
	fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
	fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
	fs["markerBorderBits"] >> params->markerBorderBits;
	fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
	fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
	fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
	fs["minOtsuStdDev"] >> params->minOtsuStdDev;
	fs["errorCorrectionRate"] >> params->errorCorrectionRate;
	return true;
}

void detectMarker(bool checkWrite)
{
	auto dictionaryId = 10;
	float markerLength = 0.04;

	auto detectorParams = aruco::DetectorParameters::create();

	detectorParams->doCornerRefinement = true; // do corner refinement in markers

	auto camId = 0;

	auto dictionary =
		aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

	Mat camMatrix, distCoeffs;
	auto readOk = readCameraParameters("camera.yaml", camMatrix, distCoeffs);
	if (!readOk) {
		cerr << "Invalid camera file" << endl;
	}

	VideoCapture inputVideo;
	inputVideo.open(camId);
	auto waitTime = 10;

	double totalTime = 0;
	auto totalIterations = 0;
	
	double theta;
	void (*wf)(double mean, double stdDev);
	wf = &writeToFile;
	vector<double> angles;

	while (inputVideo.grab()) {
		Mat image, imageCopy;
		inputVideo.retrieve(image);

		auto tick = double(getTickCount());

		vector< int > ids;
		vector< vector< Point2f > > corners, rejected;
		vector< Vec3d > rvecs, tvecs;

		// detect markers and estimate pose
		aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);
		if (ids.size() > 0)
			aruco::estimatePoseSingleMarkers(corners, markerLength, camMatrix, distCoeffs, rvecs,
			tvecs);

		// draw results
		image.copyTo(imageCopy);
		if (ids.size() > 0) {
			aruco::drawDetectedMarkers(imageCopy, corners, ids);

			for (unsigned int i = 0; i < ids.size(); i++) {
				aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rvecs[i], tvecs[i],
					markerLength * 0.5f);
				cout << "y: " << rvecs[i][2] * 57.2958 << " x :" << rvecs[i][0] * 57.2958 << " z: " << rvecs[i][1] * 57.2958 << endl;
				angles.push_back(rvecs[i][2]);
			}

			if (angles.size() == 50 && checkWrite == true)
			{
				doStatistics(angles, wf);
				break;
			}
		}

		imshow("out", imageCopy);
		auto key = char(waitKey(waitTime));
		if (key == 27) break;
	}
}

void doStatistics(vector<double> rvecs, void (*wf)(double mean, double stdDev))
{
	double mean = 0;
	double stdDev = 0;
	double stdDevCollect = 0;

	for (auto i = 0; i < 50; i++)
	{
		mean += rvecs[i] * 57.2958;
	}

	mean = mean / 50;

	for (auto i = 0; i < 50; i++)
	{
		stdDevCollect = pow(((rvecs[i] * 57.2958) - mean), 2);
		stdDev += stdDevCollect;
	}

	stdDev = sqrt(stdDev / 50);

	writeToFile(mean, stdDev);
}

void writeToFile(double mean, double stdDev)
{
	ofstream oFile;
	oFile.open("testData.txt", ios_base::app);

	oFile << "mean: " << mean << "  " << "std dev: " << stdDev << endl;

	oFile.close();
}