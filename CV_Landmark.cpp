#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
//#include "drawLandmarks.hpp"


using namespace std;
using namespace cv;
using namespace cv::face;
using namespace cv::dnn;
ofstream myfile;
ifstream inFile;
///////////////////////////////////////////////////////////////////////
const size_t inWidth_or = 300;
const size_t inHeight_or = 300;
const double inScaleFactor = 1.0;
const float confidenceThreshold = 0.7;
const cv::Scalar meanVal(104.0, 177.0, 123.0);

#define CAFFE

const std::string caffeConfigFile = "deploy.prototxt";
const std::string caffeWeightFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel";

const std::string tensorflowConfigFile = "opencv_face_detector.pbtxt";
const std::string tensorflowWeightFile = "opencv_face_detector_uint8.pb";

vector<Rect> detectFaceOpenCVDNN(Net net, Mat &frameOpenCVDNN, string file1, int inWidth, int inHeight)
{
	int frameHeight = frameOpenCVDNN.rows;
	int frameWidth = frameOpenCVDNN.cols;
	std::vector<Rect> faces;
#ifdef CAFFE
	cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth_or, inHeight_or), meanVal, false, false);
#else
	cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, true, false);
#endif
	net.setInput(inputBlob, "data");
	cv::Mat detection = net.forward("detection_out");

	cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	int q = 0;
	cout << "detection mat rows is " << detectionMat.rows << std::endl;
	for (int i = 0; i < detectionMat.rows; i++)
	{
		q = q + 1;
		float confidence = detectionMat.at<float>(i, 2);

		if (confidence > confidenceThreshold)
		{
			int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
			int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
			int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
			int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);
			myfile << file1 << "," << x1 << "," << y1 << "," << x2 << "," << y2 << std::endl;
			//myfile << x1, y1, x2, y2;
			cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2, 4);
			int width = x2 - x1;
			int height = y2 - y1;
			cv::Rect inp = Rect(x1, y1, width, height);
			faces.push_back(inp);


		}

		//cout << "i inside loop is" << q << std::endl;
	}
	return faces;

}


///////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	// Load Face Detector
	//CascadeClassifier faceDetector;
	//string path_file = "haarcascade_frontalface_default.xml";
	//faceDetector.load(path_file);
	//if (!faceDetector.load(path_file)) { printf("--(!)Error loading\n"); return -1; };
	// Create an instance of Facemark
	//Ptr<Facemark> facemark = FacemarkLBF::create();
	double tt_opencvDNN = 0;
	double fpsOpencvDNN = 0;
	///////////load deep learning model
	Net net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);

	Ptr<Facemark> facemark = createFacemarkLBF();
	// Load landmark detectors
	facemark->loadModel("lbfmodel.yaml");

	// Set up webcam for video capture
	//VideoCapture cam(0);

	// Variable to store a video frame and its grayscale 
	Mat frame, gray;
	String delimiter = ".jpg";
	String token,curr_file;
	vector<String> filenames;
	String folder = "./faces/";
	String to_open;
	glob(folder, filenames);
	int x,x1,y1,width,height,c_x,c_y;
	cv::Rect inp;
	vector<int> values;

	for (size_t i = 0; i < filenames.size(); i++)
	{
		frame = imread(filenames[i]);
		cv::Size sz = frame.size();
		int imageWidth = sz.width;
		int imageHeight = sz.height;
		if (frame.empty())
		{
			std::cerr << "Can't read image from the file: " << filenames[i] << std::endl;
			exit(-1);
		}
		cout << "processing image " << filenames[i] << endl;

		//imshow("read_image", frame);
		//waitKey(0);
		// Read a frame

			// Find face
		vector<Rect> faces;
		// Convert frame to grayscale because
		// faceDetector requires grayscale image.
		//cvtColor(frame, gray, COLOR_BGR2GRAY);
		//resize(frame, frame_1, Size(300, 300));

		//faces = detectFaceOpenCVDNN(net, frame, filenames[i],imageWidth,imageHeight);
		
		//here faces are x,y, width and height. read from text files
		//faces[0],faces[1],etc.
		curr_file = filenames[i].substr(0, filenames[i].find(delimiter));
		to_open = "./boxes\\" + curr_file.substr(8) + ".txt";
		inFile.open(to_open);

		if (!inFile) {
			cerr << "Unable to open file datafile.txt";
			exit(1);   // call system to stop
		}

		while (inFile >> x) {
			values.push_back(x); //values is vector of values read
		}
		//in this for loop we are writing first 4 values to a Rect and then
		// to faces using pushback. note you have to use Rect before pushback
		/*
		for (unsigned jj = 0; jj < values.size()/4; jj++) {
			if (!(jj == 0 || jj % 4 == 0)) {
				cout << "i is " << jj << "and is skipped" << std::endl;
				continue;
			}
			else {
				cout << "i is " << jj << std::endl;
				c_x = values.at(jj);
				c_y = values.at(jj + 1);
				width = values.at(jj + 2);
				height = values.at(jj + 3);
				x1 = c_x - (width / 2);
				y1 = c_y + (height / 2);
				inp = Rect(x1, y1, width, height);
				
				Point pt1(x1, y1);
				Point pt2(x1 + width, y1 + height);
				rectangle(frame, pt1,pt2, cv::Scalar(0, 255, 0));
				imwrite("drawn.jpg", frame);
				break;
				
				faces.push_back(inp);
				c_x = 0, c_y = 0; x1 = 0; y1 = 0; height = 0; width = 0;
			}
		}
		*/
		c_x = values.at(0);
		c_y = values.at(1);
		width = values.at(2);
		height = values.at(3);
		x1 = c_x - (width / 2);
		y1 = c_y + (height / 2);
		inp = Rect(x1, y1, width, height);
		faces.push_back(inp);
		c_x = 0, c_y = 0; x1 = 0; y1 = 0; height = 0; width = 0;

		inFile.close();
		//cvtColor(frame, gray, COLOR_BGR2GRAY);

		// Detect faces
		//faceDetector.detectMultiScale(gray, faces);

		// Variable for landmarks. 
		// Landmarks for one face is a vector of points
		// There can be more than one face in the image. Hence, we 
		// use a vector of vector of points. 
		vector< vector<Point2f> > landmarks;

		// Run landmark detector
		bool success;
		try {
			success = facemark->fit(frame, faces, landmarks);
		}
		catch (exception& e) {
			cout << "\nexception thrown" << endl;
			cout << e.what() << endl;
		}

		if (success)
		{
			token = filenames[i].substr(0, filenames[i].find(delimiter));
			myfile.open(token + ".csv");
			// If successful, render the landmarks on the face
			for (int lm = 0; lm < landmarks.size(); lm++)
			{
				int dets = landmarks.size() / 68;
				//cout << landmarks[lm] << std::endl;
				//myfile << filenames[i] << "," << landmarks[lm] << std::endl;
				myfile << landmarks[lm] << std::endl;

				//drawLandmarks(frame, landmarks[lm]);
			}

			myfile.close();

		}
		else {
			cout << "unsuccesfull" << std::endl;
		}

		// Display results 
		//imshow("Facial Landmark Detection", frame);
		//waitKey(0);

		// Exit loop if ESC is pressed
	}
	return 0;
}