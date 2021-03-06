#include "stdafx.h"
#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#define DLIB_JPEG_SUPPORT


using namespace cv;
using namespace std;
using namespace dlib;
ofstream myfile;

// Network Definition
/////////////////////////////////////////////////////////////////////////////////////////////////////
template <long num_filters, typename SUBNET> using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;

template <typename SUBNET> using downsampler = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5 = relu<affine<con5<45, SUBNET>>>;

using net_type = loss_mmod<con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;
/////////////////////////////////////////////////////////////////////////////////////////////////////

void detectFaceDlibMMOD(net_type mmodFaceDetector, Mat &frameDlibMmod, string file1, int inHeight = 300, int inWidth = 0)
{

	int frameHeight = frameDlibMmod.rows;
	int frameWidth = frameDlibMmod.cols;
	if (!inWidth)
		inWidth = (int)((frameWidth / (float)frameHeight) * inHeight);

	float scaleHeight = frameHeight / (float)inHeight;
	float scaleWidth = frameWidth / (float)inWidth;

	Mat frameDlibMmodSmall;
	resize(frameDlibMmod, frameDlibMmodSmall, Size(inWidth, inHeight));

	// Convert OpenCV image format to Dlib's image format
	cv_image<bgr_pixel> dlibIm(frameDlibMmodSmall);
	matrix<rgb_pixel> dlibMatrix;
	assign_image(dlibMatrix, dlibIm);

	// Detect faces in the image
	std::vector<dlib::mmod_rect> faceRects = mmodFaceDetector(dlibMatrix);

	for (size_t i = 0; i < faceRects.size(); i++)
	{
		int x1 = (int)(faceRects[i].rect.left() * scaleWidth);
		int y1 = (int)(faceRects[i].rect.top() * scaleHeight);
		int x2 = (int)(faceRects[i].rect.right() * scaleWidth);
		int y2 = (int)(faceRects[i].rect.bottom() * scaleHeight);
		cv::rectangle(frameDlibMmod, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), (int)(frameHeight / 150.0), 4);
		cout << x1<<","<<y1<<","<<x2<<","<<y2 << endl;
		myfile << file1 << "," << x1 << "," << y1 << "," << x2 << "," << y2 << std::endl;
	}
}

int main(int argc, const char** argv)
{
	String mmodModelPath = "./mmod_human_face_detector1.dat";
	net_type mmodFaceDetector;
	deserialize(mmodModelPath) >> mmodFaceDetector;



	std::vector<String> filesnames;
	String folder = "./FDDB/FDDB/";
	glob(folder, filesnames);
	myfile.open("./dlib_dnn_FDDB.csv");
	int q = 0;
	for (size_t i = 0; i < filesnames.size(); i++)
	{
		// VideoCapture source;
		// if (argc == 1)
		//     source.open(0);
		// else
		//     source.open(argv[1]);
		q = q + 1;
		cout << "You can get it from the following URL: " << endl;
		Mat frame = imread(filesnames[i]);
		if (frame.empty())
		{
			std::cerr << "Can't read image from the file" << filesnames[i] << std::endl;
		}

		double tt_dlibMmod = 0;
		double fpsDlibMmod = 0;
		int inHeight = 300;
		int inWidth = 0;
		// while(1)
		// {
		//source >> frame;
		// if(frame.empty())
		//     break;

		// double t = cv::getTickCount();
		detectFaceDlibMMOD(mmodFaceDetector, frame, filesnames[i], inHeight,inWidth );
		// tt_dlibMmod = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
		// fpsDlibMmod = 1/tt_dlibMmod;
		// putText(frame, format("DLIB MMOD ; FPS = %.2f",fpsDlibMmod), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.4, Scalar(0, 0, 255), 4);
		//imshow("DLIB - MMOD Face Detection", frame);
		int k = waitKey(0);

	}
	cout << "i is " << q << std::endl;
	myfile.close();
	// if(k == 27)
	// {
	//   destroyAllWindows();
	//   break;
	// }
	  // }
}

