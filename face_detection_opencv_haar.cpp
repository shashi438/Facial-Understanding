#include "stdafx.h"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;
ofstream myfile;

/** Global variables */
String faceCascadePath;
CascadeClassifier faceCascade;

void detectFaceOpenCVHaar(CascadeClassifier faceCascade, Mat &frameOpenCVHaar, string file1 ,int inHeight=300, int inWidth=0)
{
    int frameHeight = frameOpenCVHaar.rows;
    int frameWidth = frameOpenCVHaar.cols;
    if (!inWidth)
        inWidth = (int)((frameWidth / (float)frameHeight) * inHeight);

    float scaleHeight = frameHeight / (float)inHeight;
    float scaleWidth = frameWidth / (float)inWidth;

    Mat frameOpenCVHaarSmall, frameGray;
    resize(frameOpenCVHaar, frameOpenCVHaarSmall, Size(inWidth, inHeight));
    cvtColor(frameOpenCVHaarSmall, frameGray, COLOR_BGR2GRAY);

    std::vector<Rect> faces;
    faceCascade.detectMultiScale(frameGray, faces);

    for ( size_t i = 0; i < faces.size(); i++ )
    {
      int x1 = (int)(faces[i].x * scaleWidth);
      int y1 = (int)(faces[i].y * scaleHeight);
      int x2 = (int)((faces[i].x + faces[i].width) * scaleWidth);
      int y2 = (int)((faces[i].y + faces[i].height) * scaleHeight);
      rectangle(frameOpenCVHaar, Point(x1, y1), Point(x2, y2), Scalar(0,255,0), (int)(frameHeight/150.0), 4);
	  cout << x1 << "," << y1 << "," << x2 << "," << y2 << endl;
	  myfile << file1 << "," << x1 << "," << y1 << "," << x2 << "," << y2 << std::endl;
	}
}


int main( int argc, const char** argv )
{
  faceCascadePath = "./haarcascade_frontalface_default.xml";

  if( !faceCascade.load( faceCascadePath ) ){ printf("--(!)Error loading face cascade\n"); return -1; };

  // VideoCapture source;
  // if (argc == 1)
  //     source.open(0);
  // else
  //     source.open(argv[1]);
  std::vector<String> filesnames;
  String folder = "./Images_modified/Images_modified";
  glob(folder, filesnames);
  myfile.open("./opencv_haar.csv");
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



	  double tt_opencvHaar = 0;
	  double fpsOpencvHaar = 0;
	  int inHeight = 300;
	  int inWidth = 0;
	  // while(1)
	  // {
		  // source >> frame;
		  // if(frame.empty())
		  //     break;
		  // double t = cv::getTickCount();
	  detectFaceOpenCVHaar(faceCascade, frame, filesnames[i], inHeight, inWidth);
	  // tt_opencvHaar = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
	  // fpsOpencvHaar = 1/tt_opencvHaar;
	  // putText(frame, format("OpenCV HAAR ; FPS = %.2f",fpsOpencvHaar), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.4, Scalar(0, 0, 255), 4);
	//  imshow( "OpenCV - HAAR Face Detection", frame );
	  int k = waitKey(5);
	  // if(k == 27)
	  // {
	  //   destroyAllWindows();
	  //   break;
	  // }
	// }
  }
  cout << "i is " << q << std::endl;
  myfile.close();
}
