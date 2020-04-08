//copyright AZHAR TANWEER :-P

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    VideoCapture cap(0);
    Mat background;
    Mat frame;
    int p=0;
    // press ESC to break the video stream
    while(1){
        cap >> frame;
        Mat gray;
        cvtColor(frame, gray, COLOR_RGBA2GRAY);
        threshold(gray,gray,0,255,THRESH_BINARY_INV+THRESH_OTSU);


        Mat kernel = Mat::ones(3,3, CV_32F);
        Mat opening;
        morphologyEx(gray,gray,cv::MORPH_OPEN,kernel,Point(-1,-1),5);
        morphologyEx(gray,opening,cv::MORPH_DILATE,kernel,Point(-1,-1),1);
        dilate(opening,background,kernel,Point(-1,-1),5);

        Mat distrans;
        distanceTransform(gray,distrans,DIST_L2,5);
        normalize(distrans,distrans,1,0);
        Mat foreground;
        threshold(distrans,foreground,0.7,255,THRESH_BINARY);
        foreground.convertTo(foreground,CV_8U,1,0);

        Mat unknown,markers;
        subtract(background,foreground,unknown);
        connectedComponents(foreground,markers);

        for (int i = 0; i < markers.rows; i++) {
            for (int j = 0; j < markers.cols; j++) {
                markers.at<int>(i, j) = markers.at<uchar>(i, j) + 1;
                if (unknown.at<uchar>(i, j) == 255) {
                    markers.at<int>(i, j) = 0;
                }
            }
        }

        cvtColor(frame, frame, COLOR_RGBA2RGB, 0);
        watershed(frame, markers);

        for (int i = 0; i < markers.rows; i++) {
            for (int j = 0; j < markers.cols; j++) {
                if (markers.at<int>(i, j) == -1) {
                    markers.at<int>(i, j) = 255;
                }
            }
        }
        Mat mark;
        markers.convertTo(mark,CV_8UC1);

        namedWindow("tst",WINDOW_AUTOSIZE);
        imshow("tst",mark);

        Mat temp;
        temp = imread("template.png",CV_HAL_DFT_STAGE_COLS);

        Mat btemp,bimage,ctemp,cimage;

        GaussianBlur(temp,btemp,Size(3,3),1.4,1.4);
        Canny(btemp,ctemp,50,150);
        

        double d = matchShapes(mark,ctemp, CONTOURS_MATCH_I1,0);
        //cout << d << endl;
        if(d<0.01){p++;printf("BALLS!%d\n",p);}

        // can also try Hough transform:
        // vector<Vec3f>circles;
        // HoughCircles(mark,circles,HOUGH_GRADIENT,1,100);
        // if(!circles.empty())cout<<"BALLS!"<<endl;

        char c=(char)waitKey(25);
	    if(c==27)
	      break;
    }
    cap.release();
    destroyAllWindows();

    return 0;

}