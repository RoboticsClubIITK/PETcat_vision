#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace std;
using namespace cv;
int main(int argc, char *argv[])
{
    VideoCapture cap(0);
    Mat src;
    int p=0;
    while(1){
        cap >> src;
        imshow("Source stream",src);
        // Change the background from white to black, since that will help later to extract
        // better results during the use of Distance Transform
        for ( int i = 0; i < src.rows; i++ ) {
            for ( int j = 0; j < src.cols; j++ ) {
                if ( src.at<Vec3b>(i, j) == Vec3b(255,255,255) )
                {
                    src.at<Vec3b>(i, j)[0] = 0;
                    src.at<Vec3b>(i, j)[1] = 0;
                    src.at<Vec3b>(i, j)[2] = 0;
                }
            }
        }
        // Show output image
        // namedWindow("Black Background Image",WINDOW_AUTOSIZE);
        // imshow("Black Background Image", src);
        // Create a kernel that we will use to sharpen our image
        Mat kernel = (Mat_<float>(3,3) <<
                    1,  1, 1,
                    1, -8, 1,
                    1,  1, 1); // an approximation of second derivative, a quite strong kernel
        // do the laplacian filtering as it is
        // well, we need to convert everything in something more deeper then CV_8U
        // because the kernel has some negative values,
        // and we can expect in general to have a Laplacian image with negative values
        // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
        // so the possible negative number will be truncated
        Mat imgLaplacian;
        filter2D(src, imgLaplacian, CV_32F, kernel);
        Mat sharp;
        src.convertTo(sharp, CV_32F);
        Mat imgResult = sharp - imgLaplacian;
        // convert back to 8bits gray scale
        imgResult.convertTo(imgResult, CV_8UC3);
        imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
        // imshow( "Laplace Filtered Image", imgLaplacian );
        // imshow( "New Sharped Image", imgResult );
        // Create binary image from source image
        Mat bw;
        cvtColor(imgResult, bw, COLOR_BGR2GRAY);
        threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);
        imshow("Binary Image", bw);
        // Perform the distance transform algorithm
        Mat dist;
        distanceTransform(bw, dist, DIST_L2, 3);
        // Normalize the distance image for range = {0.0, 1.0}
        // so we can visualize and threshold it
        normalize(dist, dist, 0, 1.0, NORM_MINMAX);
        //imshow("Distance Transform Image", dist);
        // Threshold to obtain the peaks
        // This will be the markers for the foreground objects
        threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
        // Dilate a bit the dist image
        Mat kernel1 = Mat::ones(3, 3, CV_8U);
        dilate(dist, dist, kernel1);
        // imshow("Peaks", dist);
        // Create the CV_8U version of the distance image
        // It is needed for findContours()
        Mat dist_8u;
        dist.convertTo(dist_8u, CV_8U);
        // Find total markers
        vector<vector<Point> > contours;
        findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        // Create the marker image for the watershed algorithm
        Mat markers = Mat::zeros(dist.size(), CV_32S);
        // Draw the foreground markers
        for (size_t i = 0; i < contours.size(); i++)
        {
            drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i)+1), -1);
        }
        // Draw the background marker
        circle(markers, Point(5,5), 3, Scalar(255), -1);
        // imshow("Markers", markers*10000);
        // Perform the watershed algorithm
        watershed(imgResult, markers);
        Mat mark;
        markers.convertTo(mark, CV_8U);
        bitwise_not(mark, mark);
        // namedWindow("Markers_v2",WINDOW_AUTOSIZE);
        // imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
        // // image looks like at that point
        // Generate random colors
        vector<Vec3b> colors;
        for (size_t i = 0; i < contours.size(); i++)
        {
            int b = theRNG().uniform(0, 256);
            int g = theRNG().uniform(0, 256);
            int r = theRNG().uniform(0, 256);
            colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
        }
        // Create the result image
        Mat dst = Mat::zeros(markers.size(), CV_8UC3);
        // Fill labeled objects with random colors
        for (int i = 0; i < markers.rows; i++)
        {
            for (int j = 0; j < markers.cols; j++)
            {
                int index = markers.at<int>(i,j);
                if (index > 0 && index <= static_cast<int>(contours.size()))
                {
                    dst.at<Vec3b>(i,j) = colors[index-1];
                }
            }
        }


        Mat canny;
        Canny(dst,canny,50,150);
        namedWindow("Final Result",WINDOW_AUTOSIZE);
        imshow("Final Result", canny);

        Mat temp;
        temp = imread("template.png",CV_HAL_DFT_STAGE_COLS);

        Mat btemp,bimage,ctemp,cimage;

        GaussianBlur(temp,btemp,Size(3,3),1.4,1.4);
        Canny(btemp,ctemp,50,150);
        
        double d = matchShapes(canny,ctemp, CONTOURS_MATCH_I1,0);
        //cout << d << endl;
        if(d<0.005){p++;printf("BALLS!%d\n",p);}


        // can also try Hough transform:
        // vector<Vec3f>circles;
        // HoughCircles(mark,circles,HOUGH_GRADIENT,1,5);
        // if(!circles.empty()){p++;cout<<"BALLS!"<<p<<endl;}
        // for(int i=0;i<circles.size();i++){
        //     cout << circles[i] << endl;
        // }

        char c=(char)waitKey(25);
	    if(c==27)
	      break;
    }
    cap.release();
    destroyAllWindows();

    return 0;
}