#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
using namespace std;
using namespace cv;

int modeFingers(vector<int>v){
    int count[5]={0,0,0,0,0};
    for(int i=0;i<v.size();i++){
        for(int j=1;j<=5;j++){
            if(v[i]==j)count[j-1]++;
        }
    }
    int a=-1,p=0;
    for(int o=0;o<5;o++){
        if(a<count[o]){
            a = count[o];
            p = o;
        }
    }
    return p;
}

int main(int argc, char *argv[])
{
    //======================================================================================================================
    //CODE STARTS. KEEP YOUR HAND CLOSE TO THE CAMERA
    //======================================================================================================================
    VideoCapture cap(0);
    Mat src;
    namedWindow("skin",WINDOW_NORMAL);
    resizeWindow("skin",700,600);
    vector<int>Finger;
    int numFrame=5,fin=0;
    while(1)
    {
        cap >> src;
        Mat YCrCb,hsv;
        //GaussianBlur(src,src,Size(3,3),1.1,1.1,4);
        cvtColor(src,YCrCb,COLOR_BGR2YCrCb,0);
        cvtColor(src,hsv,COLOR_BGR2HSV,0);

        Mat emp(src.size().height,src.size().width,CV_8UC3,Scalar(0,0,0));

        double R,G,B,H,S,V,Y,Cr,Cb;

        for(int x=0;x<src.size().width;x++){
            for(int y=0;y<src.size().height;y++){
                R = src.at<Vec3b>(y,x)[2]/(1.001);
                G = src.at<Vec3b>(y,x)[1];
                B = src.at<Vec3b>(y,x)[0];
                Y = YCrCb.at<Vec3b>(y,x)[0];
                Cr = YCrCb.at<Vec3b>(y,x)[1];
                Cb = YCrCb.at<Vec3b>(y,x)[2];
                H = hsv.at<Vec3b>(y,x)[0];
                S = hsv.at<Vec3b>(y,x)[1];
                V = hsv.at<Vec3b>(y,x)[2];

                //The values you can adjust: (according to Azhar's experience doing this)
                //R-G lowest value is 15;
                //(the more this value's threshold is, the more HARD detector it would be; meaning it will detect skin less accurately and will ignore backgrond red objects efficiently)
                //R value you can leave that as it is
                if(((H>=0.0)&&(H<=50.0)&&(S>=0.23)&&(S<=0.68)&&(R>99)&&(G>40)&&(B>20)&&(R>G)&&(R>B)&&((R-G)>20))||((R>99)&&(G>40)&&(B>20)&&(R>G)&&(R>B)&&((R-G)>20)&&(Cr>135)&&(Cb>85)&&(Y>80)&&(Cr<=(1.5862*Cb+20))&&(Cr>=(0.3448*Cb+76.2069))&&(Cr>=((-4.5652*Cb)+234.5652))&&(Cr<=(-1.15*Cb+301.75))&&(Cr<=(-2.2857*Cb+432.85)))){
                    emp.at<Vec3b>(y,x)[0] = 0;
                    emp.at<Vec3b>(y,x)[1] = 225;
                    emp.at<Vec3b>(y,x)[2] = 0;
                }
            }
        }

        GaussianBlur(emp,emp,Size(3,3),1.4,1.4,4);

        Mat thresh,show;
        show = emp;
        cvtColor(emp,emp,COLOR_BGR2GRAY);
        threshold(emp,thresh,50,255,THRESH_BINARY);
        vector<vector<Point>>cont;
        vector<vector<Point>>contours;
        vector<Vec4i>hierarchy;
        findContours(thresh,cont,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE, Point(0,0));
        //i'm gonna ignore the small contours so that the image doesn't clutter
        //=========================================================================
        float min_area=500.0,area;
        for(int i=0;i<cont.size();i++){
            area = contourArea(cont[i]);
            if(area>min_area){
                contours.push_back(cont[i]);
            }                        
        }
        //=========================================================================

        vector<vector<Point>>Hull(contours.size());
        for(int i=0;i<contours.size();i++){
            convexHull(contours[i],Hull[i]);
        }
        //only cosidering the contour with maximum area but with Hull vector, not contour one
        double a=-1;
        int pos=0;
        for(int i=0;i<Hull.size();i++){
            if(a<contourArea(Hull[i])){
                a=contourArea(Hull[i]);
                pos = i;
            }
        }

        vector<int>hull;
        convexHull(Mat(contours[pos]),hull);

        Mat convex(src.size().height,src.size().width,CV_8U,Scalar(0,0,0));        
        drawContours(convex,Hull,pos,Scalar(255,225,255));
        drawContours(convex,contours,pos,Scalar(255,0,0));


        //=====================================================================================================================
        //DO NOT TOUCH ANYTHING ABOVE THIS LINE EXCEPT SOME PAPAMETERS IF YOU WANT  START YOUR CODE BELOW THIS
        //=====================================================================================================================

        //start....
      //=================================================================================================================================
        //This part basically depends on the efficiency of the skin detection.

      if((int)hull.size()>3)
      {
        vector<Vec4i>defects;
        convexityDefects(contours[pos],hull,defects);
        int numFingers=1;
        for(int i=0;i<defects.size();i++)
        {
          int s=defects[i][0];
          int e=defects[i][1];
          int f=defects[i][2];
          Point start = contours[pos][s];
          Point end = contours[pos][e];
          Point far = contours[pos][f];
          double a = sqrt((start.x-end.x)*(start.x-end.x)+(start.y-end.y)*(start.y-end.y));
          double b = sqrt((start.x-far.x)*(start.x-far.x)+(start.y-far.y)*(start.y-far.y));
          double c = sqrt((far.x-end.x)*(far.x-end.x)+(far.y-end.y)*(far.y-end.y));
          double value=(b*b+c*c-a*a)/(2*b*c);
          double angle = acos(value);
          if(angle<acos((-1)/2))
          {
            numFingers++;
          }
        }
        //taking average(mode) over some frames
        //=======================================================================================================================
        if(Finger.size()<=numFrame){
            if(numFingers<=5){
                Finger.push_back(numFingers);
            }
        }
        else{
            fin = modeFingers(Finger);
            Finger.clear();
            cout << fin << " fingers" << endl;
        }
        //========================================================================================================================

      }
      imshow("skin",show);
      imshow("Hull",convex);
  //=============================================================================================================================
        //Here is Aditya's view on which gestures are perfect:
        // Use back side of your palm because that hides all your faults.
        //       Number of Fingers        Gestures
        //             5                  Open Palm(There is no other way)
        //             4                  Thumb folded(There is no other way)
        //             3                  Both are fine.
        //             2                  Victory Symbol.
        //             1                  Middle finger. Doesn't detect single finger properly as of now
  //===============================================================================================================================
      char c=(char)waitKey(25);
  	  if(c==27)
  	  break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
