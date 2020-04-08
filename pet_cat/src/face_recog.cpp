#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include<fstream>
#include <sstream>
#include <sys/stat.h> 
#include <sys/types.h> 
#include<string>
#include <unistd.h>
#include<algorithm>
using namespace cv;
using namespace cv::face;
using namespace std;
Mat processed;
int size;
Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();

vector<Mat> images;
vector<int> labels;
String face_cascade_name = "/home/shiven/Desktop/pet_cat/src/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;

int faceDetector(Mat frame){
    std::vector<Rect> faces;
    
    face_cascade.detectMultiScale( frame, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    if(faces.size()==0){
        cout<<"No Face found"<<endl;
        return 0;}
    else{
        
        cout<<"Face found!"<<endl;
        
        frame(faces[0]).copyTo(processed);
        return 1;
    }
}
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(Error::StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
    size = *max_element(labels.begin(), labels.end());
}
void updateCSV(){
        cout<<"Storing for future"<<endl;
        string folderName = "s";
        ostringstream s1; 
        
        size++;//new label made now 
        s1<<(size+1);//since indexing from 0
        string s = s1.str(); 
        folderName+=s;
        string folderCreateCommand = "mkdir " + folderName;

        //cout<<folderCreateCommand<<endl;
        string homePath="/home/shiven/Desktop/pet_cat/rbuild";
        int r1=chdir("/home/shiven/Desktop/pet_cat/rbuild/at");
        //if(r1==0)cout<<"changed"<<endl;
        int r=system(folderCreateCommand.c_str());
        //if(r==0)cout<<"created"<<endl;

        int curr_frame=1;
        vector<Mat> newImages;
        vector<int> newLabels;
        //loop over video feed
        VideoCapture cap(0); // open the default camera
        if(!cap.isOpened())cout<<"cat's with no eyes see no one"<<endl;
        
        int x;
        //Mat edges;
        //namedWindow("edges",1);
        for(x=0;x<10;)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera

        if(frame.empty()!=1){
        cvtColor(frame, frame, CV_BGR2GRAY);
        GaussianBlur(frame, frame, Size(7,7), 1.5, 1.5);
        equalizeHist(frame, frame);

        if(faceDetector(frame)==1){

        string fullPath=homePath+"/at";
        fullPath+=("/"+folderName);
        
        ostringstream s2;
        s2<<curr_frame;
        s=s2.str();
        fullPath+="/"+s+".pgm";
        imwrite(fullPath, processed);

        fstream fout;
        fout.open((homePath+"/at.txt").c_str(), ios::out|ios::app);
        fout<<fullPath;
        fout<<";";
        fout<<size;
        fout<<endl;

         newImages.push_back(processed);
         newLabels.push_back(size);
         images.push_back(processed);
         labels.push_back(size);

         curr_frame+=1;
         x++;
        }}
        else continue;
    }
         model->update(newImages,newLabels);
}
int main(int argc, const char *argv[]) {
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    if (argc != 2) {
        cout << "usage: " << argv[0] << " <csv.ext>" << endl;
        exit(1);
    }
    string fn_csv = string(argv[1]);
    // vector<Mat> images;
    // vector<int> labels;
    try {
        read_csv(fn_csv, images, labels);
    } catch (const cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        exit(1);
    }
    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(Error::StsError, error_message);
    }
    //Get this from videocam
    //Mat testSample=imread("face.jpg");
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened()){cout<<"cat's with no eyes see no one"<<endl;return -1;}
    int predictedLabel = -2;
    while(predictedLabel==-2)
    {
    Mat testSample;
    cap >> testSample; // get a new frame from camera
    cvtColor(testSample, testSample, CV_BGR2GRAY);
    GaussianBlur(testSample, testSample, Size(7,7), 1.5, 1.5);
    equalizeHist(testSample, testSample);

   
    if(faceDetector(testSample)==1){

    testSample=processed;
    
    model->train(images, labels);
    model->setThreshold(150.0);//maybe tune this as well later
    
    double confidence = 0.0;
    model->predict(testSample, predictedLabel, confidence);
    
    cout << "Predicted class = " << predictedLabel << endl;
    //cout << "Confidence = " <<confidence<<endl;

    //WE NOW HAVE THE PREDICTION
    if(predictedLabel==-1){

        updateCSV();
        
    }
    }
    else continue;
    }
    return 0; 
}






