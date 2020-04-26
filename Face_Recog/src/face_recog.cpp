#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include<string>
#include <unistd.h>

using namespace cv;
using namespace cv::face;
using namespace std;

Mat processed;
int size;
Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();
vector<Mat> images;
vector<int> labels;
String face_cascade_name = "/home/shiven/Desktop/PETcat_vision/Face_Recog/src/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;

///////////////////////////////////////////////////////////////////////////////////////////////
int faceDetector(Mat& frame){

    cvtColor(frame, frame, CV_BGR2GRAY);
    GaussianBlur(frame,frame, Size(7,7), 1.5, 1.5);
    equalizeHist(frame,frame);
    std::vector<Rect> faces;
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    face_cascade.detectMultiScale( frame, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    if(faces.size()==0){
        cout<<"No Face found\n";
        return 0;}
    else{
        
        cout<<"Face found!\n";
        
        frame(faces[0]).copyTo(processed);
        return 1;
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////
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
///////////////////////////////////////////////////////////////////////////////////////////////
void updateModel(){

        size++;//new label made now

        //create directory for storing images
        cout<<"Storing for future"<<endl;
        string folderName = "s";
        ostringstream s1; 
        s1<<(size+1);//since indexing from 0
        string s = s1.str(); 
        folderName = folderName + s;
        string folderCreateCommand = "mkdir " + folderName;

        string homePath="/home/shiven/Desktop/PETcat_vision/Face_Recog/rbuild/at";
        int r1=chdir("/home/shiven/Desktop/PETcat_vision/Face_Recog/rbuild/at");
        int r=system(folderCreateCommand.c_str());
        
        int curr_frame=1;
        vector<Mat> newImages;
        vector<int> newLabels;

        //loop over video feed
        VideoCapture cap(0); // open the default camera
        if(!cap.isOpened())cout<<"cat's with no eyes see no one"<<endl;
        
        for(int i=0;i<10;)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera

        if(frame.empty()!=1){

            if(faceDetector(frame)==1){

                string fullPath=homePath;
                fullPath = fullPath + ("/"+folderName);
        
                ostringstream s2;
                s2<<curr_frame;
                s=s2.str();
                fullPath =fullPath + "/"+s+".pgm";
                imwrite(fullPath, processed);

                //write to .csv file
                fstream fout;
                fout.open((homePath+".txt").c_str(), ios::out|ios::app);
                fout<<fullPath;
                fout<<";";
                fout<<size;
                fout<<endl;

                //update the new train data
                newImages.push_back(processed);
                newLabels.push_back(size);
                images.push_back(processed);
                labels.push_back(size);

                curr_frame++;
                i++;
            }
        }
        else continue;
    }
        model->update(newImages,newLabels);
        cout<<"Model Updated\n";
        model->save("lbph_trained_data.yml");        
}
///////////////////////////////////////////////////////////////////////////////////////////////
void recogniser(){

    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened()){
        cout<<"cats with no eyes see no one"<<endl;
        }

    int predictedLabel = -2;
    while(predictedLabel==-2){

        Mat testSample;
        cap >> testSample;

        if(faceDetector(testSample)==1){

            testSample=processed;
    
            model->read("lbph_trained_data.yml");

            //model->setThreshold(0.0);

            double confidence = 0.0;
            model->predict(testSample, predictedLabel, confidence);
    
            cout << "Predicted class = " << predictedLabel << endl;
            //cout << "Confidence = " <<confidence<<endl;
        
            if(predictedLabel==-1){
                cap.release();
                updateModel();
            }
        }
        else 
            continue;
    }
}
void firstTrain(){
    if(images.size() <= 1) {
        string error_message = "Dataset too small";
        CV_Error(Error::StsError, error_message);
        } 
    model->train(images, labels);
    model->save("lbph_trained_data.yml");
}
///////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char *argv[]) {
    
    if (argc != 2) {
        cout << "usage: " << argv[0] << " <csv.ext>" << endl;
        exit(1);
    }
    try {
        read_csv(string(argv[1]), images, labels);
    } catch (const cv::Exception& e) {
        cerr << "Error opening file \"" << string(argv[1]) << "\". Reason: " << e.msg << endl;
        exit(1);
    }
    //firstTrain();
    recogniser();
    return 0; 
}
///////////////////////////////////////////////////////////////////////////////////////////////





