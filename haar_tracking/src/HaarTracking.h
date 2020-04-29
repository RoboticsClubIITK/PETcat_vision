#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <experimental/filesystem>

namespace fs=std::experimental::filesystem;
class HaarTracking
{
private:
  int num_cascades;
  std::vector<std::string> name;
  std::vector<std::string> label;
  cv::CascadeClassifier classifier[100];
  cv::Mat image;
  cv::Mat grayscale_image;
  std::vector<cv::Point> centre_of_rect;
public:
  HaarTracking();
  void locateObject(cv::Mat src);
  std::vector<cv::Point> getCentres();
  cv::Mat getImage();
};



HaarTracking::HaarTracking()
{

  num_cascades=0;
  /////////////////////////////********NOTE**********///////////////////////////////
  //Before building, Change the location to the folder where all Cascades are stored
  
  std::string path="/home/legolas/catkin_ws/src/haar_tracking/CascadeClassifiers";
  for (const auto & entry : fs::directory_iterator(path))
  {
    name.push_back((std::string)entry.path());
    //Extracting Labels //
     std::string temp;
     char sep = '/';
      #ifdef _WIN32
        sep = '\\';
      #endif
     size_t j = name[num_cascades].rfind(sep, name[num_cascades].length());
    if (j != std::string::npos) {
     temp=name[num_cascades].substr(j+1, name[num_cascades].length() - j);
   }
   else
   {
     temp=name[num_cascades];
   }
     int pos=temp.find(".");
     label.push_back(temp.substr(0,pos));
     //Labelled//
     classifier[num_cascades].load(name[num_cascades]);
     if( !classifier[num_cascades].load(name[num_cascades]))
     {
    std::cout << "--(!)Error loading "<<label[num_cascades]<<".xml\n";
    }
    num_cascades++;
  }
}

void HaarTracking::locateObject(cv::Mat src)
{
  image=src;
  std::vector<cv::Point> Centres;
  cv::cvtColor(image, grayscale_image, cv::COLOR_BGR2GRAY);
  cv::equalizeHist(grayscale_image, grayscale_image);
  for(int i=0;i<num_cascades;i++)
  {
    std::vector<cv::Rect> features;
    classifier[i].detectMultiScale(grayscale_image, features);
    for (auto&& feature : features) {
      int y;
      if(feature.tl().x-15 >15)
         y=feature.tl().x-20;
      else
          y=feature.tl().x+20;
      cv::Point center_of_rect = (feature.br() + feature.tl())*0.5;
      Centres.push_back(center_of_rect);
      cv::rectangle(image, feature, cv::Scalar(0, 255, 0), 2);
      cv::putText(image,label[i] , cv::Point(y, feature.tl().y), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(0, 255, 0), 2);
    }
  }
  centre_of_rect=Centres;
}

std::vector<cv::Point> HaarTracking::getCentres()
{
  return centre_of_rect;
}

cv::Mat HaarTracking::getImage()
{
  return image;
}
