#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " cascade.xml\n";
    std::exit(EXIT_FAILURE);
  }
std::cout<<argv[0];
cv::CascadeClassifier classifier[argc-1];
std::string name[argc-1];
std::string label[argc-1];
for(int i=0;i<argc-1;i++)
{  name[i]=argv[i+1];
  int pos=name[i].find(".");
  label[i]=name[i].substr(0,pos);
  classifier[i].load(name[i]);
}

  const char* const window_name{"Recognition Window"};

  cv::namedWindow(window_name, cv::WINDOW_NORMAL);

  cv::Mat image;
  image = cv::imread("face.png");
  if(! image.data )
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

  cv::Mat grayscale_image;

    cv::cvtColor(image, grayscale_image, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(grayscale_image, grayscale_image);

for(int i=0;i<argc-1;i++)

 {   std::vector<cv::Rect> features;
classifier[i].detectMultiScale(grayscale_image, features);

    for (auto&& feature : features) {
int y;
      if(feature.tl().x-15 >15)
         y=feature.tl().x-20;
      else
          y=feature.tl().x+20;
      cv::rectangle(image, feature, cv::Scalar(0, 255, 0), 2);
      cv::putText(image,label[i] , cv::Point(y, feature.tl().y), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(0, 255, 0), 2);
    }
}

    cv::imshow(window_name, image);
    cv::imwrite("test.png",image);

    cv::waitKey(0);       
        return 0;
}
