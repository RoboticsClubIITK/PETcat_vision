#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[])
{
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " classifier.xml\n";
    std::exit(EXIT_FAILURE);
  }
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
  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
  cv::VideoCapture capture(0);
  if (not capture.isOpened()) {
    std::cerr << "cannot open video capture device\n";
    std::exit(EXIT_FAILURE);
  }
  cv::Mat image;
  cv::Mat grayscale_image;
  std::vector<cv::Rect> features;

  while (capture.read(image) and (not image.empty())) {
    cv::cvtColor(image, grayscale_image, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(grayscale_image, grayscale_image);
    for(int i=0;i<argc-1;i++)
    {
      std::vector<cv::Rect> features;
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

    switch (cv::waitKey(10))
    {
      case 'q':
        std::exit(EXIT_SUCCESS);
      case 'Q':
        std::exit(EXIT_SUCCESS);
      default:
        break;
    }
  }
  return EXIT_SUCCESS;
}
