#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>


int main(int argc, char** argv)
{
  ros::init(argc,argv,"video_feed");
  ros::NodeHandle n;
  image_transport::ImageTransport it(n);
  image_transport::Publisher pub = it.advertise("image",1);
  cv::Mat image;
  cv::VideoCapture cap(0);
  ros::Rate loopRate(30);
  while(ros::ok())
  {
    cap>>image;
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", image).toImageMsg();
    pub.publish(msg);
    ros::spinOnce();

    loopRate.sleep();
  }
  cap.release();
  cv::destroyAllWindows();
}
