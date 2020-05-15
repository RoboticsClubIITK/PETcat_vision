#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <std_msgs/Int32.h>
#include <opencv2/opencv.hpp>
#include "HaarTracking.h"
#include "geometry_msgs/Point.h"
#include "haar_tracking/point_msg.h"
static const std::string OPENCV_WINDOW = "Tracked object";

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  ros::Publisher out = nh_.advertise<haar_tracking::point_msg>("center",1);

public:
  ImageConverter()
    : it_(nh_)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("image", 1, &ImageConverter::imageCb, this);
    image_pub_ = it_.advertise("center_out", 1);

    cv::namedWindow(OPENCV_WINDOW);
  }

  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }


    HaarTracking object;
    object.locateObject(cv_ptr->image);

    cv::Mat image_output = object.getImage();
    cv_ptr->image = image_output;

    haar_tracking::point_msg center_of_rect;
    std::vector<cv::Point> Centres;
    Centres=object.getCentres();
    center_of_rect.another_field=Centres.size();
    for(size_t i=0;i<Centres.size();i++)
    {
      geometry_msgs::Point point;
      point.x = Centres[i].x;
      point.y=Centres[i].y;
      point.z=0;
      center_of_rect.points.push_back(point);
    }
    out.publish(center_of_rect);

    // Update GUI Window
    cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    cv::waitKey(3);

    // Output modified video stream
    image_pub_.publish(cv_ptr->toImageMsg());
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "HaarTracking_node");
  ImageConverter ic;
  ros::spin();
  return 0;
}
