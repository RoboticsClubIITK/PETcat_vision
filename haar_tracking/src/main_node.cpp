#include "ros/ros.h"
#include "haar_tracking/point_msg.h"

//Callback function to display or log the points.
void clbk(const haar_tracking::point_msg::ConstPtr& msg) {
    for(int i=0;i<msg->another_field;i++)
    ROS_INFO("x=%.2f, y=%.2f | ", msg->points[i].x, msg->points[i].y);
}

int main(int argc, char **argv)
{
  // ROS objects
  ros::init(argc, argv, "main_node");
  ros::NodeHandle n;
  ros::Subscriber sub = n.subscribe("center", 1, clbk);
  ros::spin();
}
