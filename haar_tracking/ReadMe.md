In HaarTracking.h, go and change the location of folder where all CascadeClassifiers are stored.
video_feed captures the video and publish it.
HaarTracking_node, subscribe to the frame published by video_feed and use it to publish the center of object detected.
