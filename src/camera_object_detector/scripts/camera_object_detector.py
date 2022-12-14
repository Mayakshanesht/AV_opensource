#!/usr/bin/env python3

import cv2
import rospy

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
#from av_messages.msg import depthandimage, object, objects
from yolo import YOLO
import numpy as np
import math

class Detector:
    def __init__(self):
        self.loadParameters()
        self.bridge = CvBridge()
        self.rgb_image = Image
        self.depth_image = None
        

    def subscribeToTopics(self):
        rospy.loginfo("Subscribed to topics")
        rospy.Subscriber(self.image_topicname, Image,
                         self.callback, queue_size=1)

    def loadParameters(self):
        '''
        do something
        '''
        self.image_topicname = rospy.get_param(
            "camera_object_detector/image_topic_name", "/imageraw")
        self.pub_topic_name = rospy.get_param(
            "camera_object_detector/object_detections_topic_name", "/camera/objectdetections")
    

    def publishToTopics(self):
        rospy.loginfo("Published to topics")
        self.DetectionsPublisher = rospy.Publisher(
            self.pub_topic_name, Image, queue_size=1)

    def callback(self, img):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(img.rgb_image, 'bgr8')
            self.depth_image = self.bridge.imgmsg_to_cv2(img.depth_image, "32FC1") ## Confirm these once
            self.Actionloop()
        except CvBridgeError as e:
            rospy.loginfo(str(e))
    
    def Actionloop(self):
        '''
        Call yolo related functions here (Reuben, Mayur)
        and the final publish function (To be done by sahil)
        '''
        yolo = YOLO()
        self.oi = yolo.inference(self.rgb_image)
        self.rosimg=CvBridge.cv2_to_imgmsg(np.asarray(self.oi), encoding='rgb8')
        self.callPublisher()
    def callPublisher(self):
        '''
        the final publisher function
        '''
        while not rospy.is_shutdown():
            self.DetectionsPublisher.publish(self.rosimg)
            
        

        