#! /usr/bin/env python

from __future__ import print_function

import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

from tf.transformations import quaternion_from_matrix
import tf



class image_converter:

    def __init__(self):

    
        self.image_pub = rospy.Publisher("/final_image",Image, queue_size=10)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/color/image_raw",Image,self.image_callback)
        self.image_info_sub = rospy.Subscriber("/color/camera_info",CameraInfo,self.info_callback)

        self.image_data = None
        self.cv_image = None
        self.got_image = False
        self.projection_matrix = None
        self.got_projection_matrix = False
        
        self.loop()

    def info_callback(self, data):

        self.got_projection_matrix = True
        self.projection_matrix = data

    def image_callback(self,data):

        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.got_image = True

        except CvBridgeError as e:
            print(e)    

    def broadcase_tf (self, trans, rot, parent , child ):
        br = tf.TransformBroadcaster()
        br.sendTransform( trans, rot, rospy.Time.now(), child, parent )


    def world_to_camera_inv(self):

        # trans, rot = self.get_tf('_camera_link', 'panda_link0')
        trans = [0.6075893497936095, -0.007869040896281721, -0.5219568203271748] 
        rot = [-0.014903907095510012, -0.7277402908416677, 0.01751112481055246, 0.6854672152239257]

        # rosrun tf tf_echo /_camera_link /panda_link0
        # trans = [0.608, -0.008, -0.522]
        # rot = [-0.015, -0.728, 0.018, 0.685]

        mat_x = quaternion_matrix(rot)
        mat_x[:3, 3] = trans
        # print (trans, rot)
        return mat_x 

    def get_tf(self, parent, child):

        not_found = True
        listener = tf.TransformListener()

        rate = rospy.Rate(10)
        while not_found:
            try: 

                (trans, rot) = listener.lookupTransform( parent, child, rospy.Time(0))
                not_found = False
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

        return trans, rot


    def calculate_point_on_image(self, point):

        u, v = 0, 0
        if (self.got_projection_matrix and self.got_image):
            
            P = np.array( self.projection_matrix.P ).reshape(3,4)
            point = np.array(point).reshape(-1,1)
            image_point = np.matmul ( P, point ) 

            u, v = image_point[0] / image_point[2] , image_point[1] / image_point[2]
            # u, v = image_point[0]   , image_point[1]  

            print (image_point, u, v)

        return u, v

    def loop(self):

        while not rospy.is_shutdown():
            # self.world_to_camera_inv()

            if (self.got_image):
                try:
                    # point with respect to world
                    point = [-0.047, 0.032, 0.459, 1]

                    u, v = self.calculate_point_on_image(point)
                    (rows,cols,channels) = self.cv_image.shape
                    if cols > 60 and rows > 60 :
                        cv2.circle(self.cv_image, (u,v), 10, 255, -1)

                    self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.cv_image, "bgr8"))
                    cv2.imshow("Image window", self.cv_image)

                    cv2.waitKey(3)

                except CvBridgeError as e:
                    print(e)



def main(args):
    
    rospy.init_node('image_converter', anonymous=True)

    ic = image_converter()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

