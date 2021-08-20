#!/usr/bin/env python

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from sensor_msgs.msg import Image, CameraInfo
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
import time
import subprocess, shlex, psutil
from cv_bridge import CvBridge
import cv2 
import numpy as np
import tf
import pyrealsense2
# Switch controller server client libs
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest , SwitchControllerResponse
from controller_manager_msgs.srv import ListControllers, ListControllersRequest , ListControllersResponse


class DataCollection: 

    def __init__(self): 

        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('data_collection_moveit', anonymous=True)
        self.robot = moveit_commander.RobotCommander()

        group_name = "panda_arm"
        self.group = moveit_commander.MoveGroupCommander(group_name)

        self.bridge = CvBridge()

        self.img_sub = rospy.Subscriber('/color/camera_info', CameraInfo, self.color_camear_info_cb)
        self.img_sub = rospy.Subscriber('/color/image_raw', Image, self.color_image_cb)
        self.depth_sub = rospy.Subscriber('/depth/image_rect_raw', Image, self.depth_image_cb)
        self.img_msg = None
        self.depth_msg = None
        self.img_info_msg = None
        self.img_msg_received = False
        self.depth_msg_received = False
        self.img_info_msg_received = False
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.select_point_cb)

        self.experiment_name = "experiment_" 
        self.experiment_count = 189

        self.experiment_path = "~/data/"

        self.current_controller = "position"
        self.last_controller = "franka_zero_torque_controller"

        # recording rosbag proces 
        self.rosbag_proc = None
        self.is_recording = False

        self.loop()
        # self.plan_with_camera()

    def select_point_cb(self, event,x,y,flags,param):
        if self.img_msg_received and self.img_info_msg_received and self.depth_msg_received:

            if event == cv2.EVENT_LBUTTONDOWN: # Left mouse button
                rospy.loginfo("Mouse event: {} {}".format(x,y))
                self.convert_2d_to_3d(x, y)

    def convert_2d_to_3d(self, x,y):
  
        _intrinsics = pyrealsense2.intrinsics()
        _intrinsics.width = self.img_info_msg.width
        _intrinsics.height = self.img_info_msg.height
        _intrinsics.ppx = self.img_info_msg.K[2]
        _intrinsics.ppy = self.img_info_msg.K[5]
        _intrinsics.fx = self.img_info_msg.K[0]
        _intrinsics.fy = self.img_info_msg.K[4]
        _intrinsics.model  = pyrealsense2.distortion.none  
        _intrinsics.coeffs = [i for i in self.img_info_msg.D]  
        result = pyrealsense2.rs2_deproject_pixel_to_point(_intrinsics, [x, y], self.depth_msg[y,x])  

        rospy.loginfo("3D point: {} {} {}".format( result[0]/1000, result[1]/1000, result[2]/1000))

        trans = [result[0]/1000, result[1]/1000, result[2]/1000 ]
        rot = [0, 0, 0 , 1]
        self.broadcast_and_transform(trans, rot, "_camera_color_optical_frame" , "/nipple_point" , "/panda_link0")


    def broadcast_and_transform(self, trans, rot, parent, child, new_parent):

        not_found = True
        listerner = tf.TransformListener()
        b = tf.TransformBroadcaster()
        while not_found:
            for i in range(10):
                # print (trans)
                # point cloud is messed up in z-axis and 
                # breast phantom height is always known to us
                trans[2] = 0.472 
                b.sendTransform( trans, rot, rospy.Time.now(), child, parent )
                time.sleep(0.1)
            try: 
                (trans1, rot1) = listerner.lookupTransform( new_parent, child, rospy.Time(0))
                rospy.loginfo("Translation: {} {} {}".format(trans1[0],trans1[1],trans1[2]))
                not_found = False
            except ( tf.lookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.loginfo("TF Exception")
                continue

        self.record_bag_file()
        self.plan_with_camera(trans1)

    def depth_image_cb(self, data):
        self.depth_msg_received = True
        # The depth image is a single-channel float32 image
        # the values is the distance in mm in z axis
        self.depth_msg = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        
    def color_camear_info_cb(self, data):
        self.img_info_msg_received = True
        self.img_info_msg = data

    def color_image_cb(self , data):
        
        self.img_msg_received = True
        self.img_msg = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')

    def plan_with_camera(self, trans1):

        if self.is_recording:

            self.group.set_max_velocity_scaling_factor(0.1)
            current_pose = self.group.get_current_pose().pose
            rospy.loginfo("{} {}".format(self.group.get_end_effector_link(), self.group.get_planning_frame()))
            pose_goal = geometry_msgs.msg.PoseStamped()
            pose_goal.header.stamp = rospy.Time.now()
            pose_goal.header.frame_id = "panda_link0"       
            pose_goal.pose.orientation = current_pose.orientation
            pose_goal.pose.position.x = trans1[0]
            pose_goal.pose.position.y = trans1[1]
            pose_goal.pose.position.z = trans1[2]
            self.group.set_pose_target(pose_goal)
            self.group.go(wait=True)
            self.is_recording = False

    def go_to_home(self):
        if not self.is_recording:
            rospy.loginfo ("Moving to home pose. Please wait...")
            self.group.set_max_velocity_scaling_factor(0.1)
            self.group.set_named_target('poseA')
            self.group.go(wait=True)
            rospy.loginfo ("Moved to home pose...")
            time.sleep(1)

    def record_bag_file(self):

        topics_names = "/tf_static /tf /joint_states /color/camera_info /color/image_raw /depth/image_rect_raw /depth/color/points /extrinsics/depth_to_color"
        self.experiment_count += 1
        name =  self.experiment_name + str(self.experiment_count) 
        command = "rosrun rosbag record /tf_static /tf /joint_states /color/camera_info /color/image_raw /depth/image_rect_raw /depth/color/points /extrinsics/depth_to_color -O " + name
        command = shlex.split(command)
        # print(command)
        self.rosbag_proc = subprocess.Popen(command)
        rospy.loginfo("recording rosbag file. Press 's' to stop recording")
        self.is_recording = True


    def loop(self):

        while not rospy.is_shutdown(): 
            # pass
            if self.img_msg_received and self.depth_msg_received:
                cv2.imshow('image', self.img_msg)

                k = cv2.waitKey(33)
                if k==27:    # Esc key to stop
                    break
                elif k==-1:  # normally -1 returned,so don't print it
                    continue
                else:
                    if k == ord('h'):
                        self.go_to_home()
                    if k == ord('s'):
                        if self.rosbag_proc != None:
                                self.rosbag_proc.send_signal(subprocess.signal.SIGINT)
                                self.is_recording = False
                                self.go_to_home()

        cv2.destroyAllWindows()

if __name__ == "__main__": 

    data_collection = DataCollection()
