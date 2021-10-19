#! /usr/bin/env python

import rospy
import cv2 
import sys
import numpy 
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import os 
from cv_bridge import CvBridge
import shlex
import subprocess
import time

from sensor_msgs.msg import Image, CameraInfo, PointCloud2


# Switch controller server client libs
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest , SwitchControllerResponse
from controller_manager_msgs.srv import ListControllers, ListControllersRequest , ListControllersResponse



class BerryDataCollection: 
    def __init__(self, experiment_path):

        rospy.init_node('berry_data_collection')
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        group_name = "panda_arm"
        self.group = moveit_commander.MoveGroupCommander(group_name)
        self.scene = moveit_commander.PlanningSceneInterface()
        self.display_trajectory_publisher = rospy.Publisher( "/move_group/display_planned_path", moveit_msgs.msg.DisplayTrajectory, queue_size=20)

        self.experiment_count = 0 # increment this number
        self.experiment_path = experiment_path

        self.bridge = CvBridge()

        self.color_img = None
        self.color_img_received = False
        self.depth_img = None
        self.depth_img_received = False
        self.img_sub = rospy.Subscriber('/color/image_raw', Image, self.color_image_cb)
        # self.depth_sub = rospy.Subscriber('/depth/image_rect_raw', Image, self.depth_image_cb)

        self.current_controller = 'position'
        
        self.group.set_max_velocity_scaling_factor(0.15)
        self.rosbag_proc = None
        self.loop()

        # get_pointcloud data 
        # self.do_acquire_pc = False
        # self.pointcloud = np.array([])
        # self.pointcloud_sub = rospy.Subscriber('/depth/color/points', PointCloud2, self.pointcloud_cb)

    def color_image_cb(self, msg) :
        self.color_img_received = True
        self.color_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')        
        self.img_shown = self.color_img #cv2.cvtColor(self.color_img, cv2.COLOR_RGB2BGR )

    def depth_image_cb(self, msg):
        self.depth_img_received = True
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def go_to_pose(self, pose = "ready"):
        if pose in self.group.get_named_targets():
            self.group.set_named_target(pose)
            rospy.loginfo("Moving to {} target pose".format(pose))
            self.group.go()

    def go_to_position_mode(self):
        if self.current_controller == "torque":
            rospy.wait_for_service("/controller_manager/switch_controller")
            try:
                service = rospy.ServiceProxy("/controller_manager/switch_controller", SwitchController)
                request = SwitchControllerRequest()
                request.start_controllers.append( "position_joint_trajectory_controller" )
                request.stop_controllers.append( "franka_zero_torque_controller" )
                request.strictness = 2
                response = service(request)
                self.current_controller = "position"
                return response.ok
            except : 
                rospy.logerr("Switch controller server is down. Unable to switch contoller")
                return False
        else : 
            return True

    def go_to_zero_mode(self):
        print(self.current_controller)
        if self.current_controller == "position":
            print("Switching to torque mode")
            rospy.wait_for_service("/controller_manager/switch_controller")
            try:
                service = rospy.ServiceProxy("/controller_manager/switch_controller", SwitchController)
                request = SwitchControllerRequest()
                request.start_controllers.append( "franka_zero_torque_controller" )
                request.stop_controllers.append( "position_joint_trajectory_controller" )
                request.strictness = 2
                response = service(request)
                self.current_controller = "torque"
                return response.ok
            except : 
                rospy.logerr("Switch controller server is down. Unable to switch contoller")
                return False
        else:
            print("Already in torque mode")
            return True


    def record_bag_file(self):
        topics_names = " /tf /joint_states "
        self.experiment_count += 1
        name =  os.path.join(self.experiment_path ,  "berry_data_sample_" + str(self.experiment_count) )
        print("Recording bag", name)
        command = "rosrun rosbag record /tf /joint_states  -O " + name
        command = shlex.split(command)
        self.rosbag_proc = subprocess.Popen(command)
        print("Recording bag file. Press 'r' to stop the recording: \n")
        return True

    def recover_error(self):
        command = "rostopic pub -1 /franka_control/error_recovery/goal franka_control/ErrorRecoveryActionGoal '{}'"
        command = shlex.split(command)
        subprocess.Popen(command)


    def loop(self):

        while not rospy.is_shutdown():

            if self.color_img_received: 

                cv2.imshow('image', self.color_img)

                k = cv2.waitKey(1)

                if k == 27: 
                    cv2.destroyAllWindows()
                    break

                elif k==-1:  # normally -1 returned,so don't print it
                    continue
                else:
                    # print(k)

                    if k == ord('h'):
                        print("H: Go to home position.")
                        self.go_to_pose(pose = "poseA")

                    if k == ord('e'):
                        self.recover_error()

                    if k == ord('p'):
                        self.go_to_position_mode()

                    if k == ord('t'):
                        print("Trying to enter torque mode")
                        is_torque_mode = self.go_to_zero_mode()

                    if k == ord('r'):
                        # save image here.
                        if self.rosbag_proc is None:
                            self.record_bag_file()
                        else:
                            self.rosbag_proc.send_signal(subprocess.signal.SIGINT)
                            self.rosbag_proc = None
                            print("Finished recording")

                    if k == ord('s'):
                        is_torque_mode = self.go_to_zero_mode()
                        if self.rosbag_proc is None:
                            self.record_bag_file()

                    if k == ord('f'):
                        if self.rosbag_proc is not None:
                            self.rosbag_proc.send_signal(subprocess.signal.SIGINT)
                            self.rosbag_proc = None
                            print("Finished recording")

                        self.go_to_position_mode()
                        time.sleep(1)
                        self.go_to_pose(pose = "poseA")


                        
                            

if __name__ == '__main__':

    berryData = BerryDataCollection(experiment_path = "/home/arshad/Desktop/data/7")