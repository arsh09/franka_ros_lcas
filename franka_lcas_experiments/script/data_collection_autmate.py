#!/usr/bin/env python

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
import time
import subprocess, shlex, psutil


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

        self.experiment_name = "experiment_" 
        self.experiment_count = 191

        self.experiment_path = "~/data/"

        self.current_controller = "position"
        self.last_controller = "franka_zero_torque_controller"

        self.loop()


    def go_to_home(self):
        print ("\nMoving to home pose. Please wait...")
        self.group.set_max_velocity_scaling_factor(0.2)
        self.group.set_named_target('poseA')
        self.group.go(wait=True)
        time.sleep(1)

        self.group.set_named_target('nippleB')
        self.group.go(wait=True)
        # time.sleep(3)

        raw_input("Moved to home pose. Press Enter to put in zero torque move: ")
        return True

    def go_to_zero_mode(self):

        if self.current_controller == "position": 

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
            return True

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

    def record_bag_file(self):

        topics_names = "  /tf /joint_states /xServTopic "
        self.experiment_count += 1
        name =  self.experiment_name + str(self.experiment_count) 
        command = "rosrun rosbag record /tf /joint_states /xServTopic -O " + name
        command = shlex.split(command)
        # print(command)
        rosbag_proc = subprocess.Popen(command)
        raw_input("Recording bag file. Press Enter to stop recording and go to home: \n")
        rosbag_proc.send_signal(subprocess.signal.SIGINT)

        return True

    def loop(self): 

        while not rospy.is_shutdown():
            # self.record_bag_file()

            no_error = self.go_to_position_mode()
            if no_error: 
                no_error = self.go_to_home()
                if no_error :
                    no_error = self.go_to_zero_mode()
                    # raw_input("Robot is zero torque mode. Press Enter to start recording bag file: ")
                    if no_error: 
                        no_error = self.record_bag_file()



if __name__ == "__main__": 

    data_collection = DataCollection()
