#!/usr/bin/env python

# imports 
import sys
import os
import copy
import rospy
import time
import numpy as np

# moveit related import 
import moveit_commander
from moveit_msgs.msg import DisplayTrajectory

# used to run Python3 script from Python2
import subprocess, shlex, psutil

# image manipulation 
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge
import cv2 
import imutils

# use for joint trajectory control (not used) 
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# for forward kinematics
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
from tf.transformations import quaternion_from_matrix


class ReachToPalpateTaskValidation: 

    def __init__(self): 
        """ - init will start the move group for the arm 
            - subscribe to the image topic (from RealSense) 
            - makes some useful class variable 
            - makes the robot go to home pose (i.e. poseA)
            - prints the options 
            - and start the loop function.  
        """

        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('advance_robotics_workshop_node', anonymous=True)

        self.BASE_PATH = "/home/arshad/Documents/artemis_project_trained_models/reach_to_palpate_validation_models"

        group_name = "panda_arm"
        self.group = moveit_commander.MoveGroupCommander(group_name)
        self.display_trajectory_publisher = rospy.Publisher( "/move_group/display_planned_path", DisplayTrajectory, queue_size=20)
        self.scene = moveit_commander.PlanningSceneInterface()
        self.robot = moveit_commander.RobotCommander()
        
        # you can directly publish a joint trajectory to 
        # this topic as well. However, it will not be smooth
        # and you might have to do some pre-processing yourself 
        # on the joint trajectory.
        self.trajectory_pub = rospy.Publisher('/position_joint_trajectory_controller/command', JointTrajectory, queue_size = 10)

        # this bridge is used to convert 
        # sensor_msgs/Image to NumPy array
        self.bridge = CvBridge()        

        
        # image topic subscriber (color image) 
        self.img_sub = rospy.Subscriber('/color/image_raw', Image, self.color_image_cb)

        # some useful image related class variables
        self.img_msg = None
        self.img_msg_rtp = None
        self.img_shown = None
        self.img_msg_received = False
        cv2.namedWindow('image')

        # setup for forward kinematics 
        robot = URDF.from_parameter_server()
        tree = kdl_tree_from_urdf_model(robot)
        base_link, end_link = "panda_link0", "xela_sensor_frame"
        self.kdl_kin = KDLKinematics(robot, base_link, end_link)
 
        # this variable holds the 
        # last planned trajectory 
        # that gets executed on the 
        # robot if the 'e' is pressed.
        self.planned_path = None

        # initialize 
        self.go_to_home(pose = "poseA")
        self.print_options()
        self.loop()
    

    def color_image_cb(self , data):
        """ callback function that receives images in data. 
            pass it through cv_bring to convert Image -> NumPy array 
            resize that array 
            save one variable for RTP task (in RGB) format
            save one variable to show on the window (in BGR)
        """
        self.img_msg_received = True
        self.img_msg = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        w,h,c = self.img_msg.shape 
        self.img_msg = cv2.resize( self.img_msg, (256, 256) )
        self.img_shown = cv2.cvtColor(self.img_msg, cv2.COLOR_RGB2BGR )
        self.img_msg_rtp = self.img_msg.copy()

    def go_to_home(self, pose="nippleA"):
        """ this function takes the panda_arm group to named pose 
            which are saved under pnad_moveit_config/config folders. 
            We also set the velocity scaling to 10% so that the robot 
            does move slow! (SAFETY FIRST, FUN LATER!) 
        """
        rospy.loginfo ("Moving to home pose. Please wait...")
        self.group.set_max_velocity_scaling_factor(0.1)
        self.group.set_named_target(pose)
        self.group.go(wait=True)
        rospy.loginfo ("Moved to home pose...")
        time.sleep(1)
             
    def get_forward_kinematics(self, joints):
        """ this function basically solves the forward kinematics. 
            You will notice that we make use of URDF format. 
        """
        q = joints
        pose = self.kdl_kin.forward(q)
        trans = pose[0:3, 3].T
        rot = quaternion_from_matrix(pose)
        return trans.tolist()[0], rot.tolist()

    def move_robot_to_joint_trajectory(self, joint_trajectory):

        """ this is the function that you will have to change. 
            We get the predicted joint trajectory as a vector 
            of 1050 elements (150 points for each joint). This 
            trajectory can be reshapes from (1050, ) -> (150, 7). 

            You will need to use the forward kinematics function 
            as well as the robot MoveIt group to convert the 
            joint trajectory back into the cartesian trajectory. 

            Then you will need to pass this cartesian trajectory 
            to MoveIt and show the display the plan in RViZ. 

            Then execute this trajectory.  

            For easier execution, save the generated trajectory 
            from compute_cartesian_path path of MoveGroup into a
            class variable so you can use it in do_execute_planned_path. 
        """ 

        q1pred,q2pred,q3pred,q4pred,q5pred,q6pred,q7pred = joint_trajectory[0:150], joint_trajectory[150:300], joint_trajectory[300:450], joint_trajectory[450:600], joint_trajectory[600:750], joint_trajectory[750:900], joint_trajectory[900:1050]
        # nipple A
        nippleA_pose = [-0.0543767, 0.471072, 0.171324, -2.03379, -0.0378312, 2.4434, 0.951624]
        is_start_pose = True

        # cartesian space path plan 
        waypoints = []
        wpose = self.group.get_current_pose().pose
        

        
    def do_execute_planned_path(self):
        if (self.planned_path != None):
            self.group.execute(self.planned_path, wait=True)

    def print_options(self):
        print("\n  Click on image for target point and then select one of the options, ")
        print("  1) press s to save image and target point. ")
        print("  2) press r to predict the RTP trajectory. ")
        print("  3) press p to display the predicted trajectory. ")
        print("  4) press e to execute the predicted trajectory")
        print("  5) Press h to take robot to home pose. ")
        print("  6) Press ESC to close" )    

    def loop(self):
        """ this loop function continously run 
            and shows the image window. 

            It also listens the key presses on the 
            image window. 

            As you can see that when the user clicks 'p' 
            the predicted joint trajectory is read from 
            the npy file (that was written there by 
            a Python3 script that uses RTP task trained model.)
            
        """
        while not rospy.is_shutdown(): 
            if self.img_msg_received:

                w,h,c = self.img_msg.shape 
                if w == 256 and h == 256 :
                    cv2.imshow('image', self.img_shown)

                k = cv2.waitKey(33)
                if k==27:    # Esc key to stop
                    break
                elif k==-1:  # normally -1 returned,so don't print it
                    continue
                else:
                    self.print_options()
                    if k == ord('s'): 
                        cv2.imwrite( os.path.join( self.BASE_PATH, 'image_rtp.png' ) , self.img_msg_rtp)
                        np.save( os.path.join( self.BASE_PATH , 'image_xy_rtp.npy' ) , np.reshape( self.img_msg_rtp, (1,256,256, 3) ) )
                        
                    if k == ord('r'):
                        command = ["python3", "/home/arshad/franka_ws/src/franka_ros_lcas/franka_lcas_experiments/script/load_model_rtp.py"]
                        subprocess.call(command)
 
                    if k == ord('h'):
                        self.go_to_home(pose="poseA")

                    if k == ord('p'):
                        joint_trajectory_predicted = np.load( os.path.join( self.BASE_PATH, 'predicted_joints_values_rtp.npy') )
                        self.move_robot_to_joint_trajectory(joint_trajectory_predicted)

                    if k == ord('e'):
                        self.do_execute_planned_path()

        cv2.destroyAllWindows()

if __name__ == "__main__": 

    reachToPalpateTaskValidation = ReachToPalpateTaskValidation()
