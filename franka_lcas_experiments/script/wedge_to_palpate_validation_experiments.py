#!/usr/bin/env python

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import time
import subprocess, shlex, psutil
from cv_bridge import CvBridge
import cv2 
import imutils
import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import pandas as pd


# for forward kinematics
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
from tf.transformations import quaternion_from_matrix

# for point cloud processing 
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import tf2_py as tf2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import ros_numpy

class WedgeToPalpateValidation: 

    def __init__(self): 

        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('validation_wedge_to_palpate_node', anonymous=True)
        self.robot = moveit_commander.RobotCommander()
        group_name = "panda_arm"
        self.group = moveit_commander.MoveGroupCommander(group_name)
        self.scene = moveit_commander.PlanningSceneInterface()
        self.display_trajectory_publisher = rospy.Publisher( "/move_group/display_planned_path", moveit_msgs.msg.DisplayTrajectory, queue_size=20)

        self.bridge = CvBridge()

        self.trajectory_pub = rospy.Publisher('/position_joint_trajectory_controller/command', JointTrajectory, queue_size = 10)

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
 
        self.current_controller = "position"
        self.last_controller = "franka_zero_torque_controller"

        # recording rosbag proces 
        self.rosbag_proc = None
        self.is_recording = False

        self.imgX = None
        self.imgY = None
        self.drawXY = True

        # setup for forward kinematics 
        robot = URDF.from_parameter_server()
        tree = kdl_tree_from_urdf_model(robot)
        base_link, end_link = "panda_link0", "xela_sensor_frame"
        self.kdl_kin = KDLKinematics(robot, base_link, end_link)

        # get_pointcloud data 
        self.do_acquire_pc = False
        self.pointcloud = np.array([])
        self.pointcloud_sub = rospy.Subscriber('/depth/color/points', PointCloud2, self.pointcloud_cb)

        self.loop()

    def select_point_cb(self, event,x,y,flags,param):
        if self.img_msg_received and self.img_info_msg_received and self.depth_msg_received:
            if event == cv2.EVENT_LBUTTONDOWN: # Left mouse button
                # rospy.loginfo("Mouse event: {} {}".format(x,y))
                self.imgX = x
                self.imgY = y
 
    def depth_image_cb(self, data):
        self.depth_msg_received = True
        # The depth image is a single-channel float32 image
        # the values is the distance in mm in z axis
        self.depth_msg = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        
    def color_camear_info_cb(self, data):
        self.img_info_msg_received = True
        self.img_info_msg = data

    def pointcloud_cb(self, msg):

        if (self.do_acquire_pc):     
            target_frame = "panda_link0"
            timeout = 0.0
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
            rospy.logwarn("Please wait 5 seconds for the tf")
            time.sleep(5)
            while True : 
                try:
                    trans = self.tf_buffer.lookup_transform(target_frame, msg.header.frame_id, rospy.Time()) #msg.header.stamp, rospy.Duration(timeout))
                    cloud_out = do_transform_cloud(msg, trans)
                    # print(cloud_out.header, msg.header , trans)
                    self.pointcloud = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(cloud_out)
                    break
                except tf2.LookupException as ex:
                    rospy.logwarn(ex)
                    break
                except tf2.ExtrapolationException as ex:
                    rospy.logwarn(ex)
                    break
            

        self.do_acquire_pc = False


    def color_image_cb(self , data):
        
        self.img_msg_received = True
        self.img_msg = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        # self.img_msg = cv2.cvtColor(img_msg, cv2.COLOR_RGB2BGR )
        w,h,c = self.img_msg.shape 

        self.img_msg = self.img_msg[:, int( (h-w) / 2):int ( h - (h-w)/2 ), :]
        self.img_msg = imutils.resize( self.img_msg, width = 256 )

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

    def go_to_home(self, pose="nippleA"):
        if not self.is_recording:
            rospy.loginfo ("Moving to home pose. Please wait...")
            self.group.set_max_velocity_scaling_factor(0.1)
            self.group.set_named_target(pose)
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

    def get_forward_kinematics(self, joints):

        q = joints
        # forward kinematics (returns homogeneous 4x4 numpy.mat)
        pose = self.kdl_kin.forward(q)

        trans = pose[0:3, 3].T
        rot = quaternion_from_matrix(pose)
        return trans.tolist()[0], rot.tolist()
        # print (trans.tolist()[0], rot.tolist())

    def move_robot_to_joint_trajectory(self, traj_true):
        
        self.do_acquire_pc = True
        while self.do_acquire_pc: 
            pass 


        if (self.pointcloud.shape[0] > 0):
            print (self.pointcloud.shape)

        q1pred,q2pred,q3pred,q4pred,q5pred,q6pred,q7pred = traj_true[0:150], traj_true[150:300], traj_true[300:450], traj_true[450:600], traj_true[600:750], traj_true[750:900], traj_true[900:1050]
        # self.go_to_home(pose="nippleA")

        # joint space path plan
        msg = JointTrajectory()
        msg.header.stamp = rospy.Duration( 0 ) #rospy.Time().now()
        msg.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4' , 'panda_joint5', 'panda_joint6', 'panda_joint7']

        # cartesian space path plan 
        waypoints = []
        wpose = self.group.get_current_pose().pose
        

        dt = 0.1
        dx = 0.0025
        dy = 0.0025

        for i in range(q1pred.shape[0] - 50):
            _joints_all = [q1pred[i],q2pred[i],q3pred[i],q4pred[i],q5pred[i],q6pred[i],q7pred[i]]
            point = JointTrajectoryPoint()            
            point.positions = _joints_all 
            point.time_from_start = rospy.Duration( i * dt + 1 )
            msg.points.append( point ) 

            trans, rot = self.get_forward_kinematics ( _joints_all )            
            wpose.position.x = trans[0]
            wpose.position.y = trans[1]

            # z_index = np.where( (self.pointcloud[:, 0] > (trans[0]+dx)) & (self.pointcloud[:,0] < (trans[0] - dx)) )
            z_index = np.where( 
                (self.pointcloud[:, 0] > (trans[0] - dx )) & 
                (self.pointcloud[:, 0] < (trans[0] + dx )) & 
                (self.pointcloud[:, 1] > (trans[1] - dy )) & 
                (self.pointcloud[:, 1] < (trans[1] + dy )) )

            all_z_points_on_breast = (self.pointcloud[z_index])[:,2]

            print ( np.mean(all_z_points_on_breast), np.std(all_z_points_on_breast), np.median(all_z_points_on_breast), trans[2] , z_index[0].shape)

            if (all_z_points_on_breast.shape[0] > 0):
                wpose.position.z = np.median(all_z_points_on_breast) #- 0.0075

            else:
                if trans[2] < 0.085: 
                    trans[2] = 0.085

                wpose.position.z = trans[2]                    

            wpose.orientation.x = rot[0]
            wpose.orientation.y = rot[1]
            wpose.orientation.z = rot[2]
            wpose.orientation.w = rot[3]
            
            # print ("Z Axis: " , trans[2])
            waypoints.append( copy.deepcopy(wpose) )
    
        # make cartesian path planning 
        plan, fraction = self.group.compute_cartesian_path( waypoints, 0.01, 0.0 ) 

        replan = self.group.retime_trajectory( self.robot.get_current_state(), plan, velocity_scaling_factor = 0.05) 

        print("\nCartesian plan fraction is {}".format(fraction))
        # display cartesian path 
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(replan)
        # Publish
        self.display_trajectory_publisher.publish(display_trajectory)

        decision = raw_input("\n---------------------------\nPress 'y' to execute the trajectory and 'n' to cancel it\n----------------------\n")
        if (decision.lower() == 'y'):
            self.group.execute(replan, wait=True)
        
        # display joint trajecotry
        # robot_traj_complete = moveit_msgs.msg.RobotTrajectory()
        # robot_traj_complete.joint_trajectory =  msg 
        # display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        # display_trajectory.trajectory_start = self.robot.get_current_state()
        # display_trajectory.trajectory.append( robot_traj_complete )
        # # Publish
        # self.display_trajectory_publisher.publish(display_trajectory)
        

        # self.go_to_home()
        # rospy.loginfo("waiting...")
        # self.trajectory_pub.publish(msg) 
        

    def loop(self):

        while not rospy.is_shutdown(): 
            # pass
            if self.img_msg_received and self.depth_msg_received:
                w,h,c = self.img_msg.shape 
                if self.drawXY and self.imgX != None : 
                    cv2.circle( self.img_msg, (self.imgX, self.imgY), 10, (40,48,88), -1)

                if w == 256 and h == 256 :
                    cv2.imshow('image', self.img_msg)

                k = cv2.waitKey(33)
                if k==27:    # Esc key to stop
                    break
                elif k==-1:  # normally -1 returned,so don't print it
                    continue
                else:
                    if k == ord('c'):
                        self.drawXY = not self.drawXY
                    if k == ord('s'): 
                        rospy.loginfo("Writing image to a file in catkin_ws/ folder with X: {} and Y: {}".format(self.imgX, self.imgY))
                        cv2.circle( self.img_msg, (self.imgX, self.imgY), 10, (40,48,88), -1)
            
                        cv2.imwrite('./image.png', self.img_msg)
                        saved_img = np.reshape( self.img_msg, (1,256,256,3))
                        np.save( './image_xy.npy', saved_img )
                        target_point = np.array( [self.imgX, self.imgY ] ) 
                        np.save('./image_target_xy.npy', target_point )

                    if k == ord('r'):
                        rospy.loginfo("Activating the virtual environment")
                        command = ["python3", "/home/arshad/catkin_ws/src/franka_ros_lcas/franka_lcas_experiments/script/load_model.py"]
                        subprocess.call(command)
                        print("\n\nSelect another point on image, \n 1) press s to save.\n 2) press r to predict. \n 3) press p to display. \n 4) After display, click 'y' on terminal to execute the trajectory. \n 6) Press h to take robot to home pose. \n 6) Press ESC to close")    

                    if k == ord('h'):
                        self.go_to_home(pose="poseA")

                    if k == ord('p'):
                        
                        # _data = pd.read_csv( '/home/arshad/catkin_ws/experiment10_relu_7_traj_data.csv', skipinitialspace=True )                        
                        # traj_true = _data['q_predictions'].values                   
                        # print (traj_true.shape)

                        traj_true = np.load('/home/arshad/catkin_ws/predicted_joints_values.npy')
                        self.move_robot_to_joint_trajectory(traj_true)


        cv2.destroyAllWindows()

if __name__ == "__main__": 

    validation_exp = WedgeToPalpateValidation()
