#! /usr/bin/env python
import rospy

import numpy as np
import os
import rosbag 
import json
import time

import cv2
from cv_bridge import CvBridge
import matplotlib.pyplot as plt

import ros_numpy as ros_np
# import pptk
# import open3d as o3d
# import sensor_msgs.point_cloud2 as pc2

# for forward kinematics
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
from tf.transformations import quaternion_from_matrix

# for pointcloud2 msg to PCD 
import subprocess, shlex, psutil

# for plotting 2D point of eef (in home pose)
import matplotlib.pyplot as plt


def export_joint_and_cartesian_to_json():

    # this function requires that URDF is uploaded 
    # to ros param server so that FK can be solved.

    rospy.init_node('preprocess_data_trajectories')
    for i in range(1,308):

        bag_path = "/home/arshad/experiment_" + str(i) + ".bag"
        if (os.path.isfile(bag_path)):

            bag = rosbag.Bag(bag_path)

            json_path = "/media/arshad/Samsung_T5/process_data_zone_D/experiment_" + str(i) + ".json"
            json_data = dict()
            json_data['joint_position'] = []
            json_data['joint_speed'] = []
            json_data['joint_torque'] = []
            json_data['time'] = []
            json_data['base_to_eef_position'] = []
            json_data['base_to_eef_orientation'] = []

            robot = URDF.from_parameter_server()
            tree = kdl_tree_from_urdf_model(robot)
            base_link, end_link = "panda_link0", "xela_sensor_frame"
            kdl_kin = KDLKinematics(robot, base_link, end_link)


            for topic, msg, t in bag.read_messages(topics=["/joint_states"]):    
                json_data['time'].append( msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9 )
                json_data['joint_position'].append( msg.position ) 
                json_data['joint_speed'].append( msg.velocity ) 
                json_data['joint_torque'].append( msg.effort ) 

                q = msg.position[0:7]
                pose = kdl_kin.forward(q) # forward kinematics (returns homogeneous 4x4 numpy.mat)
                
                trans = pose[0:3, 3].T
                rot = quaternion_from_matrix(pose)
                # print (trans.tolist()[0], rot.tolist())

                json_data["base_to_eef_position"].append( trans.tolist()[0] ) 
                json_data["base_to_eef_orientation"].append( rot.tolist() ) 

            # subtract first time sample for all
            json_data['time'] = [ sample - json_data['time'][0] for sample in json_data['time']]

            with open(json_path, 'w') as write_file:
                json.dump(json_data, write_file, ensure_ascii=False, indent=4)

            rospy.loginfo("Done: {}".format(bag_path))



def export_color_and_depth_images():

    for i in range(308):

        bag_path = "/home/arshad/experiment_" + str(i) + ".bag"
        color_path = "/media/arshad/Samsung_T5/process_data_zone_D/experiment_" + str(i) + "_color.npy"
        depth_path = "/media/arshad/Samsung_T5/process_data_zone_D/experiment_" + str(i) + "_depth.npy"


        if (os.path.isfile(bag_path)):
            bag = rosbag.Bag(bag_path)
            
        # get color images
            bridge = CvBridge()
            for topic, msg, t in bag.read_messages(topics=["/color/image_raw"]):
                color_img = bridge.imgmsg_to_cv2( msg )
                np.save( color_path, color_img)     
                break   

            # get depth images
            bridge = CvBridge()
            for topic, msg, t in bag.read_messages(topics=["/depth/image_rect_raw"]):
                depth_img = bridge.imgmsg_to_cv2( msg , "32FC1")
                np.save( depth_path, depth_img)       
                break 

            print ("Done: {}".format(bag_path))

def export_point_cloud_data():

    for i in range(1,307):
    # i = 1
        bag_path = "/home/arshad/experiment_" + str(i) + ".bag"
        pst_path = "/media/arshad/Samsung_T5/process_data_zone_D/experiment_" + str(i) + "_pointcloud.npy"

        if os.path.isfile(bag_path): 
            bag = rosbag.Bag(bag_path)

            for topic, msg, t in bag.read_messages(topics=["/depth/color/points"]):
                
                pc_arr = ros_np.point_cloud2.pointcloud2_to_array( msg )
                pc_xyz = ros_np.point_cloud2.get_xyz_points( pc_arr )
                pc_all = ros_np.point_cloud2.split_rgb_field( pc_arr )
                pc_rbg = []

                for row in pc_all:
                    r, g, b = row[-3], row[-2], row[-1]
                    pc_rbg.append( [r, g, b ] )

                pst = np.hstack( (pc_xyz, pc_rbg)  )                
                np.save( pst_path , pst )
                break 
            
            print ("Done: {}".format(bag_path))
            

def set_plot_properties():

    import matplotlib as mpl
    mpl.rcParams['legend.markerscale'] = 2.0 
    mpl.rcParams['font.size'] = 20
    mpl.rcParams['lines.markersize'] = 20.0


def plot_2d_end_points():

    # range on XY plane for world (From base)
    # 0.309809791327 -0.260155091999 0.116137658563
    # 0.682918533702 0.216092091272 0.0953567279476
    # 0.685449990365 -0.231814252613 0.0927342125951

    start_points = []   
    end_points = []
    for i in range(1,308):

        json_path = "/media/arshad/Samsung_T5/process_data_zone_D/experiment_" + str(i) + ".json"

        if (os.path.isfile(json_path)):
            # get last point
            with open(json_path, 'r') as read_file:
                data = json.load( read_file )
                start_points.append( data['base_to_eef_position'][0] )
                end_points.append( data['base_to_eef_position'][-1] )

    start_points = np.array( start_points )
    end_points = np.array( end_points )
    
    set_plot_properties()
    fig = plt.figure(1, figsize=(16,10))
    plt.scatter( end_points[:,0], end_points[:,1], label = 'Stop Point', c = 'blue', s = 70)
    plt.scatter( start_points[:,0], start_points[:,1], label = 'Start Point', c = 'green', s = 70)
    plt.xlabel('X (in m)')
    plt.ylabel('Y (in m)')
    plt.xlim([0.2, 0.8])
    plt.ylim([-0.3, 0.3])
    plt.legend()
    plt.title('Cartesian XY w.r.t base')
    plt.show()


def plot_2d_end_points_for_all_zones():

    # range on XY plane for world (From base)
    # 0.309809791327 -0.260155091999 0.116137658563
    # 0.682918533702 0.216092091272 0.0953567279476
    # 0.685449990365 -0.231814252613 0.0927342125951
    
    zones = ['A', 'B', 'C', 'D'] 
    color_end = ['red', 'green', 'yellow', 'brown']
    for count, zone in enumerate(zones): 
        start_points = []   
        end_points = []
        for i in range(1,308):

            json_path = "/media/arshad/Samsung_T5/process_data_zone_" + zone + "/experiment_" + str(i) + ".json"

            if (os.path.isfile(json_path)):
                # get last point
                with open(json_path, 'r') as read_file:
                    data = json.load( read_file )
                    start_points.append( data['base_to_eef_position'][0] )
                    end_points.append( data['base_to_eef_position'][-1] )

        start_points = np.array( start_points )
        end_points = np.array( end_points )
        
        set_plot_properties()
        fig = plt.figure(1, figsize=(16,10))
        plt.scatter( end_points[:,0], end_points[:,1], label = 'StopPoint Zone: ' + zone  , c = color_end[count], s = 70)
        plt.scatter( start_points[:,0], start_points[:,1], label = 'Start Point', c = 'black', s = 70)



    plt.xlabel('X (in m)')
    plt.ylabel('Y (in m)')
    plt.xlim([0.2, 0.8])
    plt.ylim([-0.3, 0.3])
    # plt.legend()
    plt.title('Cartesian XY w.r.t base')
    plt.show()


def plot_color_and_depth_images():

    color_img = np.zeros((640, 480, 3))
    depth_img = np.zeros((640, 480, 3))
    try: 

        for i in range(1,308):
            color_path = "/media/arshad/Samsung_T5/process_data_zone_D/experiment_" + str(i) + "_color.npy"
            depth_path = "/media/arshad/Samsung_T5/process_data_zone_D/experiment_" + str(i) + "_depth.npy"

            if (os.path.isfile(color_path)):
                color_img = np.load(color_path)
            if (os.path.isfile(depth_path)):
                depth_img = np.load(depth_path)
            

            cv2.imshow('color', color_img)
            cv2.imshow('DEPTH', depth_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except: 
        cv2.destroyAllWindows()


def plot_point_cloud():

    pointcloud = None
    for i in range(1,308):
        pc_path = "/media/arshad/Samsung_T5/process_data_zone_D/experiment_" + str(i) + "_pointcloud.npy"

        if (os.path.isfile(pc_path)):
            pointcloud = np.load(pc_path)


            print (pointcloud.shape)

if __name__ == '__main__':

    # export_point_cloud_data()
    # export_color_and_depth_images()
    # export_joint_and_cartesian_to_json()

    plot_2d_end_points_for_all_zones()
    # plot_2d_end_points()
    # plot_color_and_depth_images()
    # plot_point_cloud()
