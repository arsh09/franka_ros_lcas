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

class FrankaMoveItTutorials: 

    def __init__(self): 
        """ - init will start the move group for the arm 
        """

        # MoveIt Python client 
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('advance_robotics_workshop_ex_1_node', anonymous=True)
        
        # init the group (arm). 
        group_name = "panda_arm"
        self.group = moveit_commander.MoveGroupCommander(group_name)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.display_trajectory_publisher = rospy.Publisher( "/move_group/display_planned_path", DisplayTrajectory, queue_size=20)
        
        # velocity and acceleration scaling 
        # for all the motion commands. 
        self.group.set_max_velocity_scaling_factor( 0.1 ) 
        self.group.set_max_acceleration_scaling_factor( 0.25 ) 

        self.planned_path = None

        # take the robot to 'ready' pose. 
        self.group.set_named_target("ready" ) 
        self.group.go(wait = True) 


    def send_a_joint_goal(self): 
        ''' use this function to send a joint goal. 
            Note that a single joint goal is different 
            from a joint trajectory. A joint trajectory
            contains multiple joint goals (think of 
            multiple points on a single line with start 
            and end points as joint trajectory and joint
            goal is only the end point).
        '''
        # We get the joint values from the group and change some of the values:
        joint_goal = self.group.get_current_joint_values()
        joint_goal[0] -= 0.1 
        
        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        self.group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        self.group.stop()


    def send_a_pose_goal(self): 
        ''' pose goal is similar to joint goal 
            but in cartesian space (or task space). 
            The difference between cartesian trajectory 
            and pose goal is similar to joint goal 
            and joint trajectory.
        '''
        # We get the joint values from the group and change some of the values:
        pose_goal = self.group.get_current_pose()
        pose_goal.pose.position.x += 0.1 

        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        self.group.go(pose_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        self.group.stop()

    def plan_and_display_cartesian_trajectory(self): 
        ''' this function sends the cartesian 
            trajectory to the robot. 

            One of the advantage of the cartesian 
            trajectory over pose_goal is that 
            you can have kinda via points that the 
            robot must pass through. 

            Please note that we use the moveit 
            simple path planner here.
        '''
        
        waypoints = []
        wpose = self.group.get_current_pose().pose
        wpose.position.z -=  0.1  
        wpose.position.y +=  0.2  
        waypoints.append(copy.deepcopy(wpose))

        wpose.position.x += 0.1 
        waypoints.append(copy.deepcopy(wpose))

        wpose.position.y -= 0.1  
        waypoints.append(copy.deepcopy(wpose))

        # We want the Cartesian path to be interpolated at a resolution of 1 cm
        # which is why we will specify 0.01 as the eef_step in Cartesian
        # translation.  We will disable the jump threshold by setting it to 0.0,
        # ignoring the check for infeasible jumps in joint space, which is sufficient
        # for this tutorial.
        (plan, fraction) = self.group.compute_cartesian_path(
            waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
        )  # jump_threshold

        
        # retime the trajectory so its smoother and moves at 0.1-velocity scale.
        plan = self.group.retime_trajectory(self.robot.get_current_state(), plan, 0.1)

        # display a cartsian trajectory in RViZ 
        display_trajectory = DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        # Publish
        self.display_trajectory_publisher.publish(display_trajectory)

        self.planned_path = plan


    def execute_planned_path(self): 
        ''' execute the planned cartesian path on the robot. 
        '''

        if not isinstance( self.planned_path , type(None) ): 
            self.group.execute( self.planned_path, wait=True)


if __name__ == "__main__": 

    frankaMoveItTutorials = FrankaMoveItTutorials()

    frankaMoveItTutorials.plan_and_display_cartesian_trajectory()
    frankaMoveItTutorials.execute_planned_path()
    