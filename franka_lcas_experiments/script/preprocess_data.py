import rospy
import rosbag
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

for i in range(4,217,7):

    filepath = "/home/arshad/data/config_B/experiment_" + str(i) + ".bag"
    print "Reading: {}".format(filepath)
    bagfile = rosbag.Bag(filepath)

    all_joints = []
    all_joints_t = []
    for topic, msg, t in bagfile.read_messages(topics=["/joint_states"]):
        
        all_joints_t.append( msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9 )
        all_joints.append( msg.position ) 

    all_joints_t = np.array(all_joints_t)
    all_joints_t = all_joints_t - all_joints_t[0]
    all_joints = np.array(all_joints)

    all_point_t = []
    all_point = []
    for topic, msg, t in bagfile.read_messages(topics=["/xServTopic"]):
        
        all_point_t.append( msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9 )
        _temp = []
        for p in msg.points: 
            _temp.append( [p.point.x, p.point.y, p.point.z] )
        
        all_point.append(_temp)

    all_point = np.array( all_point )
    all_point_t = np.array( all_point_t )
    all_point_t = all_point_t - all_point_t[0]


    fig = plt.figure(1, figsize=(12,8))
    plt.subplot(2,2,1)
    plt.plot(all_joints_t, all_joints[:,0], 'b--', label="J1")
    plt.plot(all_joints_t, all_joints[:,1], 'r--', label="J2")
    plt.plot(all_joints_t, all_joints[:,2], 'g--', label="J3")
    plt.plot(all_joints_t, all_joints[:,3], 'm--', label="J4")
    plt.plot(all_joints_t, all_joints[:,4], 'c--', label="J5")
    plt.plot(all_joints_t, all_joints[:,5], 'y--', label="J6")
    plt.plot(all_joints_t, all_joints[:,6], 'k--', label="J7")
    plt.xlabel("Time (in secs)")
    plt.ylabel("Position (in rads)")
    plt.legend()
    plt.grid(True)



    for i in range(all_point.shape[1]):
        plt.subplot(2,2,2)
        plt.plot(all_point_t, all_point[:, i, 0], 'b--', label="T" + str(i+1) + "_x")
        plt.grid(True)
        plt.xlabel("Time (in secs)")
        plt.ylabel("Taxel X")

    for i in range(all_point.shape[1]):
        plt.subplot(2,2,3)
        plt.plot(all_point_t, all_point[:, i, 1], 'r--', label="T" + str(i+1) + "_y")
        plt.grid(True)
        plt.xlabel("Time (in secs)")
        plt.ylabel("Taxel Y")

    for i in range(all_point.shape[1]):

        plt.subplot(2,2,4)
        plt.plot(all_point_t, all_point[:, i, 2], 'g--', label="T" + str(i+1) + "_z")
        plt.grid(True)
        plt.xlabel("Time (in secs)")
        plt.ylabel("Taxel Z")

    plt.show()
