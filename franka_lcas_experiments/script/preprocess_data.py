import rospy
import rosbag
import numpy as np
import matplotlib.pyplot as plt


for i in range(7, 210, 7):

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

    fig = plt.figure(1, figsize=(10,6))
    # plt.subplot(2,1,1)
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

    plt.show()