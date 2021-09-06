from __future__ import division
import numpy as np
import numpy.matlib as mat
import tensorflow as tf
# from skimage.io import imread
# from skimage.transform import resize
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from natsort import natsorted
import os, json, csv, math
import pandas as pd
from glob import glob
import matplotlib.patches as mpatches

JOINT_NUM = 7
JOINT_WEIGHTS = 8
CARTESIAN_NUM = 3
CARTESIAN_WEIGHTS = 3

# current_dir = os.getcwd()
# PATH = current_dir + "/data/json/"

# def return_coordinates(plot=True):
#     json_files = [pos_json for pos_json in os.listdir(PATH) if pos_json.endswith('.json')]
#     json_files = natsorted(json_files)

#     # we need both the json and an index number so use enumerate()
#     # y - 0.15, 0.12
#     # x - 0.36, 0.48
#     all_x = []
#     all_y = []
#     non_sparse_pos = []
#     for index, js in enumerate(json_files):
#         with open(os.path.join(PATH, js)) as json_file:
#             json_text = json.load(json_file)
#             x = json_text['transf_matrix'][-1][0][-1]
#             y = json_text['transf_matrix'][-1][1][-1]

#             # if (-0.15 <= y <= 0.12) and (0.36 <= x <= 0.48):
#             non_sparse_pos.append(index)
#             all_x.append(x)
#             all_y.append(y)

#     if plot == True:
#         return non_sparse_pos, all_x, all_y
#     else:
#         paired = [[m,n] for m,n in zip(all_x,all_y)]
#         return non_sparse_pos, paired

# def plot_2d():
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111)
#     _, all_x, all_y = return_coordinates(plot=True)
#     ax1.scatter(all_x, all_y, s=10, c='b', marker="o", label='XY EE Plot')
#     plt.xlabel("X", fontweight ='bold', size=14)
#     plt.ylabel("Y", fontweight ='bold', size=14)

#     d1 = mpatches.Circle((0.425, -0.03), 0.07,alpha=0.5,facecolor="red")
#     d2 = mpatches.Circle((0.55, 0.08), 0.05,alpha=0.5,facecolor="green")
#     d3 = mpatches.Circle((0.55, -0.1), 0.04,alpha=0.5,facecolor="blue")
#     d4 = mpatches.Circle((0.40, -0.23), 0.07, alpha=0.5,facecolor="purple")

#     # left1, bottom1, width1, height1 = (0.36, -0.10, 0.10, 0.13)
#     # rect1 = mpatches.Rectangle((left1,bottom1),width1,height1,alpha=0.5,facecolor="red",linewidth=2)
#     #
#     # left2, bottom2, width2, height2 = (0.51, 0.05, 0.06, 0.1)
#     # rect2 = mpatches.Rectangle((left2,bottom2),width2,height2,alpha=0.5,facecolor="green",linewidth=2)
#     #
#     # left3, bottom3, width3, height3 = (0.53, -0.13, 0.06, 0.07)
#     # rect3 = mpatches.Rectangle((left3,bottom3),width3,height3,alpha=0.5,facecolor="blue",linewidth=2)
#     #
#     # left4, bottom4, width4, height4 = (0.365, -0.28, 0.07, 0.12)
#     # rect4 = mpatches.Rectangle((left4,bottom4),width4,height4,alpha=0.5,facecolor="purple",linewidth=2)

#     legend1 = plt.legend(loc='upper right')
#     plt.gca().add_patch(d1)
#     plt.gca().add_patch(d2)
#     plt.gca().add_patch(d3)
#     plt.gca().add_patch(d4)

#     legend2 = plt.legend((d1, d2, d3, d4), ('Region 1', 'Region 2', 'Region 3', 'Region 4'), prop={'size': 6}, loc='lower right')
#     plt.gca().add_artist(legend1)
#     plt.gca().add_artist(legend2)
#     plt.savefig(current_dir + '/2D xy circle.png')

# def euler2quaternion(r, p, y):
#     ci = math.cos(r/2.0)
#     si = math.sin(r/2.0)
#     cj = math.cos(p/2.0)
#     sj = math.sin(p/2.0)
#     ck = math.cos(y/2.0)
#     sk = math.sin(y/2.0)
#     cc = ci*ck
#     cs = ci*sk
#     sc = si*ck
#     ss = si*sk
#     return [cj*sc-sj*cs, cj*ss + sj*cc, cj*cs - sj*sc, cj*cc + sj*ss]

# def Trpyxyz(r, p, y, X, Y, Z):
# 	q = euler2quaternion(r, p, y)
# 	qi = q[0]
# 	qj = q[1]
# 	qk = q[2]
# 	qr = q[3]
# 	qi = qi/math.sqrt(qi*qi + qj*qj + qk*qk + qr*qr)
# 	qj = qj/math.sqrt(qi*qi + qj*qj + qk*qk + qr*qr)
# 	qk = qk/math.sqrt(qi*qi + qj*qj + qk*qk + qr*qr)
# 	qr = qr/math.sqrt(qi*qi + qj*qj + qk*qk + qr*qr)
# 	T = np.array([[1-2*(qj*qj+qk*qk), 2*(qi*qj-qk*qr), 2*(qi*qk+qj*qr),X],
# 	      [2*(qi*qj+qk*qr), 1-2*(qi*qi+qk*qk), 2*(qj*qk-qi*qr),Y],
# 	      [2*(qi*qk-qj*qr), 2*(qj*qk+qi*qr), 1-2*(qi*qi+qj*qj),Z],
# 	      [0, 0, 0, 1]])
# 	return T

# def Trz(theta):
# 	T = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
# 		[np.sin(theta), np.cos(theta), 0, 0],
# 		[0, 0, 1, 0],
# 		[0, 0, 0, 1]])
# 	return T

# def FK(q,q2,q3,q4,q5,q6,q7):
#     # Computes the transformation matrix of the end-effector for a given configuration (MoveIt frame)
# 	# values from URDF file https://github.com/StanfordASL/PandaRobot.jl/blob/master/deps/Panda/panda.urdf
# 	T01 = np.matmul(Trpyxyz(0, 0, 0, 0, 0, 0.333),Trz(q))
# 	T12 = np.matmul(Trpyxyz(-1.57079632679, 0, 0, 0, 0, 0),Trz(q2))
# 	T23 = np.matmul(Trpyxyz(1.57079632679, 0, 0, 0, -0.316, 0),Trz(q3))
# 	T34 = np.matmul(Trpyxyz(1.57079632679, 0, 0, 0.0825, 0, 0),Trz(q4))
# 	T45 = np.matmul(Trpyxyz(-1.57079632679, 0, 0, -0.0825, 0.384, 0),Trz(q5))
# 	T56 = np.matmul(Trpyxyz(1.57079632679, 0, 0, 0, 0, 0),Trz(q6))
# 	T67 = np.matmul(Trpyxyz(1.57079632679, 0, 0, 0.088, 0, 0),Trz(q7))
# 	T78 = Trpyxyz(0, 0, 0, 0, 0, 0.107)
# 	T08 = np.matmul(T01, np.matmul(T12 ,np.matmul(T23, np.matmul(T34, np.matmul(T45, np.matmul(T56, np.matmul(T67, T78)))))))
# 	return T08

# def normalize_negative_one(img, twod=True):
#     normalized_input = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
#     if twod == True:
#         return 2*normalized_input - 1
#     else:
#         return np.amin(img), np.amax(img), 2*normalized_input - 1

# def reversed_normalize_negative_one(img, min, max):
#     po = (img + 1) /2
#     reversed_input = ((max - min) * po) + min
#     return reversed_input

# def load_images(data_dir):
#     images = glob(data_dir+ '*.*')
#     images = natsorted(images)
#     allImages = []

#     for index, filename in enumerate(images):
#         img = imread(filename, pilmode='RGB')
#         # resizing image
#         img = resize(img, (256, 256, 3))
#         # img = normalize_negative_one(img)
#         allImages.append(img)

#     allImages = np.array(allImages)
#     return allImages


# def load_images_regions(data_dir):
#     images = glob(data_dir+ '*.*')
#     images = natsorted(images)
#     allImages = []
#     allRegions = []

#     for index, filename in enumerate(images):
#         img = imread(filename, pilmode='RGB')
#         # resizing image
#         img = resize(img, (256, 256, 3))
#         # img = normalize_negative_one(img)
#         allImages.append(img)
#         # The first character of the file name identifies the region where the sample was collected.
#         region_id = os.path.basename(filename)[0]
#         allRegions.append(region_id)

#     allImages = np.array(allImages)
#     allRegions = np.array(allRegions)
#     return allImages, allRegions


# def all_joint_weights():
#     weights = pd.read_csv('/home/admin_new/workspace/artemis/data/weightsalljoints.csv', header=None)

#     all_weights = []
#     for index, row in weights.iterrows():

#         all_weights.append(row.values)
#         assert(all_weights[0].shape[0] == JOINT_WEIGHTS * JOINT_NUM)

#     assert(len(all_weights) == 200)
#     return np.vstack((all_weights))

# def each_joint_weights():
#     weights = pd.read_csv('/home/yetty/Desktop/workspace/artemis_deepromp/data/weightsalljoints.csv', header=None)

#     all_weights = []
#     first_joint_weights = []
#     second_joint_weights = []
#     third_joint_weights = []
#     fourth_joint_weights = []
#     fifth_joint_weights = []
#     six_joint_weights = []
#     seventh_joint_weights = []

#     for index, row in weights.iterrows():
#         first_joint_weights.append(row.values[0:8])
#         second_joint_weights.append(row.values[8:16])
#         third_joint_weights.append(row.values[16:24])
#         fourth_joint_weights.append(row.values[24:32])
#         fifth_joint_weights.append(row.values[32:40])
#         six_joint_weights.append(row.values[40:48])
#         seventh_joint_weights.append(row.values[48:56])

#     return np.vstack((first_joint_weights)),np.vstack((second_joint_weights)),np.vstack((third_joint_weights)),np.vstack((fourth_joint_weights)),np.vstack((fifth_joint_weights)),np.vstack((six_joint_weights)),np.vstack((seventh_joint_weights))


class ProMP:
    """
    ProMP class
        N = number of basis functions
        h = bandwidth of basis functions
        dt = time step
        covY = variance of original p(Y)
        Wm = mean of weights [Nx1]
        covW = variance of weights [NxN]
    internal:
        Phi = basis functions evaluated for every step [TxN]
    methods:
        printMP (plots an MP showing a standar deviation above and below)
            name = title of the plot
    """
    def __init__(self, n=8, h=0.07, f=1, dt=1/(150-1)):
        self.n = n
        self.h = h
        self.dt = dt
        self.f = f

    def basis_func_gauss_glb(self): #[Txn] this is the basis function used globally for loss function
        """
        Evaluates Gaussian basis functions in [0,1]
        n = number of basis functions
        h = bandwidth
        dt = time step
        f = modulation factor
        """
        # phi = ProMP().basis_func_gauss_glb()  Matrix of basis functions
        tf_ = 1/self.f;
        T = int(round(tf_/self.dt+1))
        F = np.zeros((T,self.n))
        for z in range(0,T):
            t = z*self.dt
            q = np.zeros((1, self.n))
            for k in range(1,self.n+1):
                c = (k-1)/(self.n-1)
                q[0,k-1] = np.exp(-(self.f*t - c)*(self.f*t - c)/(2*self.h))
            F[z,:self.n] = q[0, :self.n]
        F = tf.cast(F, tf.float32)

        # Normalize basis functions
        F = F/np.transpose(mat.repmat(np.sum(F,axis=1),self.n,1)); #[TxN]
        return F #[Txn]

    # def basis_func_gauss_local(self,T): #[Txn] this is the basis function used only for each trajectory
    #     """
    #     Evaluates Gaussian basis functions in [0,1]
    #     n = number of basis functions
    #     h = bandwidth
    #     dt = time step
    #     f = modulation factor
    #     """
    #     # phi = ProMP().basis_func_gauss_local()  Matrix of basis functions
    #     dt = 1/(T-1)
    #     F = np.zeros((T,self.n))
    #     for z in range(0,T):
    #         t = z*dt
    #         q = np.zeros((1, self.n))
    #         for k in range(1,self.n+1):
    #             c = (k-1)/(self.n-1)
    #             q[0,k-1] = np.exp(-(self.f*t - c)*(self.f*t - c)/(2*self.h))
    #         F[z,:self.n] = q[0, :self.n]
    #     F = tf.cast(F, tf.float32)

    #     # Normalize basis functions
    #     F = F/np.transpose(mat.repmat(np.sum(F,axis=1),self.n,1)); #[TxN]
    #     return F #[Txn]

    # def weights_from_traj_demonstrations(self):
    #      json_files = [pos_json for pos_json in os.listdir(PATH) if pos_json.endswith('.json')]
    #      json_files = natsorted(json_files)

    #      j1_weights = []
    #      j2_weights = []
    #      j3_weights = []
    #      j4_weights = []
    #      j5_weights = []
    #      j6_weights = []
    #      j7_weights = []
    #      for index, js in enumerate(json_files):
    #          with open(os.path.join(PATH, js)) as json_file:
    #              json_text = json.load(json_file)
    #              joint_pos = json_text['joint_pos']
    #              joint_matrix = np.vstack((joint_pos)) # 140 by 7
    #              num_samples = joint_matrix.shape[0]
    #              phi = ProMP().basis_func_gauss_local(num_samples) # 140 by 8
    #              weights = np.transpose(tf.matmul(np.linalg.pinv(phi),tf.cast(joint_matrix, tf.float32))) # weights.shape (7, 8)

    #          j1_weights.append(weights[0])
    #          j2_weights.append(weights[1])
    #          j3_weights.append(weights[2])
    #          j4_weights.append(weights[3])
    #          j5_weights.append(weights[4]) # 200 by 8
    #          j6_weights.append(weights[5])
    #          j7_weights.append(weights[6])
    #      return np.vstack((j1_weights)), np.vstack((j2_weights)), np.vstack((j3_weights)), np.vstack((j4_weights)), np.vstack((j5_weights)), np.vstack((j6_weights)), np.vstack((j7_weights))

    # def trajectories_from_weights(self,weights): # weights = ProMP().weights_from_traj_demonstrations()
    #     """
    #     E.g code;
    #     ProMP().trajectories_from_weights(ProMP().weights_from_traj_demonstrations())
    #     """
    #     phi = ProMP().basis_func_gauss_glb() # 150 by 8, weights is 200 by 8
    #     if weights[0].shape[0] >= 200:
    #         q_traj_1 = tf.matmul(weights[0], np.transpose(phi))
    #         q_traj_2 = tf.matmul(weights[1], np.transpose(phi))
    #         q_traj_3 = tf.matmul(weights[2], np.transpose(phi))
    #         q_traj_4 = tf.matmul(weights[3], np.transpose(phi))
    #         q_traj_5 = tf.matmul(weights[4], np.transpose(phi))
    #         q_traj_6 = tf.matmul(weights[5], np.transpose(phi))
    #         q_traj_7 = tf.matmul(weights[6], np.transpose(phi))
    #         return q_traj_1, q_traj_2, q_traj_3, q_traj_4, q_traj_5, q_traj_6, q_traj_7 # 200 BY 150
    #     else:
    #         q_traj = tf.matmul(tf.cast(weights, tf.float32), K.transpose(phi))
    #         return q_traj

    # def all_weights_from_traj_demo(self, traj_dir):
    #     json_files = [pos_json for pos_json in os.listdir(traj_dir) if pos_json.endswith('.json')]
    #     json_files = natsorted(json_files)

    #     all_weights = []
    #     for index, js in enumerate(json_files):
    #         with open(os.path.join(traj_dir, js)) as json_file:
    #             json_text = json.load(json_file)
    #             if 'joint_position' in json_text.keys():
    #                 joint_pos_key = 'joint_position'
    #             elif 'joint_pos' in json_text.keys():
    #                 joint_pos_key = 'joint_pos'
    #             else:
    #                 raise KeyError(f"Joint position not found in the trajectory file {js}")
    #             joint_pos = json_text[joint_pos_key]
    #             joint_matrix = np.vstack(np.array(joint_pos)[:, :7]) # 140 by 7
    #             num_samples = joint_matrix.shape[0]
    #             phi = ProMP().basis_func_gauss_local(num_samples) # 140 by 8
    #             weights = np.transpose(tf.matmul(np.linalg.pinv(phi),tf.cast(joint_matrix, tf.float32)))

    #         all_weights.append(np.hstack((weights)))
    #     return np.vstack((all_weights))

    # def weights_from_trajectories(self,trajectories):
    #     """
    #     E.g code;
    #     np.transpose(tf.matmul(np.linalg.pinv(phi),joint_matrix)) # weights.shape (7, 8)
    #     """
    #     phi = ProMP().basis_func_gauss_glb() # 150 by 8, trajectories is 200 by 150
    #     if len(trajectories) > 1:
    #         weights_1 = np.transpose(tf.matmul(np.linalg.pinv(phi),np.transpose(trajectories[0])))
    #         weights_2 = np.transpose(tf.matmul(np.linalg.pinv(phi),np.transpose(trajectories[1])))
    #         weights_3 = np.transpose(tf.matmul(np.linalg.pinv(phi),np.transpose(trajectories[2])))
    #         weights_4 = np.transpose(tf.matmul(np.linalg.pinv(phi),np.transpose(trajectories[3])))
    #         weights_5 = np.transpose(tf.matmul(np.linalg.pinv(phi),np.transpose(trajectories[4])))
    #         weights_6 = np.transpose(tf.matmul(np.linalg.pinv(phi),np.transpose(trajectories[5])))
    #         weights_7 = np.transpose(tf.matmul(np.linalg.pinv(phi),np.transpose(trajectories[6])))
    #         return weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7
    #     else:
    #         weights = np.transpose(tf.matmul(np.linalg.pinv(phi),np.transpose(trajectories)))
    #         return weights

    # def mean_weights(self):
    #     """
    #     E.g for joint 1 weights mean
    #     w = mean_weights(each_joint_weights()[0])
    #     """
    #     j1_weights, j2_weights, j3_weights, j4_weights, j5_weights, j6_weights, j7_weights = ProMP().weights_from_traj_demonstrations()
    #     mean_w1 = [np.mean(item) for item in j1_weights]
    #     mean_w2 = [np.mean(item) for item in j2_weights]
    #     mean_w3 = [np.mean(item) for item in j3_weights]
    #     mean_w4 = [np.mean(item) for item in j4_weights]
    #     mean_w5 = [np.mean(item) for item in j5_weights]
    #     mean_w6 = [np.mean(item) for item in j6_weights]
    #     mean_w7 = [np.mean(item) for item in j7_weights]
    #     return mean_w1, mean_w2, mean_w3, mean_w4, mean_w5, mean_w6, mean_w7

    # def residual_from_weights(self,weights=None, mean=None):
    #     if weights is None:
    #         weights = ProMP().weights_from_traj_demonstrations()
    #         mean_weights = ProMP().mean_weights()
    #         residual_1 = [x - y for (x,y) in zip(weights[0],mean_weights[0])]
    #         residual_2 = [x - y for (x,y) in zip(weights[1],mean_weights[1])]
    #         residual_3 = [x - y for (x,y) in zip(weights[2],mean_weights[2])]
    #         residual_4 = [x - y for (x,y) in zip(weights[3],mean_weights[3])]
    #         residual_5 = [x - y for (x,y) in zip(weights[4],mean_weights[4])]
    #         residual_6 = [x - y for (x,y) in zip(weights[5],mean_weights[5])]
    #         residual_7 = [x - y for (x,y) in zip(weights[6],mean_weights[6])]
    #         return np.vstack((residual_1)), np.vstack((residual_2)), np.vstack((residual_3)), np.vstack((residual_4)), np.vstack((residual_5)), np.vstack((residual_6)), np.vstack((residual_7))
    #     else:
    #         residual = [x - y for (x,y) in zip(weights,mean)]
    #         return residual

    # def weights_from_residual(self,residual=None, mean=None):
    #     if residual is None:
    #         residual = ProMP().residual_from_weights()
    #         mean_weights = ProMP().mean_weights()
    #         weights_1 = [x + y for (x,y) in zip(residual[0],mean_weights[0])]
    #         weights_2 = [x + y for (x,y) in zip(residual[1],mean_weights[1])]
    #         weights_3 = [x + y for (x,y) in zip(residual[2],mean_weights[2])]
    #         weights_4 = [x + y for (x,y) in zip(residual[3],mean_weights[3])]
    #         weights_5 = [x + y for (x,y) in zip(residual[4],mean_weights[4])]
    #         weights_6 = [x + y for (x,y) in zip(residual[5],mean_weights[5])]
    #         weights_7 = [x + y for (x,y) in zip(residual[6],mean_weights[6])]
    #         return np.vstack((weights_1)), np.vstack((weights_2)), np.vstack((weights_3)), np.vstack((weights_4)), np.vstack((weights_5)), np.vstack((weights_6)), np.vstack((weights_7))
    #     else:
    #         weights = [x + y for (x,y) in zip(residual,mean)]
    #         return np.vstack((weights))

    # def mean_joints(self):
    #     """
    #     E.g for joint 1 weights mean
    #     w = mean_weights(each_joint_weights()[0])
    #     """
    #     weights = ProMP().weights_from_traj_demonstrations()
    #     q_traj_1, q_traj_2, q_traj_3, q_traj_4, q_traj_5, q_traj_6, q_traj_7 = ProMP().trajectories_from_weights(weights)
    #     mean_j1 = [np.mean(item) for item in q_traj_1]
    #     mean_j2 = [np.mean(item) for item in q_traj_2]
    #     mean_j3 = [np.mean(item) for item in q_traj_3]
    #     mean_j4 = [np.mean(item) for item in q_traj_4]
    #     mean_j5 = [np.mean(item) for item in q_traj_5]
    #     mean_j6 = [np.mean(item) for item in q_traj_6]
    #     mean_j7 = [np.mean(item) for item in q_traj_7]
    #     return mean_j1, mean_j2, mean_j3, mean_j4, mean_j5, mean_j6, mean_j7

    # def residual_from_joints(self,trajectories=None, mean=None):
    #     if trajectories is None:
    #         weights = ProMP().weights_from_traj_demonstrations()
    #         trajectories = ProMP().trajectories_from_weights(weights)
    #         mean_joints = ProMP().mean_joints()
    #         residual_1 = [x - y for (x,y) in zip(trajectories[0],mean_joints[0])]
    #         residual_2 = [x - y for (x,y) in zip(trajectories[1],mean_joints[1])]
    #         residual_3 = [x - y for (x,y) in zip(trajectories[2],mean_joints[2])]
    #         residual_4 = [x - y for (x,y) in zip(trajectories[3],mean_joints[3])]
    #         residual_5 = [x - y for (x,y) in zip(trajectories[4],mean_joints[4])]
    #         residual_6 = [x - y for (x,y) in zip(trajectories[5],mean_joints[5])]
    #         residual_7 = [x - y for (x,y) in zip(trajectories[6],mean_joints[6])]
    #         return np.vstack((residual_1)), np.vstack((residual_2)), np.vstack((residual_3)), np.vstack((residual_4)), np.vstack((residual_5)), np.vstack((residual_6)), np.vstack((residual_7))
    #     else:
    #         residual = np.transpose([x - y for (x,y) in zip(trajectories,mean)])
    #         return p.vstack((residual))

    # def joints_from_residual(self,residual=None, mean=None):
    #     if residual is None:
    #         weights = ProMP().weights_from_traj_demonstrations()
    #         trajectories = ProMP().trajectories_from_weights(weights)
    #         residual = ProMP().residual_from_joints()
    #         mean_joints = ProMP().mean_joints()

    #         joints_1 = [x + y for (x,y) in zip(residual[0],mean_joints[0])]
    #         joints_2 = [x + y for (x,y) in zip(residual[1],mean_joints[1])]
    #         joints_3 = [x + y for (x,y) in zip(residual[2],mean_joints[2])]
    #         joints_4 = [x + y for (x,y) in zip(residual[3],mean_joints[3])]
    #         joints_5 = [x + y for (x,y) in zip(residual[4],mean_joints[4])]
    #         joints_6 = [x + y for (x,y) in zip(residual[5],mean_joints[5])]
    #         joints_7 = [x + y for (x,y) in zip(residual[6],mean_joints[6])]
    #         return np.vstack((joints_1)), np.vstack((joints_2)), np.vstack((joints_3)), np.vstack((joints_4)), np.vstack((joints_5)), np.vstack((joints_6)), np.vstack((joints_7))
    #     else:
    #         joints_traj = [x + y for (x,y) in zip(residual,mean)]
    #         return np.vstack((joints_traj))

    # def read_traj_into_csv(self):
    #     ww = ProMP().weights_from_traj_demonstrations()
    #     for index, jw in enumerate(ww):
    #         with open(current_dir + "/data/weights/joints{}".format(index+1),"w+") as my_csv:
    #             csvWriter = csv.writer(my_csv,delimiter=',')
    #             csvWriter.writerows(jw)

    # def verify_weights():
    #     ww = ProMP().weights_from_traj_demonstrations()
    #     qq = ProMP().trajectories_from_weights(weights=ww)
    #     aww = ProMP().weights_from_trajectories(qq)
    #     aqq = ProMP().trajectories_from_weights(weights=aww)
    #     # check ww[index] with aww[index] for weights_
    #     # check qq[index] with aqq[index] for weights_

    #     jr = ProMP().joints_from_residual()
    #     # check qq[index] with jr[index] for joints
