import math 
from tensorflow import keras
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
import numpy.matlib as mat
from natsort import natsorted
# from skimage.io import imread
import matplotlib.pyplot as plt
import os, json,  ipdb, csv, sys 
# from skimage.transform import resize
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Reshape, MaxPooling2D

current_dir = os.getcwd()
sys.path.insert(1, current_dir)

PATH = current_dir + "/data/config_{}/train/"

# def return_target(path, target):
#     all_target = {'train' :[], 'test': []}
#     # experiment 1 
#     # included = [0,1,2,3,6]
#     # excluded = [4,5]

#     # experiment 9 & 10
#     included = [0,1,2,3,4,5,6]
#     for i in range(1, 218):
#         with open(os.path.join(path, 'experiment_{}.json'.format(i))) as json_file:

#             json_text = json.load(json_file)
#             target_value = json_text[target]

#             if target == 'image_2d_point_resized_end':
#                 target_value = target_value
#             else:
#                 target_value = target_value[-1][0:7]

#             all_target['train'].append(target_value) if i % 7 in included else all_target['test'].append(target_value)
    
#     if all_target['test'] == []: # when included has all palpation patterns [0,1,2,3,4,5,6]
#         return np.vstack((all_target['train'])), []
#     else:
#         return np.vstack((all_target['train'])), np.vstack((all_target['test']))

# def load_images(data_dir):
#     images = glob(data_dir+ '*.*')

#     all_images = {'train' :[], 'test': []}
    
#     # experiment 1 
#     # included = [0,1,2,3,6]
#     # excluded = [4,5]

#     # experiment 9 & 10
#     included = [0,1,2,3,4,5,6]
#     for i in range(1,218):
#         filename = data_dir + 'experiment_{}.png'.format(i)
#         # os.rename(filename, filename.split('experiment_')[0] + '{}B'.format(index+1) + '.jpg')
#         img = imread(filename, pilmode='RGB')

#         all_images['train'].append(img) if i % 7 in included else all_images['test'].append(img)

#     if all_images['test'] == []: # when included has all palpation patterns [0,1,2,3,4,5,6]
#         return np.array(all_images['train']), []
#     else:
#         return np.array(all_images['train']), np.array(all_images['test'])

# def load_image(img_filename):
#     img = imread(img_filename, pilmode='RGB')
#     return img 

# def load_image_resize(img_filename):
#     img = imread(img_filename, pilmode='RGB')
#     # resizing image
#     img = resize(img, (256, 256, 3))
#     return img 

def euler2quaternion(r, p, y):
    ci = math.cos(r/2.0)
    si = math.sin(r/2.0)
    cj = math.cos(p/2.0)
    sj = math.sin(p/2.0)
    ck = math.cos(y/2.0)
    sk = math.sin(y/2.0)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk
    return [cj*sc-sj*cs, cj*ss + sj*cc, cj*cs - sj*sc, cj*cc + sj*ss]

def Trpyxyz(r, p, y, X, Y, Z):
	q = euler2quaternion(r, p, y)
	qi = q[0]
	qj = q[1]
	qk = q[2]
	qr = q[3]
	qi = qi/math.sqrt(qi*qi + qj*qj + qk*qk + qr*qr)
	qj = qj/math.sqrt(qi*qi + qj*qj + qk*qk + qr*qr)
	qk = qk/math.sqrt(qi*qi + qj*qj + qk*qk + qr*qr)
	qr = qr/math.sqrt(qi*qi + qj*qj + qk*qk + qr*qr)
	T = np.array([[1-2*(qj*qj+qk*qk), 2*(qi*qj-qk*qr), 2*(qi*qk+qj*qr),X],
	      [2*(qi*qj+qk*qr), 1-2*(qi*qi+qk*qk), 2*(qj*qk-qi*qr),Y],
	      [2*(qi*qk-qj*qr), 2*(qj*qk+qi*qr), 1-2*(qi*qi+qj*qj),Z],
	      [0, 0, 0, 1]])
	return T

def Trz(theta):
	T = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
		[np.sin(theta), np.cos(theta), 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1]])
	return T

def FK(q1,q2,q3,q4,q5,q6,q7):
    # Computes the transformation matrix of the end-effector for a given configuration (MoveIt frame)
	# values from URDF file https://github.com/StanfordASL/PandaRobot.jl/blob/master/deps/Panda/panda.urdf
	T01 = np.matmul(Trpyxyz(0, 0, 0, 0, 0, 0.333),Trz(q1))
	T12 = np.matmul(Trpyxyz(-1.57079632679, 0, 0, 0, 0, 0),Trz(q2))
	T23 = np.matmul(Trpyxyz(1.57079632679, 0, 0, 0, -0.316, 0),Trz(q3))
	T34 = np.matmul(Trpyxyz(1.57079632679, 0, 0, 0.0825, 0, 0),Trz(q4))
	T45 = np.matmul(Trpyxyz(-1.57079632679, 0, 0, -0.0825, 0.384, 0),Trz(q5))
	T56 = np.matmul(Trpyxyz(1.57079632679, 0, 0, 0, 0, 0),Trz(q6))
	T67 = np.matmul(Trpyxyz(1.57079632679, 0, 0, 0.088, 0, 0),Trz(q7))
	T78 = Trpyxyz(0, 0, 0, 0, 0, 0.107)
	T08 = np.matmul(T01, np.matmul(T12 ,np.matmul(T23, np.matmul(T34, np.matmul(T45, np.matmul(T56, np.matmul(T67, T78)))))))
	return T08

# for i in range(0,len(y_values)): 
#     implot = plt.imshow(load_image_resize(current_dir +'/data/config_{}/conifg{}_image.png'.format('A', 'A')))
#     # ipdb.set_trace()
#     plt.scatter(y_values[i][0], y_values[i][1], marker='+')
#     plt.savefig('/home/yetty/workspace/github/artemis_aaai_palp/plots/xy_A/{}.png'.format(i+1))
#     plt.clf()

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

    def __init__(self, n=10, h=0.07, f=1, dt=1/(150-1)):
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

    def basis_func_gauss_local(self,T): #[Txn] this is the basis function used only for each trajectory
        """
        Evaluates Gaussian basis functions in [0,1]
        n = number of basis functions
        h = bandwidth
        dt = time step
        f = modulation factor
        """
        # phi = ProMP().basis_func_gauss_local()  Matrix of basis functions
        dt = 1/(T-1)
        F = np.zeros((T,self.n))
        for z in range(0,T):
            t = z*dt
            q = np.zeros((1, self.n))
            for k in range(1,self.n+1):
                c = (k-1)/(self.n-1)
                q[0,k-1] = np.exp(-(self.f*t - c)*(self.f*t - c)/(2*self.h))
            F[z,:self.n] = q[0, :self.n]
        F = tf.cast(F, tf.float32)

        # Normalize basis functions
        F = F/np.transpose(mat.repmat(np.sum(F,axis=1),self.n,1)); #[TxN]
        return F #[Txn]

    def weights_from_traj_demonstrations(self):
         json_files = [pos_json for pos_json in os.listdir(PATH) if pos_json.endswith('.json')]
         json_files = natsorted(json_files)

         j1_weights = []
         j2_weights = []
         j3_weights = []
         j4_weights = []
         j5_weights = []
         j6_weights = []
         j7_weights = []
         for index, js in enumerate(json_files):
             with open(os.path.join(PATH, json_files[21])) as json_file:
                 json_text = json.load(json_file)
                 joint_pos = json_text['joint_pos']
                 joint_matrix = np.vstack((joint_pos)) # lenght_of_traj by 7
                 num_samples = joint_matrix.shape[0]
                 phi = ProMP().basis_func_gauss_local(num_samples) # 140 by 8

                 # 8 by lenght_of_traj(123) * length_by_traj by 7 =  weights.shape (7, 8)
                 weights = np.transpose(tf.matmul(np.linalg.pinv(phi),tf.cast(joint_matrix, tf.float32))) 

             j1_weights.append(weights[0])
             j2_weights.append(weights[1])
             j3_weights.append(weights[2])
             j4_weights.append(weights[3])
             j5_weights.append(weights[4]) # 200 by 8
             j6_weights.append(weights[5])
             j7_weights.append(weights[6])
         return np.vstack((j1_weights)), np.vstack((j2_weights)), np.vstack((j3_weights)), np.vstack((j4_weights)), np.vstack((j5_weights)), np.vstack((j6_weights)), np.vstack((j7_weights))

    def trajectories_from_weights(self,weights): # weights = ProMP().weights_from_traj_demonstrations()
        """
        E.g code;
        ProMP().trajectories_from_weights(ProMP().weights_from_traj_demonstrations())
        """
        phi = ProMP().basis_func_gauss_glb() # 150 by 8, weights is 200 by 8
        if weights[0].shape[0] >= 200:
            q_traj_1 = tf.matmul(weights[0], np.transpose(phi))
            q_traj_2 = tf.matmul(weights[1], np.transpose(phi))
            q_traj_3 = tf.matmul(weights[2], np.transpose(phi))
            q_traj_4 = tf.matmul(weights[3], np.transpose(phi))
            q_traj_5 = tf.matmul(weights[4], np.transpose(phi))
            q_traj_6 = tf.matmul(weights[5], np.transpose(phi))
            q_traj_7 = tf.matmul(weights[6], np.transpose(phi))
            return q_traj_1, q_traj_2, q_traj_3, q_traj_4, q_traj_5, q_traj_6, q_traj_7 # 200 BY 150
        else:
            q_traj = tf.matmul(tf.cast(weights, tf.float32), K.transpose(phi))
            return q_traj

    def all_weights_from_traj_demo(self, PATH):
        all_weights = {'train' :[], 'test': []}
        
        # experiment 1  
        # included = [0,1,2,3,6]
        # excluded = [4,5]

        # experiment 5
        included = [0,1,2,3,4,5,6]
        for i in range(1, 218):
            with open(os.path.join(PATH, 'experiment_{}.json'.format(i))) as json_file:
                json_text = json.load(json_file)
                joint_position = json_text['joint_position']
                joint_pos = [jj[0:7] for jj in joint_position]
                joint_matrix = np.vstack((joint_pos)) # 155 by 7
                num_samples = joint_matrix.shape[0]
                phi = ProMP().basis_func_gauss_local(num_samples) # 155 by self.n
                weights = np.transpose(tf.matmul(np.linalg.pinv(phi),tf.cast(joint_matrix, tf.float32)))
            
            all_weights['train'].append(np.hstack((weights))) if i % 7 in included else all_weights['test'].append(np.hstack((weights)))

        if all_weights['test'] == []: # when included has all palpation patterns [0,1,2,3,4,5,6]
            return np.vstack((all_weights['train'])), []
        else:
            return np.vstack((all_weights['train'])), np.vstack((all_weights['test']))

    def weights_from_trajectories(self,trajectories):
        """
        E.g code;
        np.transpose(tf.matmul(np.linalg.pinv(phi),joint_matrix)) # weights.shape (7, 8)
        """
        phi = ProMP().basis_func_gauss_glb() # 150 by 8, trajectories is 200 by 150
        if len(trajectories) > 1:
            weights_1 = np.transpose(tf.matmul(np.linalg.pinv(phi),np.transpose(trajectories[0])))
            weights_2 = np.transpose(tf.matmul(np.linalg.pinv(phi),np.transpose(trajectories[1])))
            weights_3 = np.transpose(tf.matmul(np.linalg.pinv(phi),np.transpose(trajectories[2])))
            weights_4 = np.transpose(tf.matmul(np.linalg.pinv(phi),np.transpose(trajectories[3])))
            weights_5 = np.transpose(tf.matmul(np.linalg.pinv(phi),np.transpose(trajectories[4])))
            weights_6 = np.transpose(tf.matmul(np.linalg.pinv(phi),np.transpose(trajectories[5])))
            weights_7 = np.transpose(tf.matmul(np.linalg.pinv(phi),np.transpose(trajectories[6])))
            return weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7
        else:
            weights = np.transpose(tf.matmul(np.linalg.pinv(phi),np.transpose(trajectories)))
            return weights

    def mean_weights(self):
        """
        E.g for joint 1 weights mean
        w = mean_weights(each_joint_weights()[0])
        """
        j1_weights, j2_weights, j3_weights, j4_weights, j5_weights, j6_weights, j7_weights = ProMP().weights_from_traj_demonstrations()
        mean_w1 = [np.mean(item) for item in j1_weights]
        mean_w2 = [np.mean(item) for item in j2_weights]
        mean_w3 = [np.mean(item) for item in j3_weights]
        mean_w4 = [np.mean(item) for item in j4_weights]
        mean_w5 = [np.mean(item) for item in j5_weights]
        mean_w6 = [np.mean(item) for item in j6_weights]
        mean_w7 = [np.mean(item) for item in j7_weights]
        return mean_w1, mean_w2, mean_w3, mean_w4, mean_w5, mean_w6, mean_w7

    def residual_from_weights(self,weights=None, mean=None):
        if weights is None:
            weights = ProMP().weights_from_traj_demonstrations()
            mean_weights = ProMP().mean_weights()
            residual_1 = [x - y for (x,y) in zip(weights[0],mean_weights[0])]
            residual_2 = [x - y for (x,y) in zip(weights[1],mean_weights[1])]
            residual_3 = [x - y for (x,y) in zip(weights[2],mean_weights[2])]
            residual_4 = [x - y for (x,y) in zip(weights[3],mean_weights[3])]
            residual_5 = [x - y for (x,y) in zip(weights[4],mean_weights[4])]
            residual_6 = [x - y for (x,y) in zip(weights[5],mean_weights[5])]
            residual_7 = [x - y for (x,y) in zip(weights[6],mean_weights[6])]
            return np.vstack((residual_1)), np.vstack((residual_2)), np.vstack((residual_3)), np.vstack((residual_4)), np.vstack((residual_5)), np.vstack((residual_6)), np.vstack((residual_7))
        else:
            residual = [x - y for (x,y) in zip(weights,mean)]
            return residual

    def weights_from_residual(self,residual=None, mean=None):
        if residual is None:
            residual = ProMP().residual_from_weights()
            mean_weights = ProMP().mean_weights()
            weights_1 = [x + y for (x,y) in zip(residual[0],mean_weights[0])]
            weights_2 = [x + y for (x,y) in zip(residual[1],mean_weights[1])]
            weights_3 = [x + y for (x,y) in zip(residual[2],mean_weights[2])]
            weights_4 = [x + y for (x,y) in zip(residual[3],mean_weights[3])]
            weights_5 = [x + y for (x,y) in zip(residual[4],mean_weights[4])]
            weights_6 = [x + y for (x,y) in zip(residual[5],mean_weights[5])]
            weights_7 = [x + y for (x,y) in zip(residual[6],mean_weights[6])]
            return np.vstack((weights_1)), np.vstack((weights_2)), np.vstack((weights_3)), np.vstack((weights_4)), np.vstack((weights_5)), np.vstack((weights_6)), np.vstack((weights_7))
        else:
            weights = [x + y for (x,y) in zip(residual,mean)]
            return np.vstack((weights))

    def mean_joints(self):
        """
        E.g for joint 1 weights mean
        w = mean_weights(each_joint_weights()[0])
        """
        weights = ProMP().weights_from_traj_demonstrations()
        q_traj_1, q_traj_2, q_traj_3, q_traj_4, q_traj_5, q_traj_6, q_traj_7 = ProMP().trajectories_from_weights(weights)
        mean_j1 = [np.mean(item) for item in q_traj_1]
        mean_j2 = [np.mean(item) for item in q_traj_2]
        mean_j3 = [np.mean(item) for item in q_traj_3]
        mean_j4 = [np.mean(item) for item in q_traj_4]
        mean_j5 = [np.mean(item) for item in q_traj_5]
        mean_j6 = [np.mean(item) for item in q_traj_6]
        mean_j7 = [np.mean(item) for item in q_traj_7]
        return mean_j1, mean_j2, mean_j3, mean_j4, mean_j5, mean_j6, mean_j7

    def residual_from_joints(self,trajectories=None, mean=None):
        if trajectories is None:
            weights = ProMP().weights_from_traj_demonstrations()
            trajectories = ProMP().trajectories_from_weights(weights)
            mean_joints = ProMP().mean_joints()
            residual_1 = [x - y for (x,y) in zip(trajectories[0],mean_joints[0])]
            residual_2 = [x - y for (x,y) in zip(trajectories[1],mean_joints[1])]
            residual_3 = [x - y for (x,y) in zip(trajectories[2],mean_joints[2])]
            residual_4 = [x - y for (x,y) in zip(trajectories[3],mean_joints[3])]
            residual_5 = [x - y for (x,y) in zip(trajectories[4],mean_joints[4])]
            residual_6 = [x - y for (x,y) in zip(trajectories[5],mean_joints[5])]
            residual_7 = [x - y for (x,y) in zip(trajectories[6],mean_joints[6])]
            return np.vstack((residual_1)), np.vstack((residual_2)), np.vstack((residual_3)), np.vstack((residual_4)), np.vstack((residual_5)), np.vstack((residual_6)), np.vstack((residual_7))
        else:
            residual = np.transpose([x - y for (x,y) in zip(trajectories,mean)])
            return p.vstack((residual))

    def joints_from_residual(self,residual=None, mean=None):
        if residual is None:
            weights = ProMP().weights_from_traj_demonstrations()
            trajectories = ProMP().trajectories_from_weights(weights)
            residual = ProMP().residual_from_joints()
            mean_joints = ProMP().mean_joints()

            joints_1 = [x + y for (x,y) in zip(residual[0],mean_joints[0])]
            joints_2 = [x + y for (x,y) in zip(residual[1],mean_joints[1])]
            joints_3 = [x + y for (x,y) in zip(residual[2],mean_joints[2])]
            joints_4 = [x + y for (x,y) in zip(residual[3],mean_joints[3])]
            joints_5 = [x + y for (x,y) in zip(residual[4],mean_joints[4])]
            joints_6 = [x + y for (x,y) in zip(residual[5],mean_joints[5])]
            joints_7 = [x + y for (x,y) in zip(residual[6],mean_joints[6])]
            return np.vstack((joints_1)), np.vstack((joints_2)), np.vstack((joints_3)), np.vstack((joints_4)), np.vstack((joints_5)), np.vstack((joints_6)), np.vstack((joints_7))
        else:
            joints_traj = [x + y for (x,y) in zip(residual,mean)]
            return np.vstack((joints_traj))

    def read_traj_into_csv(self):
        ww = ProMP().weights_from_traj_demonstrations()
        for index, jw in enumerate(ww):
            with open(current_dir + "/data/weights/joints{}".format(index+1),"w+") as my_csv:
                csvWriter = csv.writer(my_csv,delimiter=',')
                csvWriter.writerows(jw)

    def verify_weights():
        ww = ProMP().weights_from_traj_demonstrations()
        qq = ProMP().trajectories_from_weights(weights=ww)
        aww = ProMP().weights_from_trajectories(qq)
        aqq = ProMP().trajectories_from_weights(weights=aww)
        # check ww[index] with aww[index] for weights_
        # check qq[index] with aqq[index] for weights_

        jr = ProMP().joints_from_residual()
        # check qq[index] with jr[index] for joints

# ProMP().weights_from_traj_demonstrations()