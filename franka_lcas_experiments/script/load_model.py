import numpy as np
import os, sys 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow as tf
import ipdb

current_dir = os.getcwd()
sys.path.insert(1, current_dir)
from utils import ProMP


ENCODER_MODEL_PATH = '/home/arshad/Documents/wedge_to_palpate_validation_models/encoded_model'

# print ( "ENCODER_MODEL_PATH exists?", os.path.isdir( ENCODER_MODEL_PATH ))
 
encoder_model = tf.keras.models.load_model(ENCODER_MODEL_PATH)
encoder = Model(encoder_model.input, encoder_model.get_layer('bottleneck').output )
 
EXPERIMENT_PATH = '/home/arshad/Documents/wedge_to_palpate_validation_models/spatial_model_experiment10_relu'
# print ( "EXPERIMENT_PATH exists?", os.path.isdir( EXPERIMENT_PATH ))

exp_model = tf.keras.models.load_model(EXPERIMENT_PATH, compile=False)

def promp_train():
    phi = ProMP().basis_func_gauss_glb()

    zeros =  np.zeros([phi.shape[0], 10])
    h1 = np.hstack((phi, zeros, zeros, zeros, zeros, zeros, zeros))
    h2 = np.hstack((zeros, phi, zeros, zeros, zeros, zeros, zeros))
    h3 = np.hstack((zeros, zeros, phi, zeros, zeros, zeros, zeros))
    h4 = np.hstack((zeros, zeros, zeros, phi, zeros, zeros, zeros))
    h5 = np.hstack((zeros, zeros, zeros, zeros, phi, zeros, zeros))
    h6 = np.hstack((zeros, zeros, zeros, zeros, zeros, phi, zeros))
    h7 = np.hstack((zeros, zeros, zeros, zeros, zeros, zeros, phi))

    vstack = np.vstack((h1,h2,h3,h4,h5,h6,h7))
    vstack = tf.cast(vstack, tf.float32)
    return vstack
 
if __name__ == "__main__":

    img_in_arry = np.load('/home/arshad/catkin_ws/image_xy.npy')
    img_in_arry = np.reshape(img_in_arry, (1,256,256,3))
    target_point = np.load('/home/arshad/catkin_ws/image_target_xy.npy')

    target_point = np.reshape( target_point, (1,2) )

    # print ("\n\nReceived saved target image and target points. Predicting weights for DMP...")
    # print ( img_in_arry.shape, target_point ) 

    latent_dim = encoder.predict(img_in_arry)
    q_val_pred = exp_model.predict([latent_dim, target_point])
    
    all_phi = promp_train()
    traj_true = np.matmul(all_phi,np.transpose(q_val_pred[0]))
    # print (traj_true[:30])
    q1pred,q2pred,q3pred,q4pred,q5pred,q6pred,q7pred = traj_true[0:150], traj_true[150:300], traj_true[300:450], traj_true[450:600], traj_true[600:750], traj_true[750:900], traj_true[900:1050]

    print ("\n  Predicted ProMPs weights for WPP. Joint trajectory is saved in the file. \n  Press 'p' to display the trajectory...")

    np.save('/home/arshad/catkin_ws/predicted_joints_values.npy', traj_true)
