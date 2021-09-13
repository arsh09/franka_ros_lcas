import numpy as np
import os, sys 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.models import Model
import tensorflow as tf
from PIL import Image
from utils_rtp import ProMP



class Predictor:

    def __init__(self, encoder_model_path, predictor_model_path):
        self.all_phi = self.promp_train()

        encoder_model = tf.keras.models.load_model(encoder_model_path)
        self.encoder = Model(encoder_model.input, encoder_model.get_layer("bottleneck").output)
        self.exp_model = tf.keras.models.load_model(predictor_model_path, compile=False)

    def promp_train(self):
        phi = ProMP().basis_func_gauss_glb()

        zeros = np.zeros([phi.shape[0], 8])
        h1 = np.hstack((phi, zeros, zeros, zeros, zeros, zeros, zeros))
        h2 = np.hstack((zeros, phi, zeros, zeros, zeros, zeros, zeros))
        h3 = np.hstack((zeros, zeros, phi, zeros, zeros, zeros, zeros))
        h4 = np.hstack((zeros, zeros, zeros, phi, zeros, zeros, zeros))
        h5 = np.hstack((zeros, zeros, zeros, zeros, phi, zeros, zeros))
        h6 = np.hstack((zeros, zeros, zeros, zeros, zeros, phi, zeros))
        h7 = np.hstack((zeros, zeros, zeros, zeros, zeros, zeros, phi))

        vstack = np.vstack((h1, h2, h3, h4, h5, h6, h7))
        vstack = tf.cast(vstack, tf.float32)
        return vstack

    def preprocess_image(self, image):
        return np.asarray(image.resize((256, 256)))

    def predict(self, image_numpy):
        # image_numpy = np.expand_dims(image_numpy, axis=0)
        latent_img = self.encoder.predict(image_numpy/255)
        q_val_pred = self.exp_model.predict(latent_img)

        traj_pred = np.matmul(self.all_phi, np.transpose(q_val_pred)).squeeze()

        return traj_pred #np.reshape(traj_pred, (-1, 150))


if __name__ == "__main__":
    ENCODED_MODEL_PATH = "/home/arshad/Documents/reach_to_palpate_validation_models/encoded_model_regions"
    PREDICTOR_MODEL = "/home/arshad/Documents/reach_to_palpate_validation_models/model_cnn_rgb_1"
    image = np.load( "/home/arshad/catkin_ws/image_xy_rtp.npy" )

    predictor = Predictor(ENCODED_MODEL_PATH, PREDICTOR_MODEL)
    traj = predictor.predict(image)

    np.save("/home/arshad/catkin_ws/predicted_joints_values_rtp.npy", traj)
    print ("\n  Predicted ProMPs weights for RTP task. Joint trajectory is saved in the file. \n  Press 'p' to display the trajectory...")
