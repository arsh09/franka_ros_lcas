import numpy as np
# import keras
import os 
from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow as tf
# import ipdb



ENCODER_MODEL_PATH = '/home/arshad/Documents/wedge_to_palpate_validation_models/encoded_model'

print ( "ENCODER_MODEL_PATH exists?", os.path.isdir( ENCODER_MODEL_PATH ))

encoder_model = tf.keras.models.load_model(ENCODER_MODEL_PATH)
encoder = Model(encoder_model.input, encoder_model.get_layer('bottleneck').output )

EXPERIMENT_PATH = '/home/arshad/Documents/wedge_to_palpate_validation_models/spatial_model_experiment10_relu'
print ( "EXPERIMENT_PATH exists?", os.path.isdir( EXPERIMENT_PATH ))

exp_model = tf.keras.models.load_model(EXPERIMENT_PATH, compile=False)

if __name__ == "__main__":

    # img_in_arry = image in array, it should be something like (1,256,256,3)
    
    img_in_arry = np.load('/home/arshad/catkin_ws/image_xy.npy')
    target_point = np.load('/home/arshad/catkin_ws/image_target_xy.npy')
    target_point = np.reshape( target_point, (1,2) )
        
    print ( img_in_arry.shape, target_point.shape ) 
    
    # target_point = [217,2]
    latent_dim = encoder.predict(img_in_arry)
    q_val_pred = exp_model.predict([latent_dim, target_point])
    # ipdb.set_trace()
