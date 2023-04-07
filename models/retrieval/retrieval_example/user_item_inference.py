import tensorflow as tf
import os

model_save_path = 'model'
model_name = 'bruteForce'

# Load the model
path = os.path.join(model_save_path, model_name)
model = tf.saved_model.load(path)