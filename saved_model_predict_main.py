import tensorflow.compat.v2 as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def normalize_img(image):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255.

img_array=np.array(Image.open('greyscale.png'))
img_to_be_predicted=normalize_img(img_array)
img_to_be_predicted=tf.reshape(img_to_be_predicted,[1,28,28])

saved_model=tf.keras.models.load_model('digitnn_model')
predictions=saved_model.predict(img_to_be_predicted,batch_size=1) #print probabilities of classes

classes = np.argmax(predictions, axis = 1)
print("classes:",classes,"\n")

