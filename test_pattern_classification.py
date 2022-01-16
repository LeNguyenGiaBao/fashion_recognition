import tensorflow as tf
import numpy as np 
import cv2 


# model = tf.keras.models.load_model('./models/fashion_pattern_5_01_1.79.h5')
model = tf.keras.models.load_model('./models/fashion_pattern_41_0.88.h5')

classes = ['floral', 'plain', 'polka dot', 'squares', 'stripes']
# classes = ['animal', 'cartoon', 'floral', 'geometry', 'ikat', 'plain', 'polka dot', 'squares', 'stripes', 'tribal']
# print(model.summary())
img = cv2.imread('./image/caro.png')
print(img.shape)
img = cv2.resize(img, (224,224))
input_image = np.expand_dims(img, axis=0)
output = model.predict(input_image)
print(output)
print(np.argmax(output[0]))
max_index = np.argmax(output[0])
print(classes[max_index])