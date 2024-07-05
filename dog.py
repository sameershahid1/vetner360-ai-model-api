from tensorflow.keras.preprocessing import image as tkimg
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('./model.h5', compile=False)
class_names = ['husky', 'beagle', 'rottweiler', 'german-shepherd', 'dalmatian', 'poodle', 'bulldog', 'labrador-retriever']

def preprocess_image(image_path, target_size=(224, 224)):
    img = tkimg.load_img(image_path, target_size=target_size)
    img_array = tkimg.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

image_path = './check/img1.jpeg'
img_array = preprocess_image(image_path)
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)
print(f'Predicted class index: {predicted_class[0]}')
print(f'Predicted breed: {class_names[predicted_class[0]]}')
