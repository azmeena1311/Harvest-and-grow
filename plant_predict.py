
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from werkzeug.utils import secure_filename
from recommend import crop_recommend

def disease_predict(img_path, model):
    test_image = image.load_img(img_path, target_size = (64,64))
    test_image = image.img_to_array(test_image)
    test_image=test_image/255
    test_image = np.expand_dims(test_image, axis = 0)
    result = np.argmax(model.predict(test_image),axis=1)

    if result[0] == 0:
        result = 'Apple Black rot'
    elif result[0] == 1:
        result = "Apple Healthy"
    elif result[0] == 2:
        result = "Cherry Powdery mildew"
    elif result[0] == 3:
        result = "Cherry healthy"
    elif result[0] == 4:
        result = "Corn(maize) leaf_spot Gray_leaf_spot"
    elif result[0] == 5:
        result = "Corn_(maize) healthy"
    elif result[0] == 6:
        result = "Grape Black rot"
    elif result[0] == 7:
        result = "Grape healthy"
    elif result[0] == 8:
        result = "Tomato Target_Spot"
    elif result[0] == 9:
        result = "Tomato healthy"


    return result