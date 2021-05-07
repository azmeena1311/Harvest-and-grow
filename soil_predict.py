
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from flask import Markup
# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from werkzeug.utils import secure_filename
from recommend import crop_recommend

def model_predict(img_path, model):

    test_image = image.load_img(img_path, target_size = (64,64))
    test_image = image.img_to_array(test_image)
    test_image=test_image/255
    test_image = np.expand_dims(test_image, axis = 0)
    result = np.argmax(model.predict(test_image),axis=1)

    if result[0] == 0:
        result =  Markup(str(crop_recommend['Alluvial Soil']))
    elif result[0] == 1:
        result = Markup(str(crop_recommend["Black Soil"]))
    elif result[0] == 2:
        result =  Markup(str(crop_recommend["Clay Soil"]))
    elif result[0] == 3:
        result =  Markup(str(crop_recommend["Red Soil"]))
    


    return result