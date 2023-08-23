import numpy as np
import tensorflow as tf
import os
import sys
import cv2 as cv

# Map of characters, in which key is the index and value is the ascii code of the character
characters = {0: 48, 1: 49, 2: 50, 3: 51, 4: 52, 5: 53, 6: 54, 7: 55, 8: 56, 9: 57, 10: 65, 11: 66, 12: 67, 13: 68,
              14: 69, 15: 70, 16: 71, 17: 72, 18: 74, 19: 75, 20: 76, 21: 77, 22: 78, 23: 80, 24: 82, 25: 83, 26: 84,
              27: 85, 28: 86, 29: 87, 30: 88, 31: 89, 32: 90}


def find_prediction(image_path):
    model = tf.keras.models.load_model('model.h5')

    image = cv.imread(cv.samples.findFile(image_path), 0)
    image = cv.resize(image, (28, 28))  # resizing image as our model is trained only on 28*28 images
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=3)

    prediction = model.predict(image, verbose=0)
    prediction_index = np.argmax(prediction)

    return characters[prediction_index]


def perform_inference(path):
    try:
        img_files = [file for file in os.listdir(path) if file.endswith(('.jpg', '.png', '.jpeg'))]
        for file in img_files:
            img_path = os.path.join(path, file)
            prediction = find_prediction(img_path)
            print(f"0{prediction}, {img_path}")

    except Exception as ex:
        print(ex)


if '__init__.py':
    if len(sys.argv) < 2:
        print("Please provide the directory path")
        sys.exit(1)

    arg_path = sys.argv[1]
    perform_inference(arg_path)
