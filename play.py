import os
import time

import keyboard as kb
import tensorflow as tf

import tf2_processing
from utils import prediction_to_keys, grab_screen

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# model = tf.keras.models.load_model("./models/model_alexnet_V0_500e_500i")
model = tf.keras.models.load_model("./models/alexnet10")

def main():

    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    last_time = time.time()
    paused = False
    print('STARTING!!!')
    while True:

        if not paused:
            screen = grab_screen(region=(0, 40, 800, 640))
            screen = tf2_processing.process_image(screen, resize=(270, 480))
            prediction = model.predict(screen)
            keys = prediction_to_keys(prediction)
            kb.press(keys)
            time.sleep(0.20)
            kb.press("up")
            time.sleep(0.15)
            kb.release(keys)

main()