import os
import time
import cv2
import numpy as np
from utils import key_check, grab_screen, keys_to_output, speed_numerisation
import tensorflow as tf

digit_model = tf.keras.models.load_model("./models/digit_model")
starting_value = 1
speed_region = (730, 615, 800, 645)

while True:
    file_name = './tf_dataset/data/training_data-{}.npy'.format(starting_value)
    target_file_name = './tf_dataset/target/target_data-{}.npy'.format(starting_value)
    speed_file_name = './tf_dataset/speed/target_data-{}.npy'.format(starting_value)
    if os.path.isfile(file_name):
        print('File exists, moving along', starting_value)
        starting_value += 1
    else:
        print('File does not exist, starting fresh!', starting_value)

        break

        
def main(file_name, target_file_name, starting_value, speed_file_name):
    file_name = file_name
    target_file_name = target_file_name
    speed_file_name = speed_file_name
    starting_value = starting_value
    training_data = []
    target_data = []
    speed_data = []
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = False
    print('STARTING!!!')
    while True:

        if not paused:
            screen = grab_screen(region=(0, 40, 800, 640))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            speed = grab_screen(region=speed_region)
            speed = cv2.cvtColor(speed, cv2.COLOR_BGR2RGB)

            training_data.append(screen)
            keys = key_check()
            output = keys_to_output(keys)
            target_data.append(output)

            speed_value = speed_numerisation(speed, model=digit_model)
            speed_data.append(speed_value)

            if len(speed_data) % 50 == 0:
                print(len(speed_data))

                if len(speed_data) == 200:
                    np.save(file_name, training_data)
                    np.save(target_file_name, target_data)
                    np.save(speed_file_name, speed_data)
                    print('SAVED')
                    training_data = []
                    target_data = []
                    speed_data = []
                    starting_value += 1
                    file_name = './tf_dataset/data/training_data-{}.npy'.format(starting_value)
                    target_file_name = './tf_dataset/target/target_data-{}.npy'.format(starting_value)
                    speed_file_name = './tf_dataset/speed/target_data-{}.npy'.format(starting_value)


main(file_name, target_file_name, starting_value, speed_file_name)