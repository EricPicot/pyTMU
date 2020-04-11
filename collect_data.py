import os
import time
import cv2
import numpy as np
from utils import key_check, grab_screen, keys_to_output


starting_value = 1

while True:
    file_name = './tf_dataset/data/training_data-{}.npy'.format(starting_value)
    target_file_name = './tf_dataset/target/target_data-{}.npy'.format(starting_value)

    if os.path.isfile(file_name):
        print('File exists, moving along', starting_value)
        starting_value += 1
    else:
        print('File does not exist, starting fresh!', starting_value)

        break

        
def main(file_name, target_file_name, starting_value):
    file_name = file_name
    target_file_name = target_file_name

    starting_value = starting_value
    training_data = []
    target_data = []

    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    last_time = time.time()
    paused = False
    print('STARTING!!!')
    while True:

        if not paused:
            screen = grab_screen(region=(0, 40, 800, 640))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

            keys = key_check()
            output = keys_to_output(keys)
            training_data.append(screen)
            target_data.append(output)


            if len(training_data) % 50 == 0:
                print(len(training_data))

                if len(training_data) == 200:
                    np.save(file_name, training_data)
                    np.save(target_file_name, target_data)

                    print('SAVED')
                    training_data = []
                    target_data = []
                    starting_value += 1
                    file_name = './tf_dataset/data/training_data-{}.npy'.format(starting_value)
                    target_file_name = './tf_dataset/target/target_data-{}.npy'.format(starting_value)


main(file_name, target_file_name, starting_value)
