import os

import numpy as np

data_path = './tf_dataset/data/'
target_path = './tf_dataset/target/'
speed_path = './tf_dataset/speed/'

num_datasets = 2
index = 0

# samplig factors
up_sample = 1 / 3.5
nothing_sample = 1 / 2

data_final = []
target_final = []
speed_final = []

starting_value = 5
existing_value = 1

while True:
    file_name = './tf_dataset/balanced_data/data_balanced_{}.npy'.format(starting_value)
    target_file_name = './tf_dataset/balanced_target/target_balanced_{}.npy'.format(starting_value)
    speed_file_name = './tf_dataset/balanced_speed/speed_balanced_{}.npy'.format(starting_value)

    if os.path.isfile(file_name):
        print('File exists, moving along', starting_value)
        starting_value += 1
    else:
        print('File does not exist, starting fresh!', starting_value)

        break


def sample(L, factor=1):
    return np.random.choice(L, size=int(np.array(L).shape[0] * factor), replace=False).tolist()


for data_part in range(0, num_datasets):
    print(data_part)
    print("data_balanced_{}.npy".format(existing_value+data_part))
    up = []
    right = []
    left = []
    upright = []
    upleft = []
    down = []
    downleft = []
    downright = []
    nothing = []
    shape = []

    data_temp = np.load(data_path + 'training_data-{}.npy'.format(existing_value + data_part))
    target_temp = np.load(target_path + 'target_data-{}.npy'.format(existing_value + data_part))
    speed_temp = np.load(speed_path + 'target_data-{}.npy'.format(existing_value + data_part))

    #
    # shape += len(target)
    for i, t in enumerate(target_temp):
        if np.argmax(t) == 0:
            up.append(i)
        elif np.argmax(t) == 2:
            right.append(i)
        elif np.argmax(t) == 3:
            left.append(i)
        elif np.argmax(t) == 4:
            upright.append(i)
        elif np.argmax(t) == 5:
            upleft.append(i)
        elif np.argmax(t) == 1:
            down.append(i)
        elif np.argmax(t) == 6:
            downright.append(i)
        elif np.argmax(t) == 7:
            downleft.append(i)
        else:
            nothing.append(i)

    final_id = sample(up, up_sample) + sample(down) + sample(left) + sample(right) + sample(upleft) + \
               sample(upright) + sample(downright) + sample(downleft) + sample(nothing, nothing_sample)
    if (data_part == 1) | (len(data_final) == 0):
        data_final = data_temp[final_id]
        target_final = target_temp[final_id]
        speed_final = speed_temp[final_id]
    else:
        data_final = np.concatenate([data_final, data_temp[final_id]], )
        target_final = np.concatenate([target_final, target_temp[final_id]], )
        speed_final = np.concatenate([speed_final, speed_temp[final_id]], )

    if (len(target_final) > 200) | (data_part == num_datasets-1):
        print(len(target_final))
        np.save('./tf_dataset/balanced_data/data_balanced_{}.npy'.format(starting_value + index), data_final)
        np.save('./tf_dataset/balanced_target/target_balanced_{}.npy'.format(starting_value + index), target_final)
        np.save('./tf_dataset/balanced_speed/speed_balanced_{}.npy'.format(starting_value + index), speed_final)

        index += 1
        data_final = []
        target_final = []
        speed_final = []
        print(index)
