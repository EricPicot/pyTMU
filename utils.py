import numpy as np
import keyboard as kb
import cv2
import numpy as np
from PIL import ImageGrab


up = [1, 0, 0, 0, 0, 0, 0, 0, 0]
down = [0, 1, 0, 0, 0, 0, 0, 0, 0]
right = [0, 0, 1, 0, 0, 0, 0, 0, 0]
left = [0, 0, 0, 1, 0, 0, 0, 0, 0]
up_right = [0, 0, 0, 0, 1, 0, 0, 0, 0]
up_left = [0, 0, 0, 0, 0, 1, 0, 0, 0]
down_right = [0, 0, 0, 0, 0, 0, 1, 0, 0]
down_left = [0, 0, 0, 0, 0, 0, 0, 1, 0]
nothing = [0, 0, 0, 0, 0, 0, 0, 0, 1]


def grab_screen(region=None):

    if region:
            left, top, x2, y2 = region
            width = x2 - left + 1
            height = y2 - top + 1

    img = np.array(ImageGrab.grab(bbox=(0, 40, 800, 600)))

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

def key_check():
    key_list = []
    key = kb.get_hotkey_name()
    if key:
        key_list.append(key)
        key_list = list(set(key_list))
    return key_list


def prediction_to_keys(prediction):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''
    if prediction.argmax() == 0:
        output = "up"
    elif prediction.argmax() == 1:
        output = "down"
    elif prediction.argmax() == 2:
        output = "right"
    elif prediction.argmax() == 3:
        output = "left"
    elif prediction.argmax() == 4:
        output = "right+up"
    elif prediction.argmax() == 5:
        output = "left+up"
    else:
        output = "up"

    print(prediction, output)
    return output

def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''
    output = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    if 'up' in keys:
        output = up
    elif 'down' in keys:
        output = down
    elif 'right' in keys:
        output = right
    elif 'left' in keys:
        output = left
    elif 'right+down' in keys:
        output = down_right
    elif 'right+up' in keys:
        output = up_right
    elif 'left+down' in keys:
        output = down_left
    elif 'left+up' in keys:
        output = up_left
    else:
        output = nothing
    return output