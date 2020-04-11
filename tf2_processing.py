import random
import numpy as np
import tensorflow as tf

def read_image(path):
    tf_content = np.load(path, allow_pickle=True)
    tf_content = np.asarray(tf_content).astype(np.float32) / 255

    return tf_content


def resize_and_scale_gray(tf_image):
    """
    - normalize values of the tensor
    - resize images to (1, 270, 480, 3) images
    :param tf_image: Tensor of shape [X, X, 3]
    :return: Tensor of shape [1, 270, 480, 3]
    # """
    tf_image_resized = tf.image.resize(tf_image, (270, 480))
    tf_image_resized = tf.reshape(tf_image_resized, (-1, 270, 480, 3))
#     print("before enhancement")
    # Random Brightness
    tf_image = tf.image.random_brightness(tf_image_resized, max_delta=random.uniform(0, 1))
    # Random Saturation
    tf_image = tf.image.random_saturation(tf_image, lower=0.5, upper=1.5)
    # Random Contrast
    tf_image = tf.image.random_contrast(tf_image, lower=0.5, upper=1.5)
    tf_image = tf.image.convert_image_dtype(tf_image, dtype=tf.float32)
#     print("before gray scaling")
    tf_image_gray = tf.image.rgb_to_grayscale(tf_image)
#     print("before sobel")
    tf_image_gray = tf.image.sobel_edges(tf_image_gray)
    tf_image = tf.squeeze(tf_image_gray)
#     print("after sobel")
    return tf.where(tf_image > 0.4, tf.ones_like(tf_image), tf.zeros_like(tf_image))


def flipping_target(target):
    target = np.array(target)
    for i,t in enumerate(target):
        t = np.array(t)
        argmax = t.argmax()
        if argmax in [2,4,6]:
            target[i][argmax+1] = 1
            target[i][argmax] = 0
        elif argmax in [3,5,7]:
            target[i][argmax-1] = 1
            target[i][argmax] = 0
    return target
            
            
    
# def flipping_action(target):
def flipping_tensor(tf_image):
    """
    flipping the image to double the size of the dataset and to balanced a bit more the dataset regarding turns
    """
    tf_image = tf.concat([tf_image, tf.image.flip_left_right(tf_image)], axis = 0)
    return tf_image


def process_image(image):

    image = image / 255

    im_tensor = resize_and_scale_gray(image)
    im_tensor = tf.image.crop_to_bounding_box(
        im_tensor, offset_height=120, offset_width=0, target_height=150, target_width=480
    )

    im_tensor = tf.image.resize(im_tensor, (270, 480))
    # print(im_tensor)
    return tf.reshape(im_tensor[ :, :, 0], (-1, 270, 480, 1))


def process(data_path):
    im_tensor = read_image(data_path)
    im_tensor = resize_and_scale_gray(im_tensor)
    im_tensor = tf.image.crop_to_bounding_box(
        im_tensor, offset_height=120, offset_width=0, target_height=150, target_width=480
    )
    im_tensor = tf.image.resize(im_tensor, (270, 480))
    im_tensor = flipping_tensor(im_tensor)

    return tf.reshape(im_tensor[:, :, :, 0], (-1, 270, 480, 1))

#
# import matplotlib.pyplot as plt
#
# im_transformed = process_image(data)
#
# print(im_transformed.shape)
# plt.figure(figsize=(14,10))
# plt.subplot(2,2,1)
# plt.imshow(im_transformed[0, :, :, 0].numpy().reshape((270, 480)))
#
# # plt.subplot(2,2,2)
# # plt.imshow(im_transformed[0, :, :, 1].numpy().reshape((270, 480)))
#
