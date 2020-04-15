import numpy as np
import tf2_processing

digit_width = 20
speed_images_path = "speed_digit_data/speed_data.npy"
speed = np.load(speed_images_path)
print(speed.shape)




#  Let's transform three digit number image into three images of one digit each.
# Appending it to database
digit_db = []
for image in speed:
    first_digit, second_digit, third_digit = tf2_processing.digit_images(image)
    digit_db.append([first_digit])
    digit_db.append([second_digit])
    digit_db.append([third_digit])

np.save("speed_digit_data/digit_data.npy", digit_db)

