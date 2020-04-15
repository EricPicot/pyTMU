import numpy as np
import tf2_processing
from build_models import digit_model

def transform_target(target):
    target_list = []
    for t in target:
        transformed_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        transformed_target[t] = 1
        target_list.append(transformed_target)
    return np.array(target_list).reshape((len(target_list),10))


data = tf2_processing.read_image("./digit/speed_digit_data/digit_data.npy")
target = np.load("./digit/speed_digit_data/digit_target.npy").astype(int)
target = transform_target(target).astype('float32').reshape((-1,10))

model = digit_model()
model.compile(optimizer="adam", loss="categorical_crossentropy")
model.fit(data, target, epochs=50, verbose=1)
model.save('./models/digit_model')
