import random
import numpy as np
import tf2_processing
from build_models import AlexNet
from datetime import datetime
import tensorflow as tf

# Setting tensorboard
log_dir = "./log_dir"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#  parameters:
today = datetime.today().strftime('%Y%m%d')
WIDTH = 480
HEIGHT = 270
LR = 1e-3
EPOCHS = 30
n_training_set = 14
augmentation = True
# Loading/creating model
# model = tf.keras.models.load_model("./models/alexnet10")
model = AlexNet()
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")

for e in range(EPOCHS):
    data_order = [i for i in range(0, n_training_set)]
    random.shuffle(data_order)

    for count, i in enumerate(data_order):
        try:
            file_name = './tf_dataset/balanced_data/data_balanced_{}.npy'.format(i)
            train_data = tf2_processing.process(file_name,
                                                resize=(270, 480),
                                                augmentation=augmentation,
                                                flipping_data=True)
            target_data = np.load("./tf_dataset/balanced_target/target_balanced_{}.npy".format(i))

            if augmentation:
                target_data = np.concatenate([target_data, tf2_processing.flipping_target(target_data)])

            print("EPOCH ", e, '  training_data-{}.npy  '.format(i), len(train_data))

            model.fit(train_data,
                      target_data,
                      epochs=1,
                      validation_split=0.1,
                      verbose=1,
                      callbacks=[tensorboard_callback])

            if count % 10 == 0:
                print(count, ' SAVING MODEL!')
                model.save("./models/alexnet{0}_{1}".format(count, today))

        except Exception as ex:
            print("Problem on data ", str(i), str(e))
