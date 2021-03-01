import os
import pickle
from math import ceil

import matplotlib.pyplot as plt
from keras.layers import Lambda, Flatten, Dense, Convolution2D, Dropout
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from labeler import Labeler


def load_data(data):
    images = data["features"]
    labels = data["labels"]
    return images, labels


def generator(samples, batch_sz=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_sz):
            batch_samples = samples[offset : offset + batch_sz]

            tmp_features, tmp_labels = load_data(batch_samples)
            yield shuffle(tmp_features, tmp_labels)


# ======== MAIN ========
file_name = "./pickle_rick.p"
assert os.path.isfile(file_name), "data file not found"

labeler = Labeler("rb")
data = labeler.load()
data["features"] = data["features"].reshape(-1, 800, 600, 3)

print("Labels: " + str(data["labels"].size))
print("Features: " + str(data["features"].size))
print("Shape: " + str(data["features"].shape))

feature_shape = [600, 800, 3]

model = None
if os.path.isfile("model.h5"):
    model = load_model("model.h5")
elif feature_shape is not None:
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=feature_shape))
    #    tmp_model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")

train_samples, validation_samples, train_labels, validation_labels = train_test_split(
    data["features"], data["labels"], test_size=0.2
)

print("train_samples: " + str(train_samples.size))
print("validation_samples: " + str(validation_samples.size))
print("train_labels: " + str(train_labels.size))
print("validation_labels: " + str(validation_labels.size))

# Set our batch size
batch_size = 32

# compile and train the model using the generator function
train_data = {"features": train_samples, "labels": train_labels}
validation_data = {"features": validation_samples, "labels": validation_labels}
train_generator = generator(train_data, batch_sz=batch_size)
validation_generator = generator(validation_data, batch_sz=batch_size)

history_object = model.fit_generator(
    train_generator,
    steps_per_epoch=ceil(len(train_samples) / batch_size),
    validation_data=validation_generator,
    validation_steps=ceil(len(validation_samples) / batch_size),
    epochs=5,
    verbose=1,
)
model.save("model.h5")
# print the keys contained in the history object
print(history_object.history.keys())


# plot the training and validation loss for each epoch
plt.plot(history_object.history["loss"])
plt.plot(history_object.history["val_loss"])
plt.title("model mean squared error loss")
plt.ylabel("mean squared error loss")
plt.xlabel("epoch")
plt.legend(["training set", "validation set"], loc="upper right")
plt.show()
