from tensorflow.keras.datasets import imdb
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# max number of words we take from each review - 10000 most used words
NUM_WORDS = 10000

# how many reviews we are going to use to validate the model
VALIDATION_DATA_SIZE = 10000

# load the date and split it up into training and test
(train_data, trains_labels), (test_data, test_labels) = imdb.load_data(num_words=NUM_WORDS)

# vectorize the data 
# this will result in a list of len(sequences) list that consist of 10000 zeros 
# (if a word does not exist in the sequence) and ones (for every with word that
# does exist in the sequence) 
def vectorize__sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.

    return results

x_train = vectorize__sequences(train_data, NUM_WORDS)
x_test = vectorize__sequences(test_data, NUM_WORDS)

# convert the labels to numpy array of float32
y_train = np.asarray(trains_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

# split up training data and validation data
x_val = x_train[:VALIDATION_DATA_SIZE]
partial_x_train = x_train[VALIDATION_DATA_SIZE:]
y_val = y_train[:VALIDATION_DATA_SIZE]
partial_y_train = y_train[VALIDATION_DATA_SIZE:]

# define the model
model = keras.Sequential([
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

# train the model
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val),
)

# get the training metrics
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
acc_values = history_dict["accuracy"]
val_acc_values = history_dict["val_accuracy"]

epochs = range(1, len(loss_values) + 1)

# visualize it using matplotlib
plt.plot(epochs, loss_values, "bo", label="Training Loss")
plt.plot(epochs, val_loss_values, "b", label="Validation Loss")
plt.plot(epochs, acc_values, "ro", label="Training Accuracy")
plt.plot(epochs, val_acc_values, "r", label="Validation Accuracy")
plt.legend()
plt.show()

# get predictins on never before seen data (test data)
predictions = model.predict(x_test)
print(predictions)