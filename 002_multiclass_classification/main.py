from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# amount of word we will use to classify the data
NUM_WORDS = 10000

# the reuters dataset consists of newswires in 46 categories, thats why we train our model
# to generate probabilites for 46 classes.
NUM_CLASSES = 46

# the amount of data we will use to validate our model
VALIDATION_DATA_SIZE = 1000

# load the dataset
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=NUM_WORDS)

# vectorize data
def vectorize__sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.

    return results

x_train = vectorize__sequences(train_data, NUM_WORDS)
x_test = vectorize__sequences(test_data, NUM_WORDS)

# one-hot encode labels
y_train = to_categorical(train_labels) 
y_test = to_categorical(test_labels)

# extract validation data from training data
x_val = x_train[:VALIDATION_DATA_SIZE]
partial_x_train = x_train[VALIDATION_DATA_SIZE:]
y_val = y_train[:VALIDATION_DATA_SIZE]
partial_y_train = y_train[VALIDATION_DATA_SIZE:]

# implement the model
inputs = keras.Input(shape=(NUM_WORDS, ), name="input")
features = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(features)
model = keras.Model(inputs=inputs, outputs=outputs)

# compile the model
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# train the model
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=9,
    batch_size=512,
    validation_data=(x_val, y_val),
)

# evaluate the model
results = model.evaluate(x_test, y_test)
print(results) # train acc, validation acc


# visualize training
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
acc_values = history_dict["accuracy"]
val_acc_values = history_dict["val_accuracy"]

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, "bo", label="Training Loss")
plt.plot(epochs, val_loss_values, "b", label="Validation Loss")
plt.plot(epochs, acc_values, "ro", label="Training Accuracy")
plt.plot(epochs, val_acc_values, "r", label="Validation Accuracy")
plt.legend()
plt.show()

# predict category of each newswire in test data
predictions = model.predict(x_test)
print(
    np.argmax(predictions[0])
) #prediction for first data unit

