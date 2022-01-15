from tensorflow.keras.datasets import boston_housing
from tensorflow import keras

(train_data, train_targets), (test_data, test_targets) = (boston_housing.load_data())

# normalize data 
# suhbstract the mean of the feature and devide by standard deviation
# that way all the features center arround 0.
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

# build the model
inputs = keras.layers.Input((13,))
features = keras.layers.Dense(64, activation="relu")(inputs)
outputs = keras.layers.Dense(1)(features)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])

# train the model
model.fit(train_data, train_targets, epochs=150, batch_size=64)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(test_mae_score)

# predict the price for the houses in test data and check how close our first prediction is
# to the real price of the first house
predictions = model.predict(test_data)
predicted_value = predictions[0][0]
real_value = test_targets[0]
print(f"Predicted Price: {predicted_value} - Real Price: {real_value}")