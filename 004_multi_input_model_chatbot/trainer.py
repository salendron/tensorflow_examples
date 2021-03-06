from conversation_training_data import ConversationTrainingData
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Load training data
ctd = ConversationTrainingData.from_file("conversation_training_data.json")

# define
question_inputs = keras.Input(shape=(ctd.len_inputs, ))
question_features = layers.Dense((ctd.len_samples + len(ctd.dictionary)) * 3, activation="relu")(question_inputs)

context_inputs = keras.Input(shape=(ctd.len_inputs, ))
context_features = layers.Dense((ctd.len_outputs + len(ctd.dictionary)) * 3, activation="relu")(context_inputs)

inputs = layers.Concatenate()([question_features, context_features])

outputs = layers.Dense(ctd.len_outputs, activation="softmax")(inputs)
model = keras.Model(inputs=[question_inputs, context_inputs], outputs=outputs)

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# train model
model.fit(
    [ctd.sample_question_training_data,ctd.context_training_data],
    ctd.train_labels,
    epochs=60,
    batch_size=128 
)

# save model
model.save("saved_model/chatbot")
