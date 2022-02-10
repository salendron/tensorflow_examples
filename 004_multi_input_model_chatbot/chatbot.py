from tensorflow.keras.models import load_model
import numpy as np
from conversation_training_data import ConversationTrainingData

#load the trained model
model = load_model("saved_model/chatbot")

#load the conversation data, so we can answer with the defined answers.
ctd = ConversationTrainingData.from_file("conversation_training_data.json")

#the current conversation context
context = []

#conversation loop
while True:
    #read user input
    sentence = input(":")

    #conver input and current context to numerical arrays so they can be
    #used to predict the correct intent
    input_sentence = ctd.sentence_to_input(sentence)
    input_context = ctd.context_to_input(context)

    #predict the correct input
    prediction = model.predict([input_sentence, input_context])[0]
    intent_idx = np.argmax(prediction)
    predicted_intent = ctd.intents[intent_idx]

    #check if the probability is high enough and either answer with the defined
    #answer of the predicted intent or otherwise tell the user that we do not
    #understand.
    if prediction[np.argmax(prediction)] > 0.2:
        print(sentence + ":" + predicted_intent.get_answer())
    else:
        print(sentence + ": I am not sure about that.")

    #remember the current context for the next question
    context = predicted_intent.context