# How this sample works
* Define your conversation in conversation_training_data.json by defining intents, consisting of questions and answers and also the context, which basically is a single word or list of words, which define what this is about. For example if the answer is something about bananas, the context is "banana". This allows the user to ask a follow up questions like "what color do they have?" and the chatbot still knows that this question is about bananas.
* Train the model py running trainer.py, which will save the model to "./saved_model/"
* Run chatbot.py and start to communicate with the bot via a command line chat