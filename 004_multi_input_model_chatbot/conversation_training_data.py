import json
from intent import Intent
import numpy as np
from language_helper import sentence_to_normalized_word_list

class ConversationTrainingData:
    """
    Holds all the data used to train the chatbot. 
    Data is loaded from a definiton json file, which consists of default answers
    and all intent definitions.
    It also generates a dictionary of all words in all intents, and also training
    data and labels as numeric numpy arrays to train a tensorflow model.
    """

    @staticmethod
    def from_file(filepath):
        """
        Loads and prepares the training data from the given json file
        """
        f = open(filepath)
        data = json.load(f)
        f.close()

        ctd = ConversationTrainingData()
        ctd.error_answer = data["default-answers"]["error-answer"]
        ctd.no_answer_answer = data["default-answers"]["no-answer"]

        for intent_data in data["intents"]:
            intent = Intent(
                intent_data["name"],
                intent_data["samples"],
                intent_data["context"],
                intent_data["answers"]
            )

            ctd.intents.append(intent)

        ctd.prepare_dictionary()
        ctd.prepare_training_data()

        return ctd

    def __init__(self):
        """
        initializes the object
        """
        # conversation data
        self.error_answer = ""
        self.no_answer_answer = ""
        self.intents = []
        self.dictionary = []

        # training data
        self.len_samples = 0
        self.len_inputs = 0
        self.len_outputs = 0
        self.sample_question_training_data = None
        self.context_training_data = None
        self.train_labels = None

    def prepare_dictionary(self):
        """
        iterates over all intents and extracts all words to get a list of
        all words of all intents. This is our full dictionary.
        """
        for intent in self.intents:
            words = intent.get_all_words()
            for word in words:
                if word not in self.dictionary:
                    self.dictionary.append(word)

    def prepare_training_data(self):
        """
        converts all all questions and context to arrays of len(dictionary) filled
        with zeros and ones.
        zero if a word from the dictionary does not exist in the questions or context
        ones at the position of the words that do exist, matching the position of the 
        word in the dictionary.
        """

        #define dimensions based on dictionary length and how many possible intents we have
        self.len_inputs = len(self.dictionary)
        self.len_outputs = len(self.intents)

        #since we have a model with two inputs, the questions and the current context,
        #we have to create both inputs
        train_sample_questions = []
        train_context_items = []

        #a label is also a list of zeros and ones, with a one at the position of the
        #the correct intent
        train_labels = []

        self.len_samples = 0

        #iterate over all intents
        for i, intent in enumerate(self.intents):
            #get and prepare the intent context
            context_items = intent.context
            context_items.append(None) # append empty context to build training data for each sample also without context

            #iterate over all sample questions
            for sample in intent.samples:
                #for each question generate a train input for each context item
                #and also for the case of no context, the None item we've aded before.
                for context in context_items:
                    #build sample imput
                    train_sample_input = np.zeros(len(self.dictionary))
                    for word in sample:
                        word_index = self.dictionary.index(word)
                        train_sample_input[word_index] = 1.

                    train_sample_questions.append(train_sample_input)
                    self.len_samples += 1

                    #build context context
                    train_context_input = np.zeros(len(self.dictionary))

                    if context != None:
                        word_index = self.dictionary.index(context)
                        train_context_input[word_index] = 1.

                    train_context_items.append(train_context_input)
            
                    #build the label for these two inputs (a one at the position of the current intent)
                    train_label = np.zeros(len(self.intents))
                    train_label[i] = 1.
                    train_labels.append(train_label)
                
        #convert inputs and labels to numpy arrays
        self.train_labels = np.array(train_labels, dtype=float)
        self.sample_question_training_data = np.array(train_sample_questions)
        self.context_training_data = np.array(train_context_items)

    def sentence_to_input(self,sentence):
        """
        helper method to convert a user input to a numerical input array
        so our model can process it.
        """
        words = sentence_to_normalized_word_list(sentence)

        input = np.zeros(len(self.dictionary))

        for word in words:
            if word in self.dictionary:
                word_index = self.dictionary.index(word)
                input[word_index] = 1.

        return np.array([input,])

    def context_to_input(self,context):
        """
        helper method to convert the conversation context numerical input array
        so our model can process it.
        """
        input = np.zeros(len(self.dictionary))

        for word in context:
            if word in self.dictionary:
                word_index = self.dictionary.index(word)
                input[word_index] = 1.

        return np.array([input,])

                

                
                
