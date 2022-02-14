import random
from language_helper import sentence_to_normalized_word_list

class Intent:
    """
    Holds a single intent, which consists of a name, sample questions
    context words, and answers for the sample questions.
    """

    def __init__(self, name, samples, context, answers):
        self.name = name
        self.samples = self.prepare_sample(samples)
        self.context = self.prepare_context(context)
        self.answers = answers

    def prepare_sample(self, inputs):
        """
        returns a cleaned list of input samples
        """
        prepared_inputs = []

        for input in inputs:
            prepared_inputs.append(sentence_to_normalized_word_list(input))

        return prepared_inputs

    def prepare_context(self, inputs):
        """
        returns a cleaned list of inputs (sample questions or context).
        It replaces things like dots and questions marks and also converts
        everything to uppercase.
        """
        prepared_inputs = []

        for input in inputs:
            prepared_inputs += sentence_to_normalized_word_list(input)

        return prepared_inputs

    def get_answer(self):
        """
        returns a random answer sentence of the defined answers for this intent
        """
        return random.choice(self.answers)

    def get_all_words(self):
        """
        return a list of all words in a all sample questions and all context words
        """
        words = []

        for word in self.context:
            if word not in words:
                words.append(word)

        for sample in self.samples:
            for word in sample:
                if word not in words:
                    words.append(word)

        return words