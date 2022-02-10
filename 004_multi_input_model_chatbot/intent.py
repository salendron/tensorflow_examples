import random

class Intent:
    """
    Holds a single intent, which consists of a name, sample questions
    context words, and answers for the sample questions.
    """

    def __init__(self, name, samples, context, answers):
        self.name = name
        self.samples = self.prepare_inputs(samples)
        self.context = self.prepare_inputs(context)
        self.answers = answers

    def prepare_inputs(self, inputs):
        """
        returns a cleaned list of inputs (sample questions or context).
        It replaces things like dots and questions marks and also converts
        everything to uppercase.
        """
        prepared_inputs = []

        for input in inputs:
            input = input.replace(".", " ")
            input = input.replace(",", " ")
            input = input.replace("?", " ")
            input = input.replace("!", " ")
            input = input.replace("\"", " ")
            input = input.replace("'", " ")
            input = input.replace("-", " ")
            input = input.replace("/", " ")
            input = input.replace("\\", " ")
            input = input.upper()

            prepared_inputs.append(input)

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
            sample_words = sample.split(" ")
            for word in sample_words:
                if word not in words:
                    words.append(word)

        return words