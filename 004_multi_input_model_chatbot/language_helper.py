from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

def sentence_to_normalized_word_list(input):
    """
    converts a sentence into a list of uppercase words in their base form.
    """
    words = word_tokenize(input)
    return [WordNetLemmatizer().lemmatize(word, 'v').upper() for word in words]