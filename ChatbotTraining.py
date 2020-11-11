# Chat-bot Training

# Pre-processing
import numpy as np
import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.tokenize import blankline_tokenize
from nltk.util import bigrams, trigrams, ngrams
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
word_lem = WordNetLemmatizer()
from nltk.corpus import stopwords
import random
import regex as re

# Deep learning libraries
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.models import load_model

# For loading and writing files
import json
import pickle

# Loading the files

intents_file = open('intents_nutribot.json').read()
intents = json.loads(intents_file)

# Grouping the responses, and tags in separate files.

labels = [] # the unique intents
responses = [] # these are patterns corresponding to the responses a user may ask.
doc_x = [] # tokenized form of responses. Used later for BOW model
doc_y = [] # these are the intent tags that will directly correspond to that particular response. Used later for BOW model

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize the pattern sentences
        word = nltk.word_tokenize(pattern)
        # append method only add one element in list per iteration. That is 'How are you?' -> ['How','are', 'you', '?'] (one element)
        doc_x.append(word)
        # add the tag sentences in doc_y file
        doc_y.append(intent["tag"])

        responses.append(pattern)

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

labels = sorted(labels)


# Creating Vocabulary

lower_case = []
tokens = []
punctuations = []

# Lower-case the sentences
for i in responses:
    lower_case.append(i.lower())

# Creating a contractions dictionary -> essential for converting abbreviated words into proper words.
contractions_dict = {
    "ain't": "am not / are not / is not / has not / have not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is / how does",
    "i'd": "i had / i would",
    "i'd've": "i would have",
    "i'll": "i shall / i will",
    "i'll've": "i shall have / i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have",
    "plz": "please",
    "pls": "please",
    "wassup": "what has / what is",
    "whaat": "what",
}

# Defining a function to replace contracted words.

# re function specifies a set of strings that matches it.
# We loop through each word in the sentences, match it with contractions dictionary, and if it matches, we replace with
# key-value pair.
contractions = re.compile('(%s)' % '|'.join(contractions_dict.keys()))


def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]

    return contractions.sub(replace, s)


contractions_list = [expand_contractions(i) for i in lower_case]

for i in contractions_list:
    word = nltk.word_tokenize(i)
    tokens.extend(word)

punctuation = re.compile(r'[-,?!;()./\|0-9]')
# scan every word in thelist and sub the '?','.'.... with backspace (delete)
for i in tokens:
    j = punctuation.sub("", i)

    # if backspace is there, ignore it and only consider the words that have length > 0
    if len(j) > 0:
        punctuations.append(j)


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    # get the POS tag for each words and turn into upper case ('PRP,NPN...')
    # [0] -> first row, [1] -> second element, [0] -> first letter
    tag = nltk.pos_tag([word])[0][1][0].upper()
    # Make a dictionary to map POS -> wordnet dictionary commands
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    # get() function is needed to get the key-value pair. by default, it will be wordnet.NOUN if none matches.
    return tag_dict.get(tag, wordnet.NOUN)


lemma = [word_lem.lemmatize(w, get_wordnet_pos(w)) for w in punctuations]

stop_words = set(stopwords.words('english'))

stop_words_list = [w for w in lemma if not w in stop_words]

vocab_list = stop_words_list

vocab_list = sorted(list(set(vocab_list)))

print("Making lower case.. ")
print(lower_case)

print("Expanding contraction words.. ")
print(contractions_list)

print("Punctuation removal.. ")
print(punctuations)
print("The number of words are: " + str(len(punctuations)) + "\n")

print("Lemmatizing words: ")
print(lemma)
print("The number of words are: " + str(len(lemma)) + "\n")

print("Removing stop words.. ")
print(stop_words_list)
print("The number of words are: " + str(len(stop_words_list)) + "\n")

print("Final Vocabulary List: ")
print(vocab_list)
print("The number of words are: " + str(len(vocab_list)) + "\n")
print("Vocabulary Building Complete..")

# Bag-of-Words Model

def bag_of_words(response, unique_label, response_label, vocabulary):
    out_empty = [0 for _ in range(len(unique_label))]
    training = []
    output = []
    # x -> the row to consider.
    # doc -> the words within that sentence such as: ['How', 'are', 'you']
    for x, doc in enumerate(response):
        # Bag-of-words
        bag = []

        # Lemmatizing the doc_x file

        Docs = [word_lem.lemmatize(w.lower(), get_wordnet_pos(w)) for w in doc]

        # Now we are trying to match the lemmatized sentences in doc_x to the lemmatizedWords.
        # If it does, add a '1' else add '0'
        for w in vocabulary:
            if w in Docs:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]

        # We are going to look through labels list and see where the tag is and set that value to 1 in output row
        # labels -> unique values of tag
        # doc_y -> just values
        output_row[unique_label.index(response_label[x])] = 1

        training.append(bag)
        output.append(output_row)

    return training, output


training_responses, training_labels = bag_of_words(doc_x, labels, doc_y, vocab_list)

training_responses = np.array(training_responses)
training_labels = np.array(training_labels)

print("Training Response Vector Matrix: " + "\n")
print(training_responses)

print("Training Label Vector Matrix: " + "\n")
print(training_labels)

print("Training set Dimensions: " + str(len(training_responses)) + " by " + str(len(training_responses[0])))
print("Testing set Dimensions: " + str(len(training_labels)) + " by " + str(len(training_labels[0])))

# Training with Deep Neural Network

model = Sequential()
# first layer
model.add(Dense(128, input_shape=(len(training_responses[0]),), activation='relu'))
model.add(Dropout(0.5))
# second layer
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# third layer - > output with layer size of 14
model.add(Dense(len(training_labels[0]), activation='softmax'))

# Compiling model...

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# Training and saving the model
hist = model.fit(training_responses, training_labels, epochs=1000, batch_size=32, verbose=1)

print("model is created")

# Saving the results of the model and other documents for predictions later on.

with open("data.pickle", "wb") as f:
    pickle.dump((vocab_list, labels, training_responses, training_labels), f)

model.save('nutribot_model.h5', hist)
