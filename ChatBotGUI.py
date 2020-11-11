# ==============Basic Libraries===========================#
import numpy as np
import pandas as pd
import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
word_lem = WordNetLemmatizer()
from nltk.corpus import stopwords


# ============Deep Learning Libraries====================#
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.models import load_model


# ===========Miscellaneous Files==========================#
import random
import re
import json
import pickle

# ===========Kivy Tools===================================#
import kivy
from kivy.app import App
from kivy.uix.widget import Widget  # this is needed to add buttons, entry boxes and many more
from kivy.core.window import Window
from kivy.utils import get_color_from_hex
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.scrollview import ScrollView
from kivy.clock import Clock

# For defining the color of the background
Window.clearcolor = get_color_from_hex('#a87f32')

# This will represent main window
class MainWindow(Screen):
    pass

# ===============BMI calculation starts====================================================#
def bmi_metric(height, weight):

    h = height
    h_metres = h/100
    w = weight
    bmi = w/(h_metres**2)
    checker = ''

    return bmi

def bmi_checker(bmi):
    checker = ''
    if bmi <= 18.5:
        checker = "You are underweight!"
    elif 18.5 < bmi <= 24.9:
        checker = "You are within normal weight range."
    elif 25 <= bmi < 29.9:
        checker = "You are overweight!"
    else:
        checker = "You are obese!"
    return checker
# ===============BMI calculation ends====================================================#


# ===============NLP pre-processing part ================================================#
# Opening the trained model
model = keras.models.load_model('nutribot_model.h5')
# Opening the pickle file for vocabulary and encoders
with open("data.pickle", "rb") as f:
    vocab_list, labels, training_responses, training_labels = pickle.load(f)

# The intents JSON file to look up the responses
intents_file = open('intents_nutribot.json').read()
intents = json.loads(intents_file)

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


def bag_of_words(sentence, words):
    # create a '0' list with the length of words
    bag = [0 for _ in range(len(words))]

    # Lemmatization and tokenize the sentences
    sentenceProcess = nltk.word_tokenize(sentence)
    sentenceProcess = [word_lem.lemmatize(s.lower(), get_wordnet_pos(s)) for s in sentenceProcess]

    # assign 1s and 0s to give BOW representation
    for s in sentenceProcess:
        # i for row-by-row iteration
        # w for iterating through each element in a row
        for i, w in enumerate(words):
            if w == s:
                # assign 1 is current word is in the vocab position
                bag[i] = 1

    # return as a list
    return (np.array(bag))


# function to start the chat
def predict_label(sentence, label, intents_json):
    # The response we get is a probability distribution of nodes (softmax)
    # We should filter the prediction thresholds to keep the largest.
    b = bag_of_words(sentence, vocab_list)

    results = model.predict(np.array([b]))[0] # This means pick the first list. (we have list of lists as output)
    result_index = np.argmax(results) # This returns the index of the largest value in the predicted results
    tag = label[result_index] # Labels stores all our labels and the index of the

    if results[result_index] > 0.7:

        for tg in intents_json['intents']:
            if tg['tag'] == tag:
                response = random.choice(tg['responses'])
                break

        return response
    else:

        return "I'm sorry, I don't understand you!"
# ===============NLP pre-processing part ends================================================#

# This class will represent the BMI application
class BMIWindow(Screen):

    var2 = ObjectProperty(None)
    output = ObjectProperty(None)
    var1 = ObjectProperty(None)

    def bmi_btn(self):
        self.output.text = ""
        try:
            if len(self.var1.text) > 0 and len(self.var2.text) > 0:
                w = float(self.var2.text)
                h = float(self.var1.text)
                bmi = bmi_metric(h, w)
                check = bmi_checker(bmi)
                self.output.text += "Weight: " + str(w) + "kg" + " and " + "Height: " + str(h) + "cm" + "\n"
                self.output.text += "Your BMI is: " + str(bmi) + "\n"
                self.output.text += check
                self.reset()
            else:
                self.output.text = "Error: Please type in values"
                self.reset()

        except ValueError:
            self.output.text = "Input is not a number, it's a string try again!"
            self.reset()


    def PopUpBtn(self):
        popup = PopUp()
        popup.open()

    def reset(self):
        self.var1.text = ""
        self.var2.text = ""



# This class will represent the chatbot application!
class ChatWindow(Screen):
    message = ObjectProperty(None)
    message_log = ObjectProperty(None)

    def btn(self):
        msg = self.message.text
        m_token = msg.split()
        names = ['sofian', 'rasha', 'Sofian', 'Rasha']
        if m_token[0] in names:
            self.message_log.text += "You: " + msg + "\n"
            self.message_log.text += "Bot: Hey there " + msg + "!. " + "You can ask me anything related to healthy eating, weight control tips or if you have thyroid issues. Make sure you elaborate on what you need!" + "\n"
            self.reset()

        else:
            self.message_log.text += "You: " + msg + "\n"
            result = predict_label(msg, labels, intents)
            self.message_log.text += "Bot: " + result + "\n"
            self.reset()


    def reset(self):
        self.message.text = ""

# This class will represent the transitions and stuff between the windows
class WindowManager(ScreenManager):
    pass

# This class will be for the pop-up
class PopUp(Popup):
    pass

class MessageArea(BoxLayout):
    pass

kv = Builder.load_file("my.kv")


class ChatbotApp(App):
    def build(self):
        return kv

ChatbotApp().run()