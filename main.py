import re
import yaml
import numpy as np
import nltk
from nltk.stem import SnowballStemmer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import random


class AI:

    def __init__(self):
        self.intents = []
        self.words = []
        self.data = []
        self.labels = []

        self.model = None

        self.stemmer = SnowballStemmer('english')
        self.label_encoder = LabelEncoder()

        self.load_data()

    def load_data(self):
        with open("intents.yml") as stream:
            self.intents = yaml.safe_load(stream)

        for intent in self.intents:
            for pattern in intent['patterns']:

                pattern = re.sub(r'[^\w\s]', '', pattern).lower()

                tokenized_words = nltk.word_tokenize(pattern)
                stemmed_words = [self.stemmer.stem(w) for w in tokenized_words]

                self.words.extend(stemmed_words)
                self.data.append(stemmed_words)
                self.labels.append(intent['context'])

        # Remove repeated words
        self.words = set(self.words)

    # ------------------------------------------------------------------------------

    def train(self):

        # Create training data
        x = []
        y = []

        y = self.label_encoder.fit_transform(self.labels)

        for wrds in self.data:

            container = []

            for w in self.words:
                if w in wrds:
                    container.append(1)
                else:
                    container.append(0)

            x.append(container)

        x = np.array(x)
        y = np.array(y)

        # Create model
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(len(x[0]))),
            tf.keras.layers.Dense(10),
            tf.keras.layers.Dense(10),
            tf.keras.layers.Dense(len(set(y)), activation='softmax')
        ])

        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy'])

        history = self.model.fit(x, y, epochs=500, batch_size=32, verbose=2)

        # Generate model plot
        tf.keras.utils.plot_model(self.model,
                                  show_shapes=True,
                                  show_layer_activations=True,
                                  to_file='images/model.png')

        # Generate training result plot
        suptitle = 'Training result'
        title = f'Words: {len(self.words)}'
        title += ' / '
        title += f'Sentences: {len(self.data)}'
        title += ' / '
        title += f'Labels: {len(set(y))}'

        pd.DataFrame(history.history).plot()
        plt.suptitle(suptitle)
        plt.title(title, fontsize=12, color='#fd4d4d')
        plt.xlabel('Epochs')
        plt.savefig('images/training_result.png')

    # ------------------------------------------------------------------------------

    def encode_words(self, input):
        """
            Accept input as string to normalize. Return a container of normalized input.
        """

        tokenized_input = nltk.word_tokenize(input)
        stemmed_input = [self.stemmer.stem(i) for i in tokenized_input]

        container = []

        for w in self.words:
            if w in stemmed_input:
                container.append(1)
            else:
                container.append(0)

        # Change shape from (x, ) to (1, x)
        return np.expand_dims(container, axis=0)

    def predict(self, query):
        probabilities = self.model.predict(query, verbose=0)

        result = np.argmax(probabilities)

        context = self.label_encoder.inverse_transform([result])[0]

        print(f'Prediction: {context}')
        print(f'Probability: {np.amax(probabilities) * 100:.2f}%')

        return context

    def chat(self):
        if self.model is None:
            return print('I am not trained...cannot start conversation...')

        print("I'm ready to chat!")

        while True:
            inp = input("You: ")
            if inp.lower() == "quit":
                break

            query = self.encode_words(inp)

            context = self.predict(query)

            responses = next(intent['responses'] for intent in self.intents
                             if intent['context'] == context)

            print(random.choice(responses))


if __name__ == '__main__':
    ai = AI()

    ai.train()

    ai.chat()
