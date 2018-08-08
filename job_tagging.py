import scipy
import numpy as np
import pandas as pd
import keras
import re
import pickle
import os
import h5py
from keras.models import load_model
from keras.optimizers import Adam, RMSprop
from keras.preprocessing import text
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.preprocessing import LabelBinarizer

def train_model():
    job_post_data = pd.read_csv("reed_uk.csv")

    # divide between training and test data
    max_num_words_in_list = 375172*1/4


    posts = job_post_data["job_description"]
    tags = job_post_data["category"]


    train_size = int(len(posts) * .8)

    train_posts = posts[:train_size]
    train_tags = tags[:train_size]

    test_posts = posts[train_size:]
    test_tags = tags[train_size:]


    vocab_size = int(max_num_words_in_list)

    tokenize = text.Tokenizer(num_words=vocab_size)
    tokenize.fit_on_texts(train_posts)

    x_train = tokenize.texts_to_matrix(train_posts, mode="tfidf")
    x_test = tokenize.texts_to_matrix(test_posts, mode="tfidf")

    encoder = LabelBinarizer()
    encoder.fit(train_tags)
    y_train = encoder.transform(train_tags)
    y_test = encoder.transform(test_tags)


    # saving the tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenize, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # build the model

    model = Sequential()

    model.add(Dense(512, input_shape=(vocab_size,)))
    model.add(Activation('relu'))

    num_labels = 37
    model.add(Dropout(0.5))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))


    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])


    history = model.fit(x_train, y_train,
                        batch_size=250,
                        epochs=1,
                        verbose=1,
                        validation_split=0.1)

    score = model.evaluate(x_test, y_test,
                           batch_size=50, verbose=1)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])


    for i in range(10):
        prediction = model.predict(np.array([x_test[i]]))

        text_labels = encoder.classes_
        predicted_label = text_labels[np.argmax(prediction[0])]
        print(test_posts.iloc[i][:50], "...")
        print('Actual label:', test_tags.iloc[i])
        print("Predicted label: ", predicted_label)


    # save model
    model.save("savedBOWmodel.h5")      # creates a HDF5 file 'my_model.h5'


def test_model():

    model = pickle.load(open("savedBOWmodel", "rb"))

    job_post_data = pd.read_csv("reed_uk.csv")

    # divide between training and test data
    max_num_words_in_list = 375172 * 1 / 4

    posts = job_post_data["job_description"]
    tags = job_post_data["category"]

    train_size = int(len(posts) * .8)

    train_posts = posts[:train_size]
    train_tags = tags[:train_size]

    test_posts = posts[train_size:]
    test_tags = tags[train_size:]

    vocab_size = int(max_num_words_in_list)

    tokenize = text.Tokenizer(num_words=vocab_size)
    tokenize.fit_on_texts(train_posts)

    x_train = tokenize.texts_to_matrix(train_posts, mode="tfidf")
    x_test = tokenize.texts_to_matrix(test_posts, mode="tfidf")

    encoder = LabelBinarizer()
    encoder.fit(train_tags)
    y_train = encoder.transform(train_tags)
    y_test = encoder.transform(test_tags)

    for i in range(10):
        prediction = model.predict(np.array([x_test[i]]))

        text_labels = encoder.classes_
        predicted_label = text_labels[np.argmax(prediction[0])]
        # print(test_posts.iloc[i][:50], "...")
        # print('Actual label:', test_tags.iloc[i])
        # print("Predicted label: ", predicted_label)

        yield test_posts.iloc[i][:50], "..."
        yield 'Actual label:', test_tags.iloc[i]
        yield "Predicted label: ", predicted_label

