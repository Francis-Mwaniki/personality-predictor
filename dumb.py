import streamlit as st
import pandas as pd
import numpy as np
import nltk
import string
from nltk.classify import NaiveBayesClassifier

# Load the dataset (assuming it's already available)
nltk.download('stopwords')
nltk.download('punkt')
data_set = pd.read_csv("data2.csv")

# Function to tokenize the words
useless_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)

def build_bag_of_words_features_filtered(words):
    words = nltk.word_tokenize(words)
    return {word: 1 for word in words if not word in useless_words}



# Function to classify MBTI type
def MBTI(input):
    # Extract posts from the dataset
    all_posts = pd.DataFrame()
    types = np.unique(np.array(data_set['type']))
    for j in types:
        temp1 = data_set[data_set['type'] == j]['posts']
        temp2 = []
        for i in temp1:
            temp2 += i.split('|||')
        temp3 = pd.Series(temp2)
        all_posts[j] = temp3

    # Features for the bag of words model
    features = []
    for j in types:
        temp1 = all_posts[j]
        temp1 = temp1.dropna()
        if 'I' in j:
            features += [[(build_bag_of_words_features_filtered(i), 'introvert') for i in temp1]]
        if 'E' in j:
            features += [[(build_bag_of_words_features_filtered(i), 'extrovert') for i in temp1]]
        if 'N' in j:
            features += [[(build_bag_of_words_features_filtered(i), 'Intuition') for i in temp1]]
        if 'S' in j:
            features += [[(build_bag_of_words_features_filtered(i), 'Sensing') for i in temp1]]
        if 'T' in j:
            features += [[(build_bag_of_words_features_filtered(i), 'Thinking') for i in temp1]]
        if 'F' in j:
            features += [[(build_bag_of_words_features_filtered(i), 'Feeling') for i in temp1]]
        if 'J' in j:
            features += [[(build_bag_of_words_features_filtered(i), 'Judging') for i in temp1]]
        if 'P' in j:
            features += [[(build_bag_of_words_features_filtered(i), 'Percieving') for i in temp1]]

    # Data for training
    split = []
    for i in range(16):
        split += [len(features[i]) * 0.8]
    split = np.array(split, dtype=int)

    train = []
    for i in range(16):
        train += features[i][:split[i]]

    # Training the models
    intro_extro = NaiveBayesClassifier.train(train)
    intuition_sensing = NaiveBayesClassifier.train(train)
    thinking_feeling = NaiveBayesClassifier.train(train)
    judging_perceiving = NaiveBayesClassifier.train(train)

    # Classify MBTI type
    tokenize = build_bag_of_words_features_filtered(input)
    ie = intro_extro.classify(tokenize)
    is_ = intuition_sensing.classify(tokenize)
    tf = thinking_feeling.classify(tokenize)
    jp = judging_perceiving.classify(tokenize)

    mbt = ''

    if ie == 'introvert':
        mbt += 'I'
    if ie == 'extrovert':
        mbt += 'E'
    if is_ == 'Intuition':
        mbt += 'N'
    if is_ == 'Sensing':
        mbt += 'S'
    if tf == 'Thinking':
        mbt += 'T'
    if tf == 'Feeling':
        mbt += 'F'
    if jp == 'Judging':
        mbt += 'J'
    if jp == 'Percieving':
        mbt += 'P'
    return mbt

# Streamlit app
def main():
    st.title("MBTI Personality Type Classifier")

    st.sidebar.header("User Input")
    user_input = st.sidebar.text_area("Enter your posts (separated by |||):")

    if st.sidebar.button("Predict MBTI"):
        if user_input:
            result = MBTI(user_input)
            st.success("Your predicted MBTI Personality Type is: " + result)
        else:
            st.warning("Please enter your posts to predict MBTI.")

if __name__ == "__main__":
    main()
