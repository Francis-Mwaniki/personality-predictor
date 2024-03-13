import streamlit as st
import pandas as pd
import numpy as np
import nltk
import string
from nltk.classify import NaiveBayesClassifier

import logging

logging.basicConfig(filename='app_log.log', level=logging.INFO)

# Load the dataset (assuming it's already available)
nltk.download('stopwords')
nltk.download('punkt')
# data_set = pd.read_csv("data2.csv")
# Load the dataset and remove duplicates
data_set = pd.read_csv("data2.csv").drop_duplicates(subset=['posts', 'type']).reset_index(drop=True)


# Function to tokenize the words
useless_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)

def build_bag_of_words_features_filtered(words):
    words = nltk.word_tokenize(words)
    return {word: 1 for word in words if not word in useless_words}

# Function to classify Extraversion/Introversion
def ExtraversionIntroversion(input):
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
        if 'I' in j or 'E' in j:
            features += [[(build_bag_of_words_features_filtered(i), 'introvert' if 'I' in j else 'extrovert') for i in temp1]]

    # Data for training
    split = []
    for i in range(len(features)):
        split += [len(features[i]) * 0.8]
    split = np.array(split, dtype=int)

    train = []
    for i in range(len(features)):
        train += features[i][:split[i]]

    # Training the models
    intro_extro = NaiveBayesClassifier.train(train)

    # Classify Extraversion/Introversion
    tokenize = build_bag_of_words_features_filtered(input)
    ie = intro_extro.classify(tokenize)
    ie_probs = intro_extro.prob_classify(tokenize)
    ie_confidence = ie_probs.prob(ie)

    return ie, ie_confidence

def describe_personality(personality_type):
    if personality_type == 'introvert':
        return "You are likely an introvert. Introverts often enjoy spending time alone, value deep connections with a few close friends, and may prefer quiet and reflective activities."
    elif personality_type == 'extrovert':
        return "You are likely an extrovert. Extroverts tend to be outgoing, social, and enjoy interacting with a wide range of people. They often thrive in dynamic and lively environments."
# Streamlit app
def main():
    st.title("Personality Predictor")

    st.sidebar.header("User Input")
    user_input = st.sidebar.text_area("Enter your posts (separated by |||):")

    if st.sidebar.button("Predict Personality"):
        if user_input:
            result, confidence = ExtraversionIntroversion(user_input)
            description = describe_personality(result)
            st.success(f"Your predicted Personality is: {result} with confidence: {confidence:.2%} accuracy.")
            st.info(description)
            logging.info(f"Input: {user_input}, Predicted Personality: {result}, Confidence: {confidence:.2%}")
            
        else:
            st.warning("Please enter your posts to predict Extraversion/Introversion.")
    st.sidebar.write(f"Its nice to know more than 60%(confidence level) is accurate, but we can do better. Less than 60%(confidence level) is not very accurate. or unsure.")    
    st.sidebar.write(f"Training data: {len(data_set)} rows, {len(data_set.columns)} columns")
    st.sidebar.write(f"Due to limited data, the model may not be very accurate. {len(data_set)} rows is not enough to train a model. We need more data to improve the model.")
if __name__ == "__main__":
    main()
