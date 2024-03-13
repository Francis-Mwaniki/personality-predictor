
# This programme will classify people into mbti personality types based on their past 50 posts on social media using the basic naivebayesclassifier

# In[1]:


import pandas as pd
from streamlit import st
import numpy as np
import matplotlib.pyplot as plt
import nltk
import string
from nltk.classify import NaiveBayesClassifier


# ### Importing the dataset

# In[2]:


data_set = pd.read_csv("C:/Users/ADMIN/Desktop/project/mbti_1.csv")
data_set.tail()


# ### Checking the dataset for missing values

# In[3]:


data_set.isnull().any()


# ## Exploring the dataset

# The size of the dataset

# In[4]:


data_set.shape


# Explroing the posts in posts field

# In[5]:


data_set.iloc[0,1].split('|||')


# Finding the number of posts

# In[6]:


len(data_set.iloc[1,1].split('|||'))


# Finding the unique vales from type of personality column

# In[7]:


types = np.unique(np.array(data_set['type']))
types


# The total number of posts for each type

# In[8]:


total = data_set.groupby(['type']).count()*50
total


# In[10]:


all_posts= pd.DataFrame()
for j in types:
    temp1 = data_set[data_set['type']==j]['posts']
    temp2 = []
    for i in temp1:
        temp2+=i.split('|||')
    temp3 = pd.Series(temp2)
    all_posts[j] = temp3


# In[11]:


all_posts.tail()


# ### Creating a function to tokenize the words

# In[12]:


useless_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)
def build_bag_of_words_features_filtered(words):
    words = nltk.word_tokenize(words)
    return {
        word:1 for word in words \
        if not word in useless_words}


# A random check of the function

# In[13]:


build_bag_of_words_features_filtered(all_posts['INTJ'].iloc[1])


# ## Creating an array of features

# In[14]:


features=[]
for j in types:
    temp1 = all_posts[j]
    temp1 = temp1.dropna() #not all the personality types have same number of files
    features += [[(build_bag_of_words_features_filtered(i), j)     for i in temp1]]


# Because each number of personality types have different number of posts they must be splitted accordingle. Taking 80% for training and 20% for testing

# In[15]:


split=[]
for i in range(16):
    split += [len(features[i]) * 0.8]
split = np.array(split,dtype = int)


# In[16]:


split


# Data for training

# In[17]:


train=[]
for i in range(16):
    train += features[i][:split[i]] 


# Training the model

# In[18]:


sentiment_classifier = NaiveBayesClassifier.train(train)


# Testing the model on the dataset it was trained for accuracy

# In[19]:


nltk.classify.util.accuracy(sentiment_classifier, train)*100


# Creating the test data

# In[20]:


test=[]
for i in range(16):
    test += features[i][split[i]:]


# Testing the model on the test dataset which it has never seen before

# In[21]:


nltk.classify.util.accuracy(sentiment_classifier, test)*100

# In[22]:


# Features for the bag of words model
features=[]
for j in types:
    temp1 = all_posts[j]
    temp1 = temp1.dropna() #not all the personality types have same number of files
    if('I' in j):
        features += [[(build_bag_of_words_features_filtered(i), 'introvert')         for i in temp1]]
    if('E' in j):
        features += [[(build_bag_of_words_features_filtered(i), 'extrovert')         for i in temp1]]


# Data for training

# In[23]:


train=[]
for i in range(16):
    train += features[i][:split[i]] 


# Training the model

# In[24]:


IntroExtro = NaiveBayesClassifier.train(train)


# Testing the model on the dataset it was trained for accuracy

# In[25]:


nltk.classify.util.accuracy(IntroExtro, train)*100


# Creating the test data

# In[26]:


test=[]
for i in range(16):
    test += features[i][split[i]:]


# Testing the model on the test dataset which it has never seen before

# In[27]:


nltk.classify.util.accuracy(IntroExtro, test)*100


# Seeing that this model has good somewhat good results, I shall repeat the same with the rest of the traits

# ## Creating a classifyer for Intuition (N) and Sensing (S)

# In[28]:


# Features for the bag of words model
features=[]
for j in types:
    temp1 = all_posts[j]
    temp1 = temp1.dropna() #not all the personality types have same number of files
    if('N' in j):
        features += [[(build_bag_of_words_features_filtered(i), 'Intuition')         for i in temp1]]
    if('E' in j):
        features += [[(build_bag_of_words_features_filtered(i), 'Sensing')         for i in temp1]]


# Data for training

# In[29]:


train=[]
for i in range(16):
    train += features[i][:split[i]] 


# Training the model

# In[30]:


IntuitionSensing = NaiveBayesClassifier.train(train)


# Testing the model on the dataset it was trained for accuracy

# In[31]:


nltk.classify.util.accuracy(IntuitionSensing, train)*100


# Creating the test data

# In[32]:


test=[]
for i in range(16):
    test += features[i][split[i]:]


# Testing the model on the test dataset which it has never seen before

# In[33]:


nltk.classify.util.accuracy(IntuitionSensing, test)*100


# ## Creating a classifyer for Thinking (T) and Feeling (F)

# In[34]:


# Features for the bag of words model
features=[]
for j in types:
    temp1 = all_posts[j]
    temp1 = temp1.dropna() #not all the personality types have same number of files
    if('T' in j):
        features += [[(build_bag_of_words_features_filtered(i), 'Thinking')         for i in temp1]]
    if('F' in j):
        features += [[(build_bag_of_words_features_filtered(i), 'Feeling')         for i in temp1]]


# Data for training

# In[35]:


train=[]
for i in range(16):
    train += features[i][:split[i]] 


# Training the model

# In[36]:


ThinkingFeeling = NaiveBayesClassifier.train(train)


# Testing the model on the dataset it was trained for accuracy

# In[37]:


nltk.classify.util.accuracy(ThinkingFeeling, train)*100


# Creating the test data

# In[38]:


test=[]
for i in range(16):
    test += features[i][split[i]:]


# Testing the model on the test dataset which it has never seen before

# In[39]:


nltk.classify.util.accuracy(ThinkingFeeling, test)*100


# ## Creating a classifyer for Judging (J) and Percieving (P)

# In[40]:


# Features for the bag of words model
features=[]
for j in types:
    temp1 = all_posts[j]
    temp1 = temp1.dropna() #not all the personality types have same number of files
    if('J' in j):
        features += [[(build_bag_of_words_features_filtered(i), 'Judging')         for i in temp1]]
    if('P' in j):
        features += [[(build_bag_of_words_features_filtered(i), 'Percieving')         for i in temp1]]


# Data for training

# In[41]:


train=[]
for i in range(16):
    train += features[i][:split[i]] 


# Training the model

# In[42]:


JudgingPercieiving = NaiveBayesClassifier.train(train)


# Testing the model on the dataset it was trained for accuracy

# In[43]:


nltk.classify.util.accuracy(JudgingPercieiving, train)*100


# Creating the test data

# In[44]:


test=[]
for i in range(16):
    test += features[i][split[i]:]


# Testing the model on the test dataset which it has never seen before

# In[45]:


nltk.classify.util.accuracy(JudgingPercieiving, test)*100

# In[192]:


def MBTI(input):
    tokenize = build_bag_of_words_features_filtered(input)
    ie = IntroExtro.classify(tokenize)
    Is = IntuitionSensing.classify(tokenize)
    tf = ThinkingFeeling.classify(tokenize)
    jp = JudgingPercieiving.classify(tokenize)
    
    mbt = ''
    
    if(ie == 'introvert'):
        mbt+='I'
    if(ie == 'extrovert'):
        mbt+='E'
    if(Is == 'Intuition'):
        mbt+='N'
    if(Is == 'Sensing'):
        mbt+='S'
    if(tf == 'Thinking'):
        mbt+='T'
    if(tf == 'Feeling'):
        mbt+='F'
    if(jp == 'Judging'):
        mbt+='J'
    if(jp == 'Percieving'):
        mbt+='P'
    return(mbt)
    


# ### Building another functions that takes all of my posts as input and outputs the graph showing percentage of each trait seen in each posts and sums up displaying your personality as the graph title
# 
# **Note:** The input should be an array of your posts

# In[243]:


def tellmemyMBTI(input, name, traasits=[]):
    a = []
    trait1 = pd.DataFrame([0,0,0,0],['I','N','T','J'],['count'])
    trait2 = pd.DataFrame([0,0,0,0],['E','S','F','P'],['count'])
    for i in input:
        a += [MBTI(i)]
    for i in a:
        for j in ['I','N','T','J']:
            if(j in i):
                trait1.loc[j]+=1                
        for j in ['E','S','F','P']:
            if(j in i):
                trait2.loc[j]+=1 
    trait1 = trait1.T
    trait1 = trait1*100/len(input)
    trait2 = trait2.T
    trait2 = trait2*100/len(input)
    
    
    #Finding the personality
    YourTrait = ''
    for i,j in zip(trait1,trait2):
        temp = max(trait1[i][0],trait2[j][0])
        if(trait1[i][0]==temp):
            YourTrait += i  
        if(trait2[j][0]==temp):
            YourTrait += j
    traasits +=[YourTrait]
    desc=''.join(YourTrait)
    if desc == 'ISTJ':
        result= "ISTJs are reliable and responsible, known for their practical and organized approach to tasks."
    elif desc == 'INTJ':
        result= "INTJs are strategic and analytical thinkers, often seen as the 'architects' of ideas."
    elif desc == 'INTP':
        result= "INTPs are curious and inventive thinkers, known for their logical and abstract reasoning."
    elif desc == 'ENTJ':
        result= "ENTJs are natural leaders, decisive and assertive, often taking charge of situations."
    elif desc == 'ENTP':
        result= "ENTPs are innovative and adaptable, enjoying exploring new possibilities and ideas."
    elif desc == 'INFJ':
        result= "INFJs are empathetic and insightful, often referred to as the 'advocates' for their idealistic nature."
    elif desc == 'INFP':
        result= "INFPs are creative and compassionate, driven by their values and a desire for authenticity."
    elif desc == 'ENFJ':
        result= "ENFJs are charismatic and caring leaders, dedicated to bringing out the best in others."
    elif desc == 'ENFP':
        result= "ENFPs are enthusiastic and imaginative, always seeking new connections and experiences."
    elif desc == 'ISFJ':
        result= "ISFJs are supportive and nurturing, often putting others' needs before their own."
    elif desc == 'ESTJ':
        result= "ESTJs are efficient and decisive leaders, valuing order and structure in their environments."
    elif desc == 'ESFJ':
        result= "ESFJs are sociable and caring, often taking on the role of 'nurturers' in social groups."
    elif desc == 'ISTP':
        result= "ISTPs are pragmatic and adventurous, excelling in hands-on problem-solving and exploration."
    elif desc == 'ISFP':
        result= "ISFPs are artistic and sensitive, driven by a desire for self-expression and authenticity."
    elif desc == 'ESTP':
        result= "ESTPs are energetic and action-oriented, thriving in dynamic and fast-paced environments."
    elif desc == 'ESFP':
        result= "ESFPs are spontaneous and sociable, enjoying the present moment and engaging with others."
    else:
        result= "Invalid MBTI type code" 
    print('The Personality Trait is: '+result)
    return(result)