# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 17:54:12 2018

@author: mary
@modified: Xi
"""
import string, re
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
import sys
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import FreqDist
from scipy.spatial import distance
import match_evaluation

# Bag-of-words creation for corpus
# input: any text inputs
# output: unigrams and their occurrence count
def make_matrix(corpus):
    print('\n*********** start compute feature vector*************')
    BagOfWord = vectorizer.fit_transform(corpus).toarray()  # 9011 sentences * 17034 tokens
    features = vectorizer.get_feature_names()
    Vocabs = vectorizer.vocabulary_                         # 17034 tokens
    #print(type(BagOfWord))
    #print(BagOfWord.shape)
    #print(len(features))
    np.set_printoptions(precision=10)
    #print(BagOfWord)
    return BagOfWord, features

# Bag-of-words creation for query
# output: generate the dictionary of occurrence
def make_matrix2(query):
    print('*********** start compute feature vector for query *************')
    # Approach 1:
    # tokenizer = RegexpTokenizer(r'\w+') # remove punctuation
    # tokens = tokenizer.tokenize(sentence)
    # remove stop words
    # filtered_words = [w for w in tokens if not w in stopwords.words('english')] 
    # filtered_query = " ".join(filtered_words)
    
    # Approach 2:
    # simply tokenize the query
    # sentences = sent_tokenize(query)
    # words_list = []
    # for sentence in sentences:
    #    words = word_tokenize(sentence)
    #    words_list.append(words)
    #    print(words)
    #word_dist = FreqDist()
    #for s in sentences:
    #        word_dist.update(s.split(' '))
    #return dict(word_dist)
    l = []
    l.append(query)
    print(l)
    query_vector = vectorizer.fit_transform(l).toarray()
    query_features = vectorizer.get_feature_names()
    query_vocab = vectorizer.vocabulary_
    return query_vector, query_features

# function to expand query_vector to full feature vector
def fillFeatures(query_vector, query_features, BagOfWord, features):
    print('*********** start expand query feature vector according to corpus *************')
    expanded = np.zeros(len(features))  # initialize 1-D array, same length as corpus features
    for x in query_features:
        expanded.itemset(features.index(x), query_vector.item(query_features.index(x)))
        print('corpus index,', features.index(x))
        print('query index,', query_features.index(x))
        print('query number,', query_vector.item(query_features.index(x)))
    print(expanded.nonzero())   # verify the insertion
    return expanded

if __name__ == "__main__":
    
    Path2File = sys.argv[1] # corpus txt file path
    
    with open(Path2File, 'r') as fileR:
        corpus = fileR.readlines()
        lines = corpus
      
    inputcorpus = open('inputq_copy.txt','r')
    inputQuestion = inputcorpus.readlines()
    lines.extend(inputQuestion) # corpus + inputs
    printable = set(string.printable)

    result1 = make_matrix(lines)
    #print('len of lines:',len(lines))
    #print('type of lines:', type(lines))
    (BagOfWord, features) = (result1[0], result1[1])
    # get feature vectors for both input queries and corpus
    corpus_vector = BagOfWord[0:50]
    inputs_vector = BagOfWord[50:]

    dist_list = []
    for i in inputs_vector:
        dist = match_evaluation.calc_distance(corpus_vector, i)
        dist_list.append(dist)
    result_list = []   
    for d in dist_list:
        result_list.append(match_evaluation.matching(lines[0:50], d))
    ranklist = match_evaluation.generateRanklist(inputQuestion, result_list)
    print('MRR = ', match_evaluation.evaluation(ranklist))


