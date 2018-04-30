# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 17:54:12 2018

@author: mary
@modified: Xi
"""

import string, re
import sys
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.parse.stanford import StanfordDependencyParser
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
from scipy.spatial import distance
import sys
import match_evaluation
stopwords = set(stopwords.words('english'))


# Lemmatize
wordnet_lemmatizer = WordNetLemmatizer()

def Lemmatize(sentence_tag):
    sentence_lemma = []
    for i in range(0, len(sentence_tag)):
        lem_list = []
        for each in sentence_tag[i]:
            word = each[0]
            tag = each[1]
            word = word.replace('&', '')
            word = filter(lambda x: x in printable, word)
            if tag[0] == 'V':
                lem_list.append(wordnet_lemmatizer.lemmatize(word, pos='v'))
            else:
                try:
                    lem_list.append(wordnet_lemmatizer.lemmatize(word), pos = tag[0])
                except: 
                    pass
        sentence_lemma.append(' '.join(lem_list))

    #print sentence_lemma
    return sentence_lemma
    

# Stemming
porter_stemmer = PorterStemmer()

def Stem(sentence_list):
    stem_list = []
    ps = PorterStemmer()
    
    for sentence in sentence_list: 
        tokens = word_tokenize(sentence)
        stems = []
        for token in tokens: 
            stems.append(ps.stem(token))
        
        stem_list.append(' '.join(stems))
    return stem_list



# Add Part_of_speech tag
def POS_Tagging(sentence_list):
    sentences_tag = []
    sentences_tag2 = []
    tag = []
    for each_sentence in sentence_list:
        text = word_tokenize(each_sentence)
        sentence_tag_temp = nltk.pos_tag(text)
        
        tag = []
        for (w, t) in sentence_tag_temp:
            tag.append(w+'_'+t)
        
        sentences_tag.append(sentence_tag_temp)
        sentences_tag2.append(' '.join(tag))

    #print sentences_tag[0]
    return sentences_tag, sentences_tag2


# Word Net
def Extract_Synset(sentence_list):
    hypernym = []
    hyponym = []
    holonym = []
    meronym = []

    for i in range(0, len(sentence_list)):
        hypernym_tmp = []
        hyponym_tmp = []
        holonym_tmp = []
        meronym_tmp = []
        for word in sentence_list[i]:
            if len(wn.synsets(word)) > 0:
                word_sense = wn.synsets(word)[0]
                ws_1 = word_sense.hypernyms()
                for j in ws_1:
                    hypernym_tmp += j.lemma_names()

                ws_2 = word_sense.hyponyms()
                for j in ws_2:
                    hyponym_tmp += j.lemma_names()

                ws_3 = word_sense.part_holonyms()
                for j in ws_3:
                    holonym_tmp += j.lemma_names()

                ws_4 = word_sense.part_meronyms()
                for j in ws_4:
                    meronym_tmp += j.lemma_names()

        hypernym.append(' '.join(hypernym_tmp) )
        hyponym.append(' '.join(hyponym_tmp) )
        holonym.append(' '.join(holonym_tmp) )
        meronym.append(' '.join(meronym_tmp) )

    return hypernym, hyponym, holonym, meronym


# Dependency parsing
def dependencyParsing(sentence_list):
    
    parsing = []
    sentences_parse = []
    path_to_jar = './StanfordParser/stanford-parser.jar'
    path_to_models_jar = './StanfordParser/stanford-parser-3.9.1-models.jar'
    dependency_parser = StanfordDependencyParser(path_to_jar, path_to_models_jar)
    
    for sent in sentence_list:
        try: 
            result = dependency_parser.raw_parse(sent)
            dep = result.next()        
            for triple in dep.triples():
                #print triple[1],"(",triple[0][0],", ",triple[2][0],")"
                parsing.append(triple[0][0]+'_'+triple[2][0])
            
        except:
            pass
        sentences_parse.append(' '.join(parsing))
    return sentences_parse


def make_matrix(copus):
    
    BagOfWord = vectorizer.fit_transform(copus).toarray()
    Vocabs = vectorizer.vocabulary_
    
    return BagOfWord, Vocabs


# extract all features
def featureExtraction(lines): 
    print('\n*********** start compute feature vector*************')
    # Lowercase, then replace any non-letter, space, or digit character in the headlines.
    new_lines = [re.sub(r'[^\w\s\d]','',h.lower()) for h in lines]
    # Replace sequences of whitespace with a space character.
    new_lines = [re.sub("\s+", " ", h) for h in new_lines]
    BagOfWordFeature = make_matrix(new_lines)[0]    
    
    
    # Extracting the stems 
    stem_lists = Stem(lines)
    StemFeature = make_matrix(stem_lists)[0]
     
    
    # Extracting the PoS tags for each word 
    sentence_tag, sentence_tag2 = POS_Tagging(lines)
    PoSFeature = make_matrix(sentence_tag2)[0]
    
    
    # Extracting the Lemma 
    sentences_lemma = Lemmatize(sentence_tag)
    LemmaFeature = make_matrix(sentences_lemma)[0]
    
    
    # Extracting hypernym, hyponym, holonym, meronym from wordnt 
    hypernym, hyponym, holonym, meronym = Extract_Synset(lines) 
    hypernymFeature = make_matrix(hypernym)[0]
    hyponymFeature = make_matrix(hyponym)[0]
    holonymFeature = make_matrix(holonym)[0]
    meronymFeature = make_matrix(meronym)[0]
    
    # Extracting the parsing features
    parsing = dependencyParsing(lines)
    parsingFeature = make_matrix(parsing)[0]
    
    # Concating all the features 
    fea = [BagOfWordFeature, StemFeature, PoSFeature, LemmaFeature, hypernymFeature, hyponymFeature, holonymFeature, meronymFeature, parsingFeature]
    allFeatures = np.concatenate(fea, axis=1)
    #print allFeatures
    
    return allFeatures

if __name__ == "__main__":
    
    Path2File = '50corpus_copy.txt'
    extract_result = open('result.txt','w')
    with open(Path2File, 'r') as fileR:
        corpus = fileR.readlines()
        lines = corpus
    #print(type(lines), len(lines))
    inputcorpus = open('inputq_copy.txt','r')
    inputQuestion = inputcorpus.readlines()
    lines.extend(inputQuestion) # corpus + inputs  
    printable = set(string.printable)
    
    features = featureExtraction(lines) 
    #print('type of features', type(features), 'shape', features.shape)
    inputs = features[50:]
    dist_list = []
    for i in inputs:
        dist = match_evaluation.calc_distance(features[0:50], i)
        dist_list.append(dist)
    
    #print type(dist_list), len(dist_list)
    result_list = []   
    for d in dist_list:
        result_list.append(match_evaluation.matching(lines[0:50], d))

    ranklist = match_evaluation.generateRanklist(inputQuestion, result_list)
    print('MRR = ', match_evaluation.evaluation(ranklist))
