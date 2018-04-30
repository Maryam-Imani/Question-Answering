"""
Created on Mon Apr 23 17:54:12 2018

@author: Xi
"""

import numpy as np
from scipy.spatial import distance
import nltk
import re

# function to calculate euclidean distance
def calc_distance(corpus_vector, query_vector):
    print('\n*********** start calculate similarity distance *************')
    dist = []
    
    for row in corpus_vector:
        #print('shape of row', row.shape)
        #dist.append(distance.euclidean(row, query_vector))  
        dist.append(distance.cosine(row, query_vector))
    #print('size of distance list:', len(dist))
    #print(dist)
    return dist

# function to index corpus sentences based on similarity
def matching(lines, dist):
    print('\n*********** start matching results in corpus *************')
    sortdist = sorted(dist)   # ranked from min to max
    original_index = np.argsort(dist)   # original index of the sorted distance
    #print('original index, ', original_index)

    results = []
    #best = lines[dist.index(max(dist))]   # max similarity
    #print(sortdist[0:10])
    for x in original_index[0:10]:
        #print('index', x, lines[x],'\n-------------------------')
        results.append(lines[x])
    print('the best 10 matching results: \n\n-------------------------')
    print('\n-------------------------'.join(map(str, results)))
    return(results)

# function to generate rank list for input queries
def generateRanklist(inputs, result_list):
    print('\n*********** start generate rank list *************')
    #print(type(result_list), len(result_list))
    rl = np.zeros(len(inputs))
    qlist = []  # store questions only 
    for i in range(0, len(inputs)):
        cur_input = inputs[i]
        print('--------------current input is: ', cur_input, '---------------')
        cur_results = result_list[i]    # list of 10 QApairs to current input 
        #print('**********resuls_list[i]********\n', result_list[i])
        for j in range(0, len(result_list[i])):
            matchedq = cur_results[j].split(':')[0]
            #print ('***********cur_results[j][0]******',matchedq)
            #qlist.append(matchedq)
            #print('**********qlist********\n', qlist)

            if cur_input.replace("\n","") != matchedq:
                rl[i] = 0
            else:
                print('rank is,',j + 1)
                rl[i] = j + 1
                break
        
    print('rank list:', rl)
    return rl
    
# function to calculate MRR for evaluation
# input: list of rank number
# output: MRR point
def evaluation(ranklist):
    print('\n*********** start compute MRR evaluation result *************')
    mask = ranklist != 0
    ranklist[mask] = np.reciprocal(ranklist[mask])
    #print(ranklist)
    result = np.mean(ranklist)
    return result

if __name__ == "__main__":
    l = [1,2,1,2,3]
    print(evaluation(l))