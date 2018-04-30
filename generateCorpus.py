"""
Created on Mon Apr 23 17:54:12 2018

@author: Xi

This file handles the raw dataset from QAs of Amazon
generates a corpus.txt
each line is a QA pair, Q and A divided by a colon
"""

import json as js
import ast

txtfile = open('qa.txt','r')	# QA of appliances from Amazon
corpus = open('50corpus.txt','w')	# corpus to store pairs of QAs
onlyQ = open('50onlyQ.txt', 'w')
cnt = 0
stop = 0
for line in txtfile:
	#js_line = js.dumps(line)
	line_dict = ast.literal_eval(line)
	assert type(line_dict) is dict
	question = line_dict['question']
	answer = line_dict['answer']	
	cnt += 1
	print('cnt',cnt,'stop',stop)
	if (cnt % 20 == 0):
		corpus.write(question + ':' + answer + '\n')
		onlyQ.write(question + '\n')
		stop += 1
	if (stop == 50):
		break;

corpus.close()
txtfile.close()
onlyQ.close()
print('there are ' + str(cnt) + ' pairs of Q&A')
print(str(stop), 'pairs have been written into corpus')