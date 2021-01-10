import numpy as np
import pandas as pd
import os
import cv2
import re
import string
import emoji
import nltk
from nltk.corpus import stopwords
#print(stopwords.words('english'))
all_lang=stopwords.fileids()
en_df = pd.read_csv("train.csv")

en_list = en_df.values.tolist() 

en_size = len(en_list)

def Punctuation(string): 
  
	# punctuation marks 
	punctuations = '''!()-[]{};:'"\\,<>./?@#$%^&*_~【】â€‹ÂðŸ˜”¯'''
  
	# traverse the given string and if any punctuation 
	# marks occur replace it with null 
	for x in string.lower(): 
		if x in punctuations: 
			string = string.replace(x, "")
	return string


def Stopwords(string):
    string = string.split()
    stop=stopwords.words("turkish") #23
    string = [w for w in string if not w in stop]
    string = str(string)
    return string

#Too slow and long
'''
def Stopwords(string):
    string = string.split()
    all_stop=[]

    for lang in all_lang:
        stop =stopwords.words(lang)
        #all_stop.append(stopwords.words(lang))
        string = [w for w in string if not w in stop]
        string = str(string)
    return string
'''        
    
#stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
	

for i in range(0, en_size):
	en_list[i][1] = str(en_list[i][1])
	en_list[i][1] = re.sub("\d", "", en_list[i][1])
	en_list[i][1] = re.sub(r'[^\x00-\x7f]', "", en_list[i][1])
	en_list[i][1] = Stopwords(en_list[i][1]) 
	en_list[i][1] = Punctuation(en_list[i][1])
	en_list[i][1] = en_list[i][1].lower()

clean_train_df = pd.DataFrame(en_list, columns=['review_id', 'review', 'rating'])
clean_train_df.to_csv('clean_train.csv', index = False)