# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 22:02:23 2022

@author: jtman
"""

import spacy
from spacy.matcher import Matcher


#extracts name, birth, death, nationality, and short bio for each match
def getinfo(matches):
    temp_info = []
    #looping through each match
    for tok_id, start, end in matches:
        words = [word for word in tokenized[start].subtree]
        names = [word for word in words if word.is_upper == True]
        dates = [word for word in words if word.is_alpha == False and str(word)[-1].isalpha != True]
        dates = [str(date) for date in dates if str(date) not in bad_dates]
        words = [word for word in words if word not in names and str(word) not in dates and str(word) not in bad_dates]
        nationality = [word for word in words if word.ent_type_ == "NORP"]
        try: 
            first_name = ' '.join([str(name) for name in names][1:])
        except:
            first_name = "unknown"
        try:
            last_name = str(names[0])
        except:
            last_name = "unknown"
        try: 
            born = str(dates[0])
        except:
            born = "unknown"
        try:
            died = str(dates[1])
        except:
            died = "unknown"
        try:
            description = ' '.join([str(word) for word in words])
        except:
            description = "unknown"
        try:
            nationality = ', '.join([str(word) for word in nationality])
        except:
            nationality = "unknown"
        person_info = {'first_name':first_name , "last_name" : last_name, "born": born, "died" : died, "nationality" : nationality, "BE" : description}
        temp_info.append(person_info)
        return temp_info
    
    
#read in file
file = open("D:/encyclopedia_a1.txt")
enc_text = file.read()
enc_text = enc_text.split()

#load our spacy model
nlp =  spacy.load('en_core_web_sm')

#creating matcher object
matcher = Matcher(nlp.vocab)
pattern = [{"IS_UPPER" : True, "DEP" : {"IN" :["nsubjpass", "ROOT"]}},{"ORTH": ","},{"IS_UPPER" : True, "OP" : "+"}] #,{},{},{},{},{},{}, {}, {},{},{}]#, {"OP" : "+"}]#{"LEMMA" : "be"}]
matcher.add("is_something", [pattern], greedy = "LONGEST")


#creating batches, since large spacy objects cause memory issues
batch_size = 100000
tot_batches = len(enc_text) / batch_size
batches = []
for x in range(int(tot_batches)):
    if x == tot_batches:
        batches.append(enc_text[x*batch_size:])
    else:
        batches.append(enc_text[x*batch_size:(x+1)* batch_size])

#items not to include as dates
bad_dates = ["", "-", ",", "(", ")"]

#loop through each batch, finding matches and then extracting the info
info = []
for i, batch in enumerate(batches):    
    print ("Batch: " + str(i))
    tokenized = nlp(' '.join(batch))
    matches = matcher(tokenized)
    temp_info = getinfo(matches)
    info += temp_info

#put extracted info in dataframe
import pandas as pd
df = pd.DataFrame(info)
print (df)

df.to_pickle("info_extraction.pkl")
