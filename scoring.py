import json
import gzip
import re
import time
import copy
import os
import ast
import collections
import random

from re import S
import ast
import nlgeval
from nlgeval import NLGEval

nlgeval = NLGEval()

# Testing
print("###############TEST###############")
refs = [['what is perceived as the general priority of the food processing industry in this sort of economy ?'], ["what was kublai khan 's relation to ogedei khan ?"], ['in which two countries is the uk installing military bases ?'], ['when does byu refuse to play athletic games that got the attention of the sports networks ?', 'what violation can lead to a player being expelled from a sports team ?'], ['what do australian citizens need in order to travel to norfolk island ?'], ['carthage and tunisia are in what general area ?'], ['who proved that air is necessary for combustion ?', 'what researcher showed that air is a necessity for combustion ?', 'in what century did mayow and boyle perform their experiments ?'], ['why were laserdiscs more popular in japan ?']]
hyps = ['in a profit-driven economy, health considerations are hardly a priority.', 'who offered kublai a position in xingzhou?', 'where is the uk establishing air and naval bases?', 'why has byu been criticized for refusing to play games on sunday?', 'what must australian citizens carry to travel to the island?', 'where is the bardo museum located?', 'who proved that air is necessary for combustion in the late 17th century?', 'why was laserdisc more popular in japan than north america?']
print('len refs sample: ', len(refs))
print('len hyps sample: ', len(hyps))
metrics_dict_sample = nlgeval.compute_metrics(refs, hyps)

print(metrics_dict_sample)
      
print("###############DONE###############")


CRSQuAD_Q = open('CRSQuAD_Q_lists.txt', 'r')
CRSQuAD_Q = [ast.literal_eval(line.strip("\n")) for line in CRSQuAD_Q.readlines()]

CRSQuAD_SA = open('CRSQuAD_SA_lists.txt', 'r')
CRSQuAD_SA = [line.strip("\n") for line in CRSQuAD_SA.readlines()]

print("CRSQUAD_Q: ", CRSQuAD_Q[0:10])
print("CRSQUAD_SA: ", CRSQuAD_SA[0:10])

# reading the hypothesis file in format of "one sentence per line"
hypothesis = open('hypothesis_test.txt', 'r')
hypothesis = [line.strip("\n") for line in hypothesis.readlines()]

# reading the references file in format of "one list per line"
references = open('CRSQuAD_test_lists.txt', 'r')
references = [ast.literal_eval(line.strip("\n")) for line in references.readlines()]

# testing to see whether the refrences and hypothesis files are aligned or not
print("REFERENCES: ", references[0:10])
print("HYPOTHESIS: ", hypothesis[0:10])

# getting the performance scores
metrics_dict = nlgeval.compute_metrics(references, hypothesis)


print(metrics_dict)







