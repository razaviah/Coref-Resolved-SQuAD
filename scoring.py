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
from nlgeval import NLGEval 

CRSQuAD_Q = open('CRSQuAD_Q_lists.txt', 'r')
CRSQuAD_Q = [json.loads(line.strip("\n")) for line in CRSQuAD_Q.readlines()]

CRSQuAD_SA = open('CRSQuAD_SA_lists.txt', 'r')
CRSQuAD_SA = [line.strip("\n") for line in CRSQuAD_SA.readlines()]

CRSQuAD_test = open('CRSQuAD_test_lists.txt', 'r')
CRSQuAD_test = [json.loads(line.strip("\n")) for line in CRSQuAD_test.readlines()]

print("Questions: ", CRSQuAD_Q[0:10])
print("Test Questions: ", CRSQuAD_test[0:10])
print("Short Answers: ", CRSQuAD_SA[0:10])

hypothesis = open('hypothesis_test.txt', 'r')
hypothesis = [line.strip("\n") for line in hypothesis.readlines()]

print(hypothesis[0:10])

nlgeval = NLGEval()
references = CRSQuAD_test
metrics_dict = nlgeval.compute_metrics(references, hypothesis)


print(metrics_dict)







