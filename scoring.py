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







