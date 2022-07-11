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

CRSQuAD_Q = open('CRSQuAD_Q.txt', 'r')
CRSQuAD_Q = [line.strip("\n") for line in CRSQuAD_Q.readlines()]

CRSQuAD_LA = open('CRSQuAD_LA.txt', 'r')
CRSQuAD_LA = [line.strip("\n") for line in CRSQuAD_LA.readlines()]

CRSQuAD_SA = open('CRSQuAD_SA.txt', 'r')
CRSQuAD_SA = [line.strip("\n") for line in CRSQuAD_SA.readlines()]

CRSQuAD_dict = collections.defaultdict(list)

for i in range(len(CRSQuAD_Q)):
	CRSQuAD_dict[CRSQuAD_SA[i]].append(CRSQuAD_Q[i])


print("tedad key tooye dictionary: ", len(CRSQuAD_dict))
print("tedad question-SA pairs: ", len(CRSQuAD_Q))

SAs, Qs = zip(*CRSQuAD_dict.items())

print("SAs[1]: ", SAs[1])
print("Qs[1]: ", Qs[1])

print("type(SAs): ", type(SAs))
print("type(Qs): ", type(Qs))
print("type(Qs[0]): ", type(Qs[0]))

c = list(zip(SAs, Qs))
random.shuffle(c)
SAs, Qs = zip(*c)


# train, dev, and test sets out of CRSQUAD dataset
X_train = SAs[:50000]
X_dev = SAs[50000:56000]
X_test = SAs[56000:]

y_train = Qs[:50000]
y_dev = Qs[50000:56000]
y_test = Qs[56000:]


# extracting the sentences and lists of references for predicted questions
# in format of "one sentence per line" and "one list of questions per line"
with open('CRSQuAD_Q_lists.txt', 'w', encoding="utf-8") as outfile:
    outfile.write("\n".join(str(q) for q in Qs))

with open('CRSQuAD_SA_lists.txt', 'w', encoding="utf-8") as outfile:
    outfile.write("\n".join(SAs))

with open('CRSQuAD_test_lists.txt', 'w', encoding="utf-8") as outfile:
    outfile.write("\n".join(str(q) for q in y_test))


# extracting the train, dev, and test sets out of CRSQUAD
with open('myData/para-dev.txt', 'w', encoding="utf-8") as outfile:
    outfile.write("\n".join(X_dev))

with open('myData/para-train.txt', 'w', encoding="utf-8") as outfile:
    outfile.write("\n".join(X_train))

with open('myData/para-test.txt', 'w', encoding="utf-8") as outfile:
    outfile.write("\n".join(X_test))

# extracting the first question from the list of questions related to one sentence
with open('myData/tgt-dev.txt', 'w', encoding="utf-8") as outfile:
    outfile.write("\n".join(q[0] for q in y_dev))

with open('myData/tgt-train.txt', 'w', encoding="utf-8") as outfile:
    outfile.write("\n".join(q[0] for q in y_train))

with open('myData/tgt-test.txt', 'w', encoding="utf-8") as outfile:
    outfile.write("\n".join(q[0] for q in y_test))
