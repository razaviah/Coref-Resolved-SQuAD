import json
import nltk
import gzip
import re
import time
import copy
import os
#nltk.download("punkt")

from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")

from re import S
import spacy
from spacy.tokens import Doc
from allennlp.common.util import get_spacy_model
from allennlp.common.util import JsonDict
import en_core_web_sm
from allennlp_models.coref.predictors.coref import CorefPredictor
nlp = en_core_web_sm.load()



SQuAD_SA_train = open('para-train.txt', 'r')
SQuAD_SA_train = [line for line in SQuAD_SA_train.readlines()]

SQuAD_SA_dev = open('para-dev.txt', 'r')
SQuAD_SA_dev = [line for line in SQuAD_SA_dev.readlines()]

SQuAD_SA_test = open('para-test.txt', 'r')
SQuAD_SA_test = [line for line in SQuAD_SA_test.readlines()]


SQuAD_Q_train = open('tgt-train.txt', 'r')
SQuAD_Q_train = [line for line in SQuAD_Q_train.readlines()]

SQuAD_Q_dev = open('tgt-dev.txt', 'r')
SQuAD_Q_dev = [line for line in SQuAD_Q_dev.readlines()]

SQuAD_Q_test = open('tgt-test.txt', 'r')
SQuAD_Q_test = [line for line in SQuAD_Q_test.readlines()]


SQuAD_SA = SQuAD_SA_train + SQuAD_SA_dev + SQuAD_SA_test
SQuAD_Q = SQuAD_Q_train + SQuAD_Q_dev + SQuAD_Q_test



print("len SQuAD_SA: ", len(SQuAD_SA))
print("len set(SQuAD_SA): ", len(set(SQuAD_SA)))