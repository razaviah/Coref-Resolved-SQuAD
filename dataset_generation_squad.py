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


# original_Dataset_Counter = 0

# SNQD_Counter = 0
# SNQD_Q = []
# SNQD_SA = []
# SNQD_LA = []

# # 'wh' question words
# whQuestions = ['what', 'which', 'who', 'where', 'why', 'when', 'how', 'whose']

# SNQD_Index = list(list())

# # name of the file of original dataset
# mFileName = 'v1.0-simplified_simplified-nq-train.jsonl.gz'

# with gzip.open(mFileName , 'rb') as mother_file:
#     lines = mother_file.readlines()
#     for line in lines:
#         data = json.loads(line)
#         lastart_token = data["annotations"][0]["long_answer"]["start_token"]
#         laend_token = data["annotations"][0]["long_answer"]["end_token"]
#         numOfShortAnswers = len(data["annotations"][0]["short_answers"])


#         # modified questions and answers
#         if numOfShortAnswers != 0 and (lastart_token != -1) and (laend_token != -1):
#             sastart_token_list = []
#             saend_token_list = []
#             # adding all the start and end tokens of short answers to a list
#             for n in range(numOfShortAnswers):
#                 sastart_token = data["annotations"][0]["short_answers"][n]["start_token"]
#                 saend_token = data["annotations"][0]["short_answers"][n]["end_token"]
#                 sastart_token_list.append(sastart_token)
#                 saend_token_list.append(saend_token)
#             # using min and max of that list to get the actual sentence
#             minSaStartToken = min(sastart_token_list)
#             maxSaEndToken = max(saend_token_list)
#             # checking whether the short answers are in long answer or not
#             if (minSaStartToken >= lastart_token) and (maxSaEndToken <= laend_token):
#                 documentSplit = data["document_text"].split(" ")
#                 documentSplitLength = len(documentSplit)
#                 startingIndex = minSaStartToken
#                 endingIndex = maxSaEndToken
#                 # finding the starting and ending points of the short answer's sentence
#                 while True:
#                     startingIndex = startingIndex - 1
#                     if startingIndex == 0:
#                         startingIndex = -1
#                         break
#                     if (documentSplit[startingIndex] == ".") or (documentSplit[startingIndex] =="<P>") or (documentSplit[startingIndex] =="</P>"):
#                         break
                

#                 while True:
#                     if endingIndex == documentSplitLength:
#                         break
#                     if (documentSplit[endingIndex] == ".") or (documentSplit[endingIndex] =="<P>") or (documentSplit[endingIndex] =="</P>"):
#                         break
#                     endingIndex = endingIndex + 1

#                 # if the sentence of short answer has less than 50 words, we continue with the data
#                 if ((endingIndex - startingIndex) < 50):
#                     # check whether the current data's short answer does not contain more than 1 sentence and its long answer does not contain any list, tables, etc.
#                     # As if the long answer contains any of those types, the parsed sentence would not be good
#                     if ("</Td>" not in documentSplit[lastart_token:laend_token]) and ("<Li>" not in documentSplit[lastart_token:laend_token]) and documentSplit[startingIndex+1:endingIndex - 1].count(".") == 0:
#                         doesItHaveHTMLTags = False
#                         for word in documentSplit[startingIndex+1:endingIndex]:
#                             # Checking if any of the words in short answer is an HTML tag
#                             if bool(re.match("<[^<>]+>", word)):
#                                 doesItHaveHTMLTags = True
#                                 break
#                         # The resulting data should not have any HTML tag, and the first word in the corresponding question should start with a 'wh' question
#                         if doesItHaveHTMLTags == False and documentSplit[lastart_token]=="<P>" and data["question_text"].split(' ')[0] in whQuestions:
#                             SNQD_SA.append(" ".join(documentSplit[startingIndex+1:endingIndex]))                # parsed short answer
#                             SNQD_Q.append(data["question_text"])                                                # question corresponding to the short answer
#                             SNQD_LA.append(" ".join(documentSplit[lastart_token+1:laend_token-1]))              # long answer taken from the whole Wikipedia page
#                             SNQD_Index.append([startingIndex+1 - lastart_token, endingIndex - lastart_token])   # we store the indexes for coreference resolution usage later
#                             SNQD_Counter = SNQD_Counter + 1




#         original_Dataset_Counter  = original_Dataset_Counter + 1















SQuAD_LA_train = open('para-train.txt', 'r')
SQuAD_LA_train = [line for line in SQuAD_LA_train.readlines()]

SQuAD_LA_dev = open('para-dev.txt', 'r')
SQuAD_LA_dev = [line for line in SQuAD_LA_dev.readlines()]

SQuAD_LA_test = open('para-test.txt', 'r')
SQuAD_LA_test = [line for line in SQuAD_LA_test.readlines()]



SQuAD_SA_train = open('src-train.txt', 'r')
SQuAD_SA_train = [line for line in SQuAD_SA_train.readlines()]

SQuAD_SA_dev = open('src-dev.txt', 'r')
SQuAD_SA_dev = [line for line in SQuAD_SA_dev.readlines()]

SQuAD_SA_test = open('src-test.txt', 'r')
SQuAD_SA_test = [line for line in SQuAD_SA_test.readlines()]



SQuAD_Q_train = open('tgt-train.txt', 'r')
SQuAD_Q_train = [line for line in SQuAD_Q_train.readlines()]

SQuAD_Q_dev = open('tgt-dev.txt', 'r')
SQuAD_Q_dev = [line for line in SQuAD_Q_dev.readlines()]

SQuAD_Q_test = open('tgt-test.txt', 'r')
SQuAD_Q_test = [line for line in SQuAD_Q_test.readlines()]


SQuAD_LA = SQuAD_LA_train + SQuAD_LA_dev + SQuAD_LA_test
SQuAD_SA = SQuAD_SA_train + SQuAD_SA_dev + SQuAD_SA_test
SQuAD_Q = SQuAD_Q_train + SQuAD_Q_dev + SQuAD_Q_test





# coreference resolution
# Taken from AllenNLP's github with modification: https://github.com/allenai/allennlp-models/blob/main/allennlp_models/coref/predictors/coref.py
# Taken from Marta Maślankowska's Towards Data Science article with modification: https://towardsdatascience.com/how-to-make-an-effective-coreference-resolution-model-55875d2b5f19
def get_span_noun_indices(doc, cluster):
    spans = [doc[span[0]:span[1]+1] for span in cluster]
    spans_pos = [[token.pos_ for token in span] for span in spans]
    span_noun_indices = [i for i, span_pos in enumerate(spans_pos)
        if any(pos in span_pos for pos in ['NOUN', 'PROPN'])]
    return span_noun_indices


def get_cluster_head(doc, cluster, noun_indices):
    head_idx = noun_indices[0]
    head_start, head_end = cluster[head_idx]
    head_span = doc[head_start:head_end+1]
    return head_span, [head_start, head_end]


def is_containing_other_spans(span, all_spans):
    return any([s[0] >= span[0] and s[1] <= span[1] and s != span for s in all_spans])


def replace_corefs(document, clusters, shortAnswer_startIndex, shortAnswer_endIndex) -> str:
    resolved = list(tok.text_with_ws for tok in document)
    all_spans = [span for cluster in clusters for span in cluster]  # flattened list of all spans

    for cluster in clusters:
        noun_indices = get_span_noun_indices(document, cluster)
        if noun_indices:
            mention_span, mention = get_cluster_head(document, cluster, noun_indices)

            for coref in cluster:
                if coref != mention and not is_containing_other_spans(coref, all_spans):
                    # the rest of the logic happens here
                  # inja dg baghiash faghat jaygozin kardane
                  # yani barmidarim, shortAList[coref[0]-sIndex:coref[1]+1-sIndex] = longAList[mention[0]:mention[1]+1]
                  final_token = document[coref[1]]
                  if final_token.tag_ in ["PRP$", "POS"]:
                    resolved[coref[0]] = mention_span.text + "'s" + final_token.whitespace_
                  else:
                    # If not possessive, then replace first token with main mention directly
                    resolved[coref[0]] = mention_span.text + final_token.whitespace_
                  # Mask out remaining tokens
                  for i in range(coref[0] + 1, coref[1] + 1):
                    resolved[i] = ""
    return "".join(resolved[shortAnswer_startIndex:shortAnswer_endIndex])
    # Here we are trying to handle the low level indexing problem to some degree, but there still is going to be '.' included sentences
    # if '. ' in resolved[shortAnswer_startIndex-1:shortAnswer_endIndex-1]:
    #     if resolved[shortAnswer_startIndex-1] == '. ':
    #         return "".join(resolved[shortAnswer_startIndex:shortAnswer_endIndex])
    #     elif resolved[shortAnswer_startIndex] == '. ':
    #         return "".join(resolved[shortAnswer_startIndex+1:shortAnswer_endIndex+1])
    #     elif resolved[shortAnswer_startIndex+1] == '. ':
    #         return "".join(resolved[shortAnswer_startIndex+2:shortAnswer_endIndex+2])
    #     else:
    #         return "".join(resolved[shortAnswer_startIndex-1:shortAnswer_endIndex-1])
    # else:
    #     return "".join(resolved[shortAnswer_startIndex-1:shortAnswer_endIndex-1])


def coref_resolved(document: str, shortAnswer_startIndex, shortAnswer_endIndex) -> str:
    """
    Produce a document where each coreference is replaced by its main mention
    # Parameters
    document : `str`
        A string representation of a document.
    # Returns
    A string with each coreference replaced by its main mention

    """
    sIndex = shortAnswer_startIndex
    eIndex = shortAnswer_endIndex

    spacy_document = nlp(document)
    clusters = predictor.predict(document = document)["clusters"]
    # print('clusters: ', clusters)
    new_clusters = editing_clusters(spacy_document, clusters, shortAnswer_startIndex, shortAnswer_endIndex)
    


    # Passing a document with no coreferences returns its original form
    if not new_clusters:
      res = list(tok.text_with_ws for tok in spacy_document)
      return "".join(res[shortAnswer_startIndex:shortAnswer_endIndex])
    return replace_corefs(spacy_document, clusters, shortAnswer_startIndex, shortAnswer_endIndex)

# editing the clusters so that we only keep the clusters that are in the short answer span
def editing_clusters(document, clusters, shortAnswer_startIndex, shortAnswer_endIndex):
    sIndex = shortAnswer_startIndex
    eIndex = shortAnswer_endIndex
    j = 0
    while j < len(clusters):
      cluster = clusters[j]
      cluster_has_indicies_in_ShortA = False
      noun_indices = get_span_noun_indices(document, cluster)
      if noun_indices:
        mention_span, mention = get_cluster_head(document, cluster, noun_indices)
        i = 0
        while i < len(cluster):
          coref = cluster[i]
          if ((coref[0] in range(sIndex, eIndex+1)) or (coref[1] in range(sIndex, eIndex+1))) and (not mention==coref):  ############ sIndex-1
            cluster_has_indicies_in_ShortA = True
          elif (not mention==coref):
            cluster.remove(cluster[i])
            i = i - 1 # as we remove the coref which is not important to us for our problem, the indicies in cluster become 1 less
          i = i + 1
      if cluster_has_indicies_in_ShortA == False:
        clusters.remove(cluster)
        j = j - 1
      j = j + 1
    return clusters

# correcting the indexing issue
def contains(SA, LA):
    for i in range(len(LA)-len(SA)+1):
        for j in range(len(SA)):
            if LA[i+j] != SA[j]:
                break
        else:
            return i, i+len(SA)
    return False

SQuAD_Index = list(list())
counter = 0
for i in range(len(SQuAD_Q)):
    spacy_sa = list(tok.text_with_ws.strip() for tok in nlp(SQuAD_SA[i]))
    spacy_la = list(tok.text_with_ws.strip() for tok in nlp(SQuAD_LA[i]))
    spacy_sa = spacy_sa[:-1]
    spacy_la = spacy_la[:-1]

    res = contains(spacy_sa, spacy_la)
    if res==False:
        counter = counter + 1
        print("Are dg, 'res' False daroomad")
        print("SA: ", SQuAD_SA[i])
        print("spacy sa: ", spacy_sa)
        print("spacy la: ", spacy_la)
        print("i: ", i)
        print("Counter: ", counter)
    (start_index, end_index) = res
    SQuAD_Index.append([start_index, end_index])



print("TABRIIIIIIIK!!!!")

# resolving the coreference resolution in each of the sentences
# Coref_SNQD_SA is the list of short answers in SNQD, getting their coreferences solved
start_time = time.time()
new_start_time = time.time()
Coref_SQuAD_SA = []
for i in range(len(SQuAD_Q)):
  s = coref_resolved(SQuAD_LA[i], SQuAD_Index[i][0], SQuAD_Index[i][1])
  Coref_SQuAD_SA.append(s)
  if i%500==0 and i!=0:
    print("currently solving %dth short answer!" % i)
    end_time = time.time()
    print("time taken for this 500 Qs (in seconds): ", end_time - new_start_time)
    print("\n")
    new_start_time = time.time()
final_end_time = time.time()

print("total time taken (in minutes): ", (final_end_time - start_time)/60)





defectives = 0 # A counter for number of defectives in our dataset (the short answers that have '.' in them)
# deep copying the lists we have since we dont want to accidentally change the current dataset we have
CRSQuAD_SA = copy.deepcopy(Coref_SQuAD_SA)
CRSQuAD_LA = copy.deepcopy(SQuAD_LA)
CRSQuAD_Q = copy.deepcopy(SQuAD_Q)
CRSQuAD_Index = copy.deepcopy(SQuAD_Index)

# Getting the final Coref Resolved SNQD dataset
# CRSNQD and SNQD datasets are almost the same except the short answers in CRSNQD are coreference resolved of short answers in SNQD
#                          and the final CRSQND has less data in it due to the defectives which we eliminate in the following code
for i in range(len(Coref_SQuAD_SA)):
  saWordList = Coref_SQuAD_SA[i].split(' ')
  if ('.' in saWordList) \
  or (len(SQuAD_Q[i].split(' ')) > len(saWordList) \
  or len(SQuAD_Q[i]) > len(Coref_SQuAD_SA[i])) \
  or len(saWordList) < 8:
    CRSQuAD_SA.pop(i-defectives)
    CRSQuAD_LA.pop(i-defectives)
    CRSQuAD_Q.pop(i-defectives)
    CRSQuAD_Index.pop(i-defectives)
    defectives = defectives + 1

print("len before deletion (should be equal to SQuAD's length): ", len(SQuAD_Q))
print('len after deletion (Coref Resolved SQuAD length): ', len(CRSQuAD_Q))



# train, dev, and test sets out of CRSNQD dataset
X_train = CRSQuAD_SA[:70484]
X_dev = CRSQuAD_SA[70484:81054]
X_test = CRSQuAD_SA[81054:]

y_train = CRSQuAD_Q[:70484]
y_dev = CRSQuAD_Q[70484:81054]
y_test = CRSQuAD_Q[81054:]



# # extracting the whole SNQD
# with open('SNQD/SNQD_Q.txt', 'w', encoding="utf-8") as outfile:
#     outfile.write("\n".join(SNQD_Q))

# with open('SNQD/SNQD_SA.txt', 'w', encoding="utf-8") as outfile:
#     outfile.write("\n".join(SNQD_SA))

# with open('SNQD/SNQD_LA.txt', 'w', encoding="utf-8") as outfile:
#     outfile.write("\n".join(SNQD_LA))

# # with open('SNQD/SNQD_Index.txt', 'w', encoding="utf-8") as outfile:
# #     outfile.write("\n".join(str(e) for e in SNQD_Index))


# extracting the original dataset with coref resolved and no deletion of defectives
with open('CRSQuAD/CRSQuAD_Q_noDeletion.txt', 'w', encoding="utf-8") as outfile:
    outfile.write("\n".join(SQuAD_Q))

with open('CRSQuAD/CRSQuAD_SA_noDeletion.txt', 'w', encoding="utf-8") as outfile:
    outfile.write("\n".join(Coref_SQuAD_SA))

with open('CRSQuAD/CRSQuAD_LA_noDeletion.txt', 'w', encoding="utf-8") as outfile:
    outfile.write("\n".join(SQuAD_LA))


# extracting the whole CRSNQD
with open('CRSQuAD/CRSQuAD_Q.txt', 'w', encoding="utf-8") as outfile:
    outfile.write("\n".join(CRSQuAD_Q))

with open('CRSQuAD/CRSQuAD_SA.txt', 'w', encoding="utf-8") as outfile:
    outfile.write("\n".join(CRSQuAD_SA))

with open('CRSQuAD/CRSQuAD_LA.txt', 'w', encoding="utf-8") as outfile:
    outfile.write("\n".join(CRSQuAD_LA))

# with open('CRSNQD/CRSNQD_Index.txt', 'w', encoding="utf-8") as outfile:
#     outfile.write("\n".join(str(e) for e in CRSNQD_Index))



# extracting the train, dev, and test sets out of CRSNQD
with open('data/myData/para-dev.txt', 'w', encoding="utf-8") as outfile:
    outfile.write("\n".join(X_dev))

with open('data/myData/para-train.txt', 'w', encoding="utf-8") as outfile:
    outfile.write("\n".join(X_train))

with open('data/myData/para-test.txt', 'w', encoding="utf-8") as outfile:
    outfile.write("\n".join(X_test))

with open('data/myData/tgt-dev.txt', 'w', encoding="utf-8") as outfile:
    outfile.write("\n".join(y_dev))

with open('data/myData/tgt-train.txt', 'w', encoding="utf-8") as outfile:
    outfile.write("\n".join(y_train))

with open('data/myData/tgt-test.txt', 'w', encoding="utf-8") as outfile:
    outfile.write("\n".join(y_test))