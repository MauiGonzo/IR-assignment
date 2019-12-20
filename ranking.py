from rank_bm25 import BM25Okapi
import pandas as pd
import numpy as np
from timeit import default_timer as timer


def convert(list):
    return tuple(list)
#make it quicker
QUICK = True
# corpus read
start = timer()
# df = pd.read_csv("/home/maurice/PycharmProjects/hello_ranking/output.csv", skiprows = 0)
corpus_col = pd.read_csv("/Users/mauriceverbrugge/Google Drive/COMPUTING SCIENCE/IR/IR_assignment/robust04/output.csv", skiprows = 0)

if QUICK:
    corpus_col = corpus_col[0:10000] # schrinking for quick testing purposes

tokenized_corpus = [doc.split(" ") for doc in corpus_col['corpus_text']] # expensive!!

# query read
df_q = pd.read_csv("/Users/mauriceverbrugge/Google Drive/COMPUTING SCIENCE/IR/IR_assignment/robust04/title_querries.csv", skiprows = 0)
tokenized_query = [doc.split(" ") for doc in df_q['query_text']]

for i in range(len(tokenized_query)): #dedicated bm25.get_scores uses dict, which is hash and can handle tuple but no list
    tokenized_query[i] = convert(tokenized_query[i])
    #remove "" from tuple

# run ranking
bm25 = BM25Okapi(tokenized_corpus)
i = 0
s = (len(tokenized_query), len(tokenized_corpus))
n = 3
sn = (len(tokenized_query), n)
doc_scores = np.zeros(s)
top_n_short_agg = []
for q in tokenized_query:
    doc_scores[i] = bm25.get_scores(q) # relevancy per document given the i-th query, for every document
    top_n = bm25.get_top_n(q, corpus_col['corpus_text'], n)
    top_n_short = []
    for top_corpus in top_n: # shorten strings of docs corpus text. TODO: make use of doc id
        top_n_short.append(top_corpus[0:75])
    top_n_short_agg.append(top_n_short)
    print("Query {}   /{} processed".format(i, len(tokenized_query)))
    i += 1
end = timer()
time_to_process = end - start
print("running took: {} s".format(time_to_process)) # time in seconds
