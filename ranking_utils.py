from pandas import DataFrame
from rank_bm25 import BM25Okapi
import pandas as pd
import numpy as np
from timeit import default_timer as timer
import glob

QUICK = True
# corpus read

# helper functions
def convert(list):
    return tuple(list)

def tokenize(collection, column_name, conver_tuples=False):
    tokenized_data = [doc.split(" ") for doc in collection[column_name]]  # expensive!!
    if conver_tuples:
        for i_token in range(len(tokenized_data)): # relevant for query for BM25Okapi, it requires tuples
            tokenized_data[i_token] = convert(tokenized_data[i_token])
    return tokenized_data


def init_minicorpi(path, n=100):
    """
    :param path: the path where the corpus text is located
    :param n: size of the corpus, not that is QUIK=True, this param is ommitted
    :return: none, but be aware of output: len(tokenized_querry) csv files of n rows
    """
    QUIK = False  # DEBUG OPTION
    # 1) read input
    corpus_col = pd.read_csv(path + '/output.csv', skiprows=0)
    if QUIK:
        corpus_col = corpus_col[0:10000]
        n = 10
    df_q = pd.read_csv(path + '/title_querries.csv', skiprows=0)
    # 2) tokenize
    tokenized_corpus = tokenize(corpus_col, column_name='corpus_text')
    tokenized_query = tokenize(df_q, column_name='query_text', conver_tuples=True)
    bm25 = BM25Okapi(tokenized_corpus)
    # 3) process to top-n
    i = 0
    for q in tokenized_query:
        top_n = bm25.get_top_n(q, corpus_col['corpus_text'], n)
        write_mini_corpi(i, top_n)
        print('rating and writing mini corpus of size {}: {}%'.format(n, i / len(tokenized_query) * 100))
        i += 1
    return


def appenddocandscore(doc, score):
    """append a document with score to the container
        container   = panda with two rows, on for doc-id and for score
        doc         = the id of the document
        score       = the BM25 Okapi score
    """
    return


def write_mini_corpi(queryid, topn):
    """
        write a minicorpus, it creates a csv in dir /topn_corpus with a file name that is the file id
        the file has the same structure a the main corpus, with headers: article_id,corpus_text. Not all docs are in it,
        but contains only the top-n docs
    """
    carrier: DataFrame = pd.DataFrame({'article_id': [], 'corpus_text': []})
    for doc in topn:
        doc = doc[1:len(doc)]  # remove leading zero
        docid = doc.partition("_")[0]
        if docid[-1] == '':
            # occasionally some ids are shorter and have a "" at the end
            docid = docid[0:-1]
        # append id and text to DataFrame
        df = pd.DataFrame({'article_id': [docid], 'corpus_text': [doc]})
        carrier = carrier.append(df)
    # write carrier to file with name docid
    carrier.to_csv('topn_corpus/{}.csv'.format(queryid))
    return


def getscores(tokenized_queries, method='BM25Okapi'):
    """
    getscores evaluates the method string, than checks the queries. Given that queries are the one evaluated and have a
    mini corpus in the 'topn_corpus/-dir, where i=5 relates to the fifth file that in turn contains the top-n for
    query 5.
    :param method: string with method, default = 'BM25Okapi'
    :param queries: list of the corpus query
    :return: -
    """
    # get all the files in dir 'topn_corpus/***.csv'.
    # for every file/query combination:
    # get the top_n with n=20?
    # write the top in a dir
    minicorpus_files = glob.glob('topn_corpus/*.csv')
    if method == 'BM25Okapi':
        i = 0
        for query in tokenized_queries:
            # read corpus top1000
            i_corpus = minicorpus_files[i]
            df_corpus = pd.read_csv(i_corpus, index_col=0)
            tokenized_corpus = tokenize(i_corpus, column_name='corpus_text')
            bm25 = BM25Okapi(tokenized_corpus)
            top_n = bm25.get_top_n(query, i_corpus['corpus_text'], n=20)
            scores = bm25.get_scores(query)
            result_df = pd.DataFrame({docid, score})
            i += 1
    return

# run ranking

# tokenized_corpus,
# bm25 = BM25Okapi(tokenized_corpus)
# i = 0
# s = (len(tokenized_query), len(tokenized_corpus))
# n = 3
# sn = (len(tokenized_query), n)
# doc_scores = np.zeros(s)
# top_n_short_agg = []
# # create minicorpi
# for q in tokenized_query:
#     # sort doc_scores
#     top_n = bm25.get_top_n(q, corpus_col['corpus_text'], n)
#     write_mini_corpi(i, top_n)
#
#     appenddocandscore(cont, doc, score)
#     top_n_short = []
#     for top_corpus in top_n:  # shorten strings of docs corpus text. TODO: make use of doc id
#         top_n_short.append(top_corpus[0:75])
#     top_n_short_agg.append(top_n_short)
#     print("Query {}   /{} processed".format(i, len(tokenized_query)))
#     i += 1
# end = timer()
# time_to_process = end - start
# # set the baseline 10 with BM25 Okapi
# for q in tokenized_query:
#     doc_scores[i] = bm25.get_scores(q)  # relevancy per document given the i-th query, for every document
#     # get top n of the reduced corpi
#
# print("running took: {} s".format(time_to_process))  # time in seconds
