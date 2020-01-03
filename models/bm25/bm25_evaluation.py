# Special version of BM25 used for the evaluation of documents with a sentence removed

from rank_bm25 import BM25Okapi
import numpy as np

class bm25_evaluation(BM25Okapi):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        super().__init__(corpus, tokenizer, k1, b, epsilon)


    def get_new_scores(self, query, documents):

        _score = np.zeros(len(documents))
        _doc_len = np.array(list(map(len, documents)))
        _freq_info = [np.unique(np.array(doc), return_counts=True) for doc in documents]
        _doc_freqs = [dict(zip(doc[0], doc[1])) for doc in _freq_info]
        for q in query:
            _q_freq = np.array([(doc.get(q) or 0) for doc in _doc_freqs])
            _score += (self.idf.get(q) or 0) * (_q_freq * (self.k1 + 1) /
                                               (_q_freq + self.k1 * (1 - self.b + self.b * _doc_len / self.avgdl)))
        return _score

