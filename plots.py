import json
import numpy as np
import matplotlib.pyplot as plt

with open('/Users/larsholdijk/Local/CodeIR/results/sentence_evaluation/drmm_sentence_evaluation_final.json',
          'r') as handle:
    data_drmm = json.load(handle)
with open('/Users/larsholdijk/Local/CodeIR/results/sentence_evaluation/bm25_sentence_evaluation_final.json',
          'r') as handle:
    data_bm25 = json.load(handle)


def statistics(data):
    rank_change = []
    length = []
    location = []
    for query in data['querry_data']:
        first_doc_id = list(query['sentences'].keys())[0]
        first_doc_sentence = query['sentences'][first_doc_id][0]

        rank_change.append(first_doc_sentence['rank_change'])
        length.append(first_doc_sentence['length'])
        location.append(first_doc_sentence['location'])

    rank_change = np.array(rank_change)
    length = np.array(length)
    location = np.array(location)

    return rank_change, length, location

def print_median(rank_change, length, location, model):
    print(f"Median rank_change {model} : {np.median(rank_change)}")
    print(f"Median length {model} : {np.median(length)}")
    print(f"Median location {model} : {np.median(location)}")



rank_change_drmm, length_drmm, location_drmm = statistics(data_drmm)
print_median(rank_change_drmm, length_drmm, location_drmm, 'drmm')

rank_change_bm25, length_bm25, location_bm25 = statistics(data_bm25)
print_median(rank_change_bm25, length_bm25, location_bm25, 'bm25')

width = 0.35
plt.style.use('ggplot')

bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 2000]
y_pos = np.arange(len(bins)-1)
counts_drmm = plt.hist(rank_change_drmm, bins)[0]
counts_bm25 = plt.hist(rank_change_bm25, bins)[0]
plt.clf()
bars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20+']
plt.bar(y_pos, counts_drmm, width, color='black', edgecolor='white', hatch='//')
plt.bar(y_pos + width, counts_bm25, width, color='darkgrey', edgecolor='white', hatch='.')
plt.xticks(y_pos + width / 2, bars)
plt.legend(['DRMM', 'BM25'])
plt.xlabel('Change in rank')
plt.ylabel('Count')
plt.show()
plt.clf()

bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 99999999]
y_pos = np.arange(len(bins)-1)
counts_drmm = plt.hist(length_drmm, bins)[0]
counts_bm25 = plt.hist(length_bm25, bins)[0]
plt.clf()
bars = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99', '100+']
plt.bar(y_pos, counts_drmm, width, color='black', edgecolor='white', hatch='//')
plt.bar(y_pos + width, counts_bm25, width, color='darkgrey', edgecolor='white', hatch='.')
plt.xticks(y_pos + width / 2, bars)
plt.legend(['DRMM', 'BM25'])
plt.xlabel('Length of sentence')
plt.ylabel('Count')
plt.show()
plt.clf()

bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
y_pos = np.arange(len(bins)-1)
counts_drmm = plt.hist(location_drmm, bins)[0]
counts_bm25 = plt.hist(location_bm25, bins)[0]
plt.clf()
bars = ['0.00-0.09', '0.10-0.19', '0.20-0.29', '0.30-0.39', '0.40-0.49', '0.50-0.59', '0.60-0.69', '0.70-0.79', '0.80-0.89', '0.90-1.00']
plt.bar(y_pos, counts_drmm, width, color='black', edgecolor='white', hatch='//')
plt.bar(y_pos + width, counts_bm25, width, color='darkgrey', edgecolor='white', hatch='.')
plt.xticks(y_pos + width / 2, bars, rotation='vertical')
plt.legend(['DRMM', 'BM25'])
plt.xlabel('Location of sentence')
plt.ylabel('Count')
plt.show()
plt.clf()
