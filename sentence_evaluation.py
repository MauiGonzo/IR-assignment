import pickle
from os import path

import numpy as np
import matchzoo as mz
import nltk
import pandas as pd
import json

from matchzoo import load_data_pack, DataPack, chain_transform
from matchzoo.preprocessors import units
from matchzoo.preprocessors.units import Unit

from rank_bm25 import BM25Okapi

from nltk.tokenize import sent_tokenize

from models.bm25.bm25_evaluation import bm25_evaluation

nltk.download('stopwords')
nltk.download('punkt')

print('matchzoo version', mz.__version__)
print()


# DIRECTORIES USED:
MODEL_DRMM = 'models/drmm/19'

PREPROCESSED_DATASET = "dataset/robust/csvs/full.csv"
UNPROCESSED_DATASET = 'dataset/robust/csvs'

MATRIX = 'dataset/robust/preprocessed/matrix.npy'
UNITS = 'dataset/robust/preprocessing_units'

BASE_EVA = "results/base_evaluation"

# OPTIONS
OVERWRITE_BM25_EVALUATION = False
OVERWRITE_DRMM_EVALUATION = False


def calculate_prec_at(sorted, relevant_ids, k = 20):
    top_k = sorted[:k]
    retrieved = np.count_nonzero(top_k.isin(relevant_ids))

    return retrieved / k


def calculate_map(sorted, relevant_ids):
    sorted_position = np.argwhere(sorted.isin(relevant_ids))[:,0] + 1
    if len(sorted_position) == 0:
        return 0
    step = np.arange(len(sorted_position)) + 1
    precisions = np.divide(step, sorted_position)
    summed = np.sum(precisions)

    return summed / len(sorted_position)


def calculate_ndcg(sorted, relevant_ids, k = 20):
    top_k = sorted[:k]
    sorted_position = np.argwhere(top_k.isin(relevant_ids))[:, 0] + 1

    if len(sorted_position > 0) and sorted_position[0] == 1:
        ndcg = 1
        sorted_position = sorted_position[1:]
    else:
        ndcg = 0

    ndcg = ndcg + np.sum(1 / np.log2(sorted_position))

    step = np.arange(k)[1:] + 1
    idcg = 1 + np.sum(1 / np.log2(step))

    return ndcg / idcg


def evaluatebm_25(override = False):

    if not override:
        if path.exists(BASE_EVA + '/bm25_final_top_2000.json'):
            with open(BASE_EVA + '/bm25_final_top_2000.json', 'r') as handle:
                bm25_data = json.load(handle)

            return bm25_data

    preprocessed = load_data_pack("PREPROCESSED_DATASET")

    corpus = preprocessed.right.text_right

    tokenized_corpus = list(corpus)
    bm25 = BM25Okapi(tokenized_corpus)

    precision = 0
    average_precision = 0
    ndcg = 0

    bm25_data = {}

    for index, row in preprocessed.left.iterrows():
        print(index)
        # if index < 634:
        #     continue
        q = row.text_left[:row.length_left]
        relevant_ids = preprocessed.relation.query('id_left == ' + str(index) + ' & label >= 1').id_right

        scores = bm25.get_scores(q)
        sorted_scores = -np.sort(-scores)[:2000]
        sorted = np.argsort(-scores)[:2000]
        sorted_ids = corpus[sorted].index

        precision_q = calculate_prec_at(sorted_ids, relevant_ids)
        average_precision_q = calculate_map(sorted_ids, relevant_ids)
        ndcg_q = calculate_ndcg(sorted_ids, relevant_ids)

        top_2000 = sorted_ids

        info = {
            "top_2000" : list(top_2000),
            "scores": list(sorted_scores),
            "precision" : precision_q,
            "average_precision" : average_precision_q,
            "ndcg" : ndcg_q
        }

        bm25_data[index] = info

        precision += precision_q
        average_precision += average_precision_q
        ndcg += ndcg_q

        with open('./bm25_top_2000.json', 'wb') as f:
            pickle.dump(bm25_data, f)

    combined_info = {
        "precision" : precision / len(preprocessed.left),
        "average_precision" : average_precision / len(preprocessed.left),
        "ndcg" : ndcg / len(preprocessed.left)
    }
    bm25_data["final"] = combined_info

    with open('./bm25_final_top_2000.json', 'wb') as f:
        pickle.dump(bm25_data, f)

    return bm25_data


def evaluate_drmm(bm25_data, override = False):

    if not override:
        if path.exists(BASE_EVA+'/drmm_final_top_2000.json'):
            with open(BASE_EVA+'/drmm_final_top_2000.json', 'r') as handle:
                drmm_data = json.load(handle)

            return drmm_data

    embedding_matrix = np.load(MATRIX)
    bin_size = 30
    mode = 'LCH'

    preprocessed = load_data_pack(PREPROCESSED_DATASET)

    drmm_model = mz.load_model(MODEL_DRMM)
    drmm_model.load_embedding_matrix(embedding_matrix)

    ground_truth_labels = preprocessed.relation.copy()

    classes = list(bm25_data.keys())
    classes.remove("final")

    precision = 0
    average_precision = 0
    ndcg = 0

    drmm_data = {}

    for class_id in classes:
        print(class_id)
        ids = bm25_data[class_id]['top_2000']
        querry = [class_id] * len(ids)
        label = [0] * len(ids)
        d = {'id_left': querry, 'id_right': ids, 'label': label}
        df = pd.DataFrame(data=d)

        gt = list(ground_truth_labels.query('id_left == ' + str(class_id) + ' & label >= 1').id_right)

        df.loc[df.id_right.isin(gt), 'label'] = 1

        preprocessed._relation = df

        x, y = build_match_histogram(preprocessed, embedding_matrix, bin_size, mode)

        pred = drmm_model.predict(x)

        sorted_scores = -np.sort(-pred[:, 0])
        sorted = np.argsort(-pred[:, 0])
        sorted_ids_nd = x['id_right'][sorted]
        sorted_ids = pd.Series(sorted_ids_nd)  # preprocessed.right[preprocessed.right.index.isin(sorted_ids_nd)].index

        relevant_ids = list(x['id_right'][np.argwhere(y[:, 0])][:, 0])

        precision_q = calculate_prec_at(sorted_ids, relevant_ids)
        average_precision_q = calculate_map(sorted_ids, relevant_ids)
        ndcg_q = calculate_ndcg(sorted_ids, relevant_ids)

        info = {
            "top_2000": list(sorted_ids),
            "scores": list(map(float, sorted_scores)),
            "precision": float(precision_q),
            "average_precision": float(average_precision_q),
            "ndcg": float(ndcg_q)
        }

        drmm_data[class_id] = info

        precision += precision_q
        average_precision += average_precision_q
        ndcg += ndcg_q

        with open('./drmm_top_2000.json', 'w') as f:
            json.dump(drmm_data, f)

    combined_info = {
        "precision": float(precision / len(preprocessed.left)),
        "average_precision": float(average_precision / len(preprocessed.left)),
        "ndcg": float(ndcg / len(preprocessed.left))
    }
    drmm_data["final"] = combined_info

    with open('./drmm_final_top_2000.json', 'w') as f:
        json.dump(drmm_data, f)


def build_match_histogram(data, embedding_matrix, bin_size, mode):

    x, y = data.unpack()

    matching_hists = []

    text_left = x['text_left'].tolist()
    text_right = [row[:x['length_right'].tolist()[idx]] for idx, row in enumerate(x['text_right'].tolist())]

    for pair in zip(text_left, text_right):
        left, right = pair

        matching_hist = np.ones((len(left), bin_size),
                                dtype=np.float32)
        embed_left = embedding_matrix[left]
        embed_right = embedding_matrix[right]
        matching_matrix = embed_left.dot(np.transpose(embed_right))

        for (i, j), value in np.ndenumerate(matching_matrix):
            bin_index = int((value + 1.) / 2. * (bin_size - 1.))
            matching_hist[i][bin_index] += 1.0
        if mode == 'NH':
            matching_sum = matching_hist.sum(axis=1)
            matching_hist = matching_hist / matching_sum[:, np.newaxis]
        elif mode == 'LCH':
            matching_hist = np.log(matching_hist)

        matching_hists.append(matching_hist)

    x['match_histogram'] = np.asarray(matching_hists)
    return x, y


def load_preprocessed_data():
    preprocessed = load_data_pack(PREPROCESSED_DATASET)
    return preprocessed


def load_data():

    print("Loading data")
    _relations = pd.read_csv(UNPROCESSED_DATASET+"/relations.csv", index_col=0)

    _left = pd.read_csv(UNPROCESSED_DATASET+"/title_querries.csv", index_col=0)
    _left.index.name = "id_left"

    _right = pd.read_csv(UNPROCESSED_DATASET+"/full.csv", index_col=0)
    _right.index.name = "id_right"

    dp = DataPack(relation=_relations, left=_left, right=_right)

    return dp


def load_units():
    # load preprocessing units
    with open(UNITS+"/vocab_unit", 'rb') as f:
        vocab_unit = pickle.load(f)
    with open(UNITS+"/fitted_filter_unit",'rb') as f:
        fitted_filter_unit = pickle.load(f)
    return vocab_unit, fitted_filter_unit


def create_sentence_dataset(highest_scoring_id, class_id):

    full_dataset = load_data() # We only want to gave the dataset in memory for a short period
    document = full_dataset.right.loc[highest_scoring_id]

    text = document.text_right
    sentences = sent_tokenize(text)

    new_documents = []

    for idx, val in enumerate(sentences):
        new_documents.append(" ".join(sentences[:idx] + sentences[idx + 1:]))

    new_documents.append(" ".join(sentences))
    sentences.append("")

    # Create new dataframe for documents
    data = {'text_right': new_documents, 'sentence': sentences}
    f = lambda x: "doc-" + str(x)
    ids = list(map(f, range(0, len(sentences))))
    new_right = pd.DataFrame(data, index=ids)
    new_right.index.name = 'id_right'

    # Create new dataframe for relations
    querry = [int(class_id)] * len(sentences)
    label = [1] * len(sentences)
    d = {'id_left': querry, 'id_right': ids, 'label': label}
    new_relations = pd.DataFrame(data=d)

    new_left = full_dataset.left.loc[int(class_id)].to_frame().T
    new_left.index.name = 'id_left'

    del full_dataset

    sentence_data_pack = DataPack(relation=new_relations, left=new_left, right = new_right)

    return sentence_data_pack #full_dataset[:50]#


def preproces(sentence_data_pack, fitted_filter_unit, vocab_unit):
    class remove_empty_string(Unit):
        def transform(self, input_: str) -> list:
            x = filter(None, input_)
            return list(x)

    left_fixedlength_unit = units.FixedLength(10, pad_mode='post')

    default = [mz.preprocessors.units.tokenize.Tokenize(),
               mz.preprocessors.units.lowercase.Lowercase(),
               mz.preprocessors.units.punc_removal.PuncRemoval()]

    sentence_data_pack.apply_on_text(chain_transform(default), inplace=True)

    sentence_data_pack.apply_on_text(fitted_filter_unit.transform, mode='right', inplace=True)
    sentence_data_pack.apply_on_text(remove_empty_string().transform, inplace=True)

    sentence_data_pack.apply_on_text(vocab_unit.transform, mode='both', inplace=True)

    sentence_data_pack.append_text_length(inplace=True)
    sentence_data_pack.apply_on_text(left_fixedlength_unit.transform, mode='left', inplace=True)

    max_len_left = 10
    sentence_data_pack.left['length_left'] = sentence_data_pack.left['length_left'].apply(lambda val: min(val, max_len_left))

    return sentence_data_pack


def sentence_evaluation_bm25(bm25_data):
    print("Started sentence evaluation of BM25")
    vocab_unit, fitted_filter_unit = load_units()
    classes = list(bm25_data.keys())
    classes.remove("final")

    print("Loading preprocessed data")
    corpus = load_preprocessed_data()

    print("Create bm25 classifier")
    tokenized_corpus = list(corpus.right.text_right)
    bm25 = bm25_evaluation(tokenized_corpus)

    bins = np.arange(0, 1.0, 0.04)
    location_count = np.zeros(25)
    location_change = np.zeros(25)
    location_highest_impact = np.zeros(25)
    data = []

    for class_id in classes:

        querry_data = {'querry_id': class_id,
                'sentences': {}}

        for rank in range(0, 3):
            print("Build sentence dataset for querry : " + str(class_id))
            document_id = bm25_data[class_id]['top_2000'][rank]
            sentence_data_pack = create_sentence_dataset(document_id, class_id)

            preprocessed = preproces(sentence_data_pack, fitted_filter_unit, vocab_unit)

            print("Obtain scores for sentence augmented dataset")
            q = preprocessed.left.loc[int(class_id)].text_left[:preprocessed.left.loc[int(class_id)].length_left]
            # scores_ori = bm25.get_scores(query = q)
            new_scores = bm25.get_new_scores(q, list(preprocessed.right.text_right))
            ori_scores = bm25_data[class_id]['scores']

            # Calculate change in ranking
            change_in_ranks = [int(np.searchsorted(-np.array(ori_scores[:rank] + ori_scores[rank+1:]), -new_score)) - rank for new_score in
                               new_scores[:]]
            change_in_scores = np.abs(new_scores - ori_scores[rank])

            sorted_sentences = np.lexsort((-change_in_scores, -np.abs(np.array(change_in_ranks))))

            sentence_data = []
            for sentence_id in sorted_sentences[:5]:
                sentence_data.append({
                    "sentence": str(sentence_data_pack.right.iloc[sentence_id].sentence),
                    "rank_change": int(change_in_ranks[sentence_id]),
                    "score_change": float(change_in_scores[sentence_id]),
                    "length": int(len(sentence_data_pack.right.iloc[sentence_id].sentence.split())),
                    "location" : float(sentence_id / len(sorted_sentences))
                })

            querry_data['sentences'][document_id] = sentence_data

            # print(querry_data)

            if rank == 0:
                # Update ranking
                ori_ranking = bm25_data[class_id]['top_2000'].copy()
                top_ranking_document = ori_ranking[0]
                new_ranking = []
                for i in range(0, len(new_scores)):
                    new_ranking.append(ori_ranking.copy()[1:])
                    new_ranking[i].insert(change_in_ranks[i], top_ranking_document)

                # Calculate change in scores
                relevant_ids = corpus.relation.query('id_left == ' + str(class_id) + ' & label >= 1').id_right
                precisions = [calculate_prec_at(pd.Series(new_ranking[j]), relevant_ids) for j in range(0, len(new_ranking))]
                average_precisions = [calculate_map(pd.Series(new_ranking[j]), relevant_ids) for j in range(0, len(new_ranking))]
                ncdgs = [calculate_ndcg(pd.Series(new_ranking[j]), relevant_ids) for j in range(0, len(new_ranking))]

                ori_precision = bm25_data[class_id]['precision']
                ori_average_precision =bm25_data[class_id]['average_precision']
                ori_ndcg = bm25_data[class_id]['ndcg']

                highest_change_precision = np.abs(np.array(precisions) - ori_precision).max()
                highest_change_average_precision = np.abs(np.array(average_precisions) - ori_average_precision).max()
                highest_change_ndcg = np.abs(np.array(ncdgs) - ori_ndcg).max()

                querry_data['change_precision'] = float(highest_change_precision)
                querry_data['change_average_precision'] = float(highest_change_average_precision)
                querry_data['change_ndcg'] = float(highest_change_ndcg)
                querry_data['change_rank'] = int(max(change_in_ranks))
                querry_data['sentence_length'] = int(len(sentence_data_pack.right.iloc[np.argmax(np.abs(
                    change_in_ranks))].sentence.split()))

                highest_bin = np.digitize(np.argmax(np.abs(change_in_ranks)) / len(change_in_ranks), bins) - 1
                location_highest_impact[highest_bin] = location_highest_impact[highest_bin] + 1

                for idx, value in enumerate(change_in_ranks):
                    location = idx/len(change_in_ranks)
                    binplace = np.digitize(location, bins) - 1
                    location_count[binplace] = location_count[binplace] + 1
                    location_change[binplace] = location_change[binplace] + np.abs(value)

        data.append(querry_data)

        with open('./bm25_sentence_evaluation.json', 'w') as f:
            json.dump(data, f)

    final = {
        "querry_data" : data,
        "highest_impact_location": list(map(float, list(location_highest_impact / 250))),
        "average_impact_location": list(map(float, list((location_change / location_count) / 250)))
    }

    print(final)

    with open('./bm25_sentence_evaluation_final.json', 'w') as f:
        json.dump(final, f)

    print("Done with sentence evaluation bm25")


def sentence_evaluation_drmm(drmm_data):
    print("Started sentence evaluation of DRMM")
    vocab_unit, fitted_filter_unit = load_units()
    classes = list(drmm_data.keys())
    classes.remove("final")

    print("Loading preprocessed data")
    corpus = load_preprocessed_data()

    print("Load model")
    embedding_matrix = np.load(MATRIX)
    bin_size = 30
    mode = 'LCH'

    drmm_model = mz.load_model(MODEL_DRMM)
    drmm_model.load_embedding_matrix(embedding_matrix)


    bins = np.arange(0, 1.0, 0.04)
    location_count = np.zeros(25)
    location_change = np.zeros(25)
    location_highest_impact = np.zeros(25)
    data = []

    for class_id in classes:

        querry_data = {'querry_id': int(class_id),
                'sentences': {}}

        for rank in range(0, 3):
            print("Build sentence dataset for querry : " + str(class_id))
            document_id = drmm_data[class_id]['top_2000'][rank]
            sentence_data_pack = create_sentence_dataset(document_id, int(class_id))

            preprocessed = preproces(sentence_data_pack, fitted_filter_unit, vocab_unit)

            print("Obtain scores for sentence augmented dataset")
            x, y = build_match_histogram(preprocessed, embedding_matrix, bin_size, mode)
            new_scores = drmm_model.predict(x)
            ori_scores = drmm_data[class_id]['scores']

            # Calculate change in ranking
            change_in_ranks = np.array([int(np.searchsorted(-np.array(ori_scores[:rank] + ori_scores[rank+1:]), -new_score)) - rank for new_score in new_scores[:]])
            change_in_scores = np.abs(new_scores - ori_scores[rank])[:, 0]

            sorted_sentences = np.lexsort((-change_in_scores, -np.abs(np.array(change_in_ranks))))

            sentence_data = []
            for sentence_id in sorted_sentences[:5]:
                sentence_data.append({
                    "sentence": str(sentence_data_pack.right.iloc[sentence_id].sentence),
                    "rank_change": int(change_in_ranks[sentence_id]),
                    "score_change": float(change_in_scores[sentence_id]),
                    "length": int(len(sentence_data_pack.right.iloc[sentence_id].sentence.split())),
                    "location" : float(sentence_id / len(sorted_sentences))
                })

            querry_data['sentences'][document_id] = sentence_data

            # print(querry_data)

            if rank == 0:
                # Update ranking
                ori_ranking = drmm_data[class_id]['top_2000'].copy()
                top_ranking_document = ori_ranking[0]
                new_ranking = []
                for i in range(0, len(new_scores)):
                    new_ranking.append(ori_ranking.copy()[1:])
                    new_ranking[i].insert(change_in_ranks[i], top_ranking_document)

                # Calculate change in scores
                relevant_ids = corpus.relation.query('id_left == ' + str(class_id) + ' & label >= 1').id_right
                precisions = [calculate_prec_at(pd.Series(new_ranking[j]), relevant_ids) for j in range(0, len(new_ranking))]
                average_precisions = [calculate_map(pd.Series(new_ranking[j]), relevant_ids) for j in range(0, len(new_ranking))]
                ncdgs = [calculate_ndcg(pd.Series(new_ranking[j]), relevant_ids) for j in range(0, len(new_ranking))]

                ori_precision = drmm_data[class_id]['precision']
                ori_average_precision =drmm_data[class_id]['average_precision']
                ori_ndcg = drmm_data[class_id]['ndcg']

                highest_change_precision = np.abs(np.array(precisions) - ori_precision).max()
                highest_change_average_precision = np.abs(np.array(average_precisions) - ori_average_precision).max()
                highest_change_ndcg = np.abs(np.array(ncdgs) - ori_ndcg).max()

                querry_data['change_precision'] = float(highest_change_precision)
                querry_data['change_average_precision'] = float(highest_change_average_precision)
                querry_data['change_ndcg'] = float(highest_change_ndcg)
                querry_data['change_rank'] = int(max(change_in_ranks))

                highest_bin = np.digitize(np.argmax(np.abs(change_in_ranks)) / len(change_in_ranks), bins) - 1
                location_highest_impact[highest_bin] = location_highest_impact[highest_bin] + 1

                for idx, value in enumerate(change_in_ranks):
                    location = idx/len(change_in_ranks)
                    binplace = np.digitize(location, bins) - 1
                    location_count[binplace] = location_count[binplace] + 1
                    location_change[binplace] = location_change[binplace] + np.abs(value)

        data.append(querry_data)

        with open('./drmm_sentence_evaluation.json', 'w') as f:
            json.dump(data, f)

    final = {
        "querry_data" : data,
        "highest_impact_location": list(map(float, list(location_highest_impact / 250))),
        "average_impact_location": list(map(float, list((location_change / location_count) / 250)))
    }

    print(final)

    with open('./drmm_sentence_evaluation_final.json', 'w') as f:
        json.dump(final, f)

    print("Done with sentence evaluation drmm")


bm25_data = evaluatebm_25(OVERWRITE_BM25_EVALUATION)
drmm_data = evaluate_drmm(bm25_data, OVERWRITE_DRMM_EVALUATION)

sentence_evaluation_drmm(drmm_data)
sentence_evaluation_bm25(bm25_data)
