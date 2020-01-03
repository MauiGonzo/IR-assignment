import pandas as pd
import numpy as np
import matchzoo as mz
import nltk
import pickle

from matchzoo import DataPack, build_unit_from_data_pack, chain_transform, build_vocab_unit
from matchzoo.preprocessors import units

nltk.download('stopwords')
nltk.download('punkt')
from matchzoo.preprocessors.units import Unit

print('matchzoo version', mz.__version__)
print()

#DIRECTORIES
DATA = "dataset/robust/csvs"

class remove_empty_string(Unit):
    def transform(self, input_: str) -> list:
        x = filter(None, input_)
        return list(x)


def load_data():

    print("Loading data")
    _relations = pd.read_csv(DATA+"/relations.csv", index_col=0)

    _left = pd.read_csv(DATA+"/title_querries.csv", index_col=0)
    _left.index.name = "id_left"

    _right = pd.read_csv(DATA+"/full.csv", index_col=0)
    _right.index.name = "id_right"

    dp = DataPack(relation=_relations, left=_left, right=_right)

    return dp


# Data management
data_pack = load_data()

left_fixedlength_unit = units.FixedLength(10, pad_mode='post')
filter_unit = units.FrequencyFilter(low=10, mode='df')

default = [mz.preprocessors.units.tokenize.Tokenize(),
    mz.preprocessors.units.lowercase.Lowercase(),
    mz.preprocessors.units.punc_removal.PuncRemoval()]

context = {}

data_pack.apply_on_text(chain_transform(default), inplace=True)

fitted_filter_unit = build_unit_from_data_pack(filter_unit, data_pack, flatten=False, mode='right')
data_pack.apply_on_text(fitted_filter_unit.transform, mode='right', inplace=True)
data_pack.apply_on_text(remove_empty_string().transform, inplace=True)
with open("fitted_filter_unit", 'wb') as f:
    pickle.dump(fitted_filter_unit, f)

vocab_unit = build_vocab_unit(data_pack)
data_pack.apply_on_text(vocab_unit.transform, mode='both', inplace=True)
vocab_size = len(vocab_unit.state['term_index'])
with open("vocab_unit", 'wb') as f:
    pickle.dump(vocab_unit, f)

data_pack.append_text_length(inplace=True)
data_pack.apply_on_text(left_fixedlength_unit.transform, mode='left', inplace=True)

max_len_left = 10
data_pack.left['length_left'] = data_pack.left['length_left'].apply(lambda val: min(val, max_len_left))

data_pack.save("./processed_data_full")

glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=300)
embedding_matrix = glove_embedding.build_matrix(vocab_unit.state['term_index'])
# normalize the word embedding for fast histogram generating.
l2_norm = np.sqrt((embedding_matrix*embedding_matrix).sum(axis=1))
embedding_matrix_2 = embedding_matrix / l2_norm[:, np.newaxis]

np.save("./matrix", embedding_matrix_2)

