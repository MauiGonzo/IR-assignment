import numpy as np
import matchzoo as mz
import nltk

from matchzoo import load_data_pack


nltk.download('stopwords')
nltk.download('punkt')

print('matchzoo version', mz.__version__)
print()


# Data management
preprocessed = load_data_pack("./processed_data_full")

ranking_task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=1))
ranking_task.metrics = [
    mz.metrics.Precision(k=20),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=20),
    mz.metrics.MeanAveragePrecision()
]

bin_size = 30
model = mz.models.DRMM()
model.params['input_shapes'] = [[10,], [10, bin_size,]]
model.params['task'] = ranking_task
model.params['mask_value'] = 0
model.params['embedding_input_dim'] = 160799
model.params['embedding_output_dim'] = 300
model.params['embedding_trainable'] = False
model.params['mlp_num_layers'] = 1
model.params['mlp_num_units'] = 10
model.params['mlp_num_fan_out'] = 1
model.params['mlp_activation_func'] = 'tanh'
model.params['optimizer'] = 'adadelta'
model.build()
model.compile()
model.backend.summary()

embedding_matrix = np.load('./matrix.npy')
model.load_embedding_matrix(embedding_matrix)

hist_callback = mz.data_generator.callbacks.Histogram(embedding_matrix, bin_size=30, hist_mode='LCH')
        # Bin size specified at end of par 5.2
        # Hist_mode based on best scoring

pred_generator = mz.DataGenerator(preprocessed, mode='point', callbacks=[hist_callback])
pred_x, pred_y = pred_generator[:]
evaluate = mz.callbacks.EvaluateAllMetrics(model,
                                           x=pred_x,
                                           y=pred_y,
                                           once_every=1,
                                           batch_size=len(pred_y),
                                           model_save_path='./drmm_pretrained_model/'
                                          )

train_generator = mz.DataGenerator(preprocessed, mode='pair', num_dup=1, num_neg=1, batch_size=64,
                                   callbacks=[hist_callback])
print('num batches:', len(train_generator))

history = model.fit_generator(train_generator, epochs=30, callbacks=[evaluate], use_multiprocessing=False, verbose=1)

drmm_model = mz.load_model('./drmm_pretrained_model/30')
test_generator = mz.DataGenerator(data_pack=preprocessed[:100], mode='point', callbacks=[hist_callback])
test_x, test_y = test_generator[:]
prediction = drmm_model.predict(test_x)
print(prediction)
print(test_y)

np.savetxt("./prediction.txt", prediction, fmt='%1.3f')
np.savetxt("./test_y.txt", test_y, fmt='%1.3f')
zipped = np.array(list(zip(test_y[:, 0], prediction[:, 0])))
print(zipped)
np.savetxt("./zipped.txt", zipped, fmt='%1.3f')