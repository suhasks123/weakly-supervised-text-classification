import numpy as np
np.random.seed(1234)
import os
from time import time
import csv
import keras.backend as K
# K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=30, inter_op_parallelism_threads=30)))
from keras.engine.topology import Layer
from keras.layers import Dense, Input, Convolution1D, Embedding, GlobalMaxPooling1D, GRU, TimeDistributed
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import initializers, regularizers, constraints
from keras.initializers import VarianceScaling, RandomUniform
from sklearn.metrics import f1_score


def f1(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    f1 = f1_score(y_true, y_pred, average='macro')
    return f1

def accuracy(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    no_of_right_pred = 0
    for i in range(0, y_pred.size):
        if y_pred[i] == y_true[i]:
            no_of_right_pred += 1
    accuracy_value = no_of_right_pred/y_pred.size
    return accuracy_value

def ConvolutionLayer(input_shape, n_classes, filter_sizes=[2, 3, 4, 5], num_filters=20, word_trainable=False, vocab_sz=None,
                     embedding_matrix=None, word_embedding_dim=100, hidden_dim=20, act='relu', init='ones'):
    x = Input(shape=(input_shape,), name='input')
    z = Embedding(vocab_sz, word_embedding_dim, input_length=(input_shape,), name="embedding",
                    weights=[embedding_matrix], trainable=word_trainable)(x)
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters, kernel_size=sz, padding="valid", activation=act, strides=1, kernel_initializer=init)(z)
        conv = GlobalMaxPooling1D()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    z = Dense(hidden_dim, activation="relu")(z)
    y = Dense(n_classes, activation="softmax")(z)
    return Model(inputs=x, outputs=y, name='classifier')


def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class WSTC(object):
    def __init__(self,
                 input_shape,
                 n_classes=None,
                 init=RandomUniform(minval=-0.01, maxval=0.01),
                 y=None,
                 model='cnn',
                 vocab_sz=None,
                 word_embedding_dim=100,
                 embedding_matrix=None
                 ):

        super(WSTC, self).__init__()

        self.input_shape = input_shape
        self.y = y
        self.n_classes = n_classes

        # Obtain CNN model
        self.classifier = ConvolutionLayer(self.input_shape[1], n_classes=n_classes,
                                                vocab_sz=vocab_sz, embedding_matrix=embedding_matrix, 
                                                word_embedding_dim=word_embedding_dim, init=init)

        self.model = self.classifier
        self.sup_list = {}

    def pretrain(self, x, pretrain_labels, sup_idx=None, optimizer='adam',
                 loss='kld', epochs=200, batch_size=256, save_dir=None):

        self.classifier.compile(optimizer=optimizer, loss=loss)
        print("\nNeural model summary: ")
        self.model.summary()

        if sup_idx is not None:
            for i, seed_idx in enumerate(sup_idx):
                for idx in seed_idx:
                    self.sup_list[idx] = i

        # begin pretraining
        t0 = time()
        print('\nPretraining...')
        self.classifier.fit(x, pretrain_labels, batch_size=batch_size, epochs=epochs)
        print('Pretraining time: {:.2f}s'.format(time() - t0))
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.classifier.save_weights(save_dir + '/pretrained.h5')
            print('Pretrained model saved to {}/pretrained.h5'.format(save_dir))
        self.pretrained = True

    def load_weights(self, weights):
        self.model.load_weights(weights)

    def predict(self, x):
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    def target_distribution(self, q, power=2):
        weight = q**power / q.sum(axis=0)
        p = (weight.T / weight.sum(axis=1)).T
        for i in self.sup_list:
            p[i] = 0
            p[i][self.sup_list[i]] = 1
        return p

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, x, y=None, maxiter=5e4, batch_size=256, tol=0.1, power=2,
            update_interval=140, save_dir=None, save_suffix=''):

        print('Update interval: {}'.format(update_interval))

        pred = self.classifier.predict(x)
        y_pred = np.argmax(pred, axis=1)
        y_pred_last = np.copy(y_pred)

        # logging file
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/self_training_log_{}.csv'.format(save_suffix), 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'f1_score', 'accuracy'])
        logwriter.writeheader()

        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)

                y_pred = q.argmax(axis=1)
                p = self.target_distribution(q, power)
                print('\nIter {}: '.format(ite), end='')
                if y is not None:
                    f1_score= np.round(f1(y, y_pred), 5)
                    accuracy_value = np.round(accuracy(y, y_pred), 5)
                    logdict = dict(iter=ite, f1_score=f1_score, accuracy=accuracy)
                    logwriter.writerow(logdict)
                    print('f1_score = {}, Accuracy = {}'.format(f1_score, accuracy_value))
                    
                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                print('Fraction of documents with label changes: {} %'.format(np.round(delta_label*100, 3)))
                if ite > 0 and delta_label < tol/100:
                    print('\nFraction: {} % < tol: {} %'.format(np.round(delta_label*100, 3), tol))
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
            self.model.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

            ite += 1

        logfile.close()

        if save_dir is not None:
            self.model.save_weights(save_dir + '/final.h5')
            print("Final model saved to: {}/final.h5".format(save_dir))
        return self.predict(x)
