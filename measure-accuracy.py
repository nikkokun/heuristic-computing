#!/usr/bin/env python3

__author__ = "nicoroble"
__version__ = "0.1.0"
__license__ = "MIT"

import tensorflow as tf
import keras.backend as K
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist
from keras.models import load_model
from annoy import AnnoyIndex
from sklearn.metrics import classification_report
import yagmail
import os

def reset_tf_session():
    curr_session = tf.get_default_session()
    # close current session
    if curr_session is not None:
        curr_session.close()
    # reset graph
    K.clear_session()
    # create new session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    s = tf.InteractiveSession(config=config)
    K.set_session(s)

class ModelMetrics():

    def __init__(self, model_id, bitsize):
        reset_tf_session()

        model_path = 'fashion-models/'
        self.model = load_model(F'{model_path}encoder{model_id}.h5')
        self.bitsize = bitsize

        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        self.x_val, self.x_test, self.y_val, self.y_test = train_test_split(x_test,
                                                                            y_test,
                                                                            test_size=0.2,
                                                                            random_state=33
                                                                            )

        self.x_test = self.x_test.reshape([-1, 28, 28, 1])
        self.x_val = self.x_val.reshape([-1, 28, 28, 1])
        self.x_test = self.x_test.astype('float32') / 255.0
        self.x_val = self.x_val.astype('float32') / 255.0

    def create_codes(self):
        print('creating codes...\n')
        self.test_codes = self.model.predict(self.x_test)
        self.val_codes = self.model.predict(self.x_val)
        print('codes have been created\n')

    def build_annoy_index(self):
        print('building annoy index...\n')

        self.annoy_index = AnnoyIndex(self.bitsize, metric='angular')

        for i in range(len(self.val_codes)):
            self.annoy_index.add_item(i, self.val_codes[i])

        self.annoy_index.build(1000)

    def get_scores(self, n):
        y_true_vals = []
        y_pred_vals = []

        for i in range(len(self.test_codes)):
            y_true = [self.y_test[i]] * n
            for val in y_true:
                y_true_vals.append(val)

            y_pred = self.annoy_index.get_nns_by_vector(vector=self.test_codes[i],
                                               n=n,
                                               search_k=200000
                                               )
            y_pred = self.y_val[y_pred]
            for val in y_pred:
                y_pred_vals.append(val)

        result = classification_report(y_true_vals, y_pred_vals, output_dict=True)['weighted avg']

        return result

def main():
    """ Main entry point of the app """

    reset_tf_session()

    model_id = 'multitaskfinal'

    email = os.environ.get('GMAIL')
    pswd = os.environ.get('GMAILPASS')
    yag = yagmail.SMTP(email, pswd)

    top_n = {1: None, 2: None}

    model_metrics = ModelMetrics(model_id, 32)

    model_metrics.create_codes()
    model_metrics.build_annoy_index()

    for key, value in top_n.items():
        print(F'getting scores for {model_id} for top-{key} results...')
        results = model_metrics.get_scores(key)
        top_n[key] = results
        print(results)
        print('\n')

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()