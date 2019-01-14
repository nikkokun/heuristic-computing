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
import sys
import getopt
import json

def usage():
    print('-m = string\tmodel id')

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

def main(opts):
    """ Main entry point of the app """

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit(2)
        elif opt in ('-m', '--model'):
            model_id = arg
        else:
            usage()
            sys.exit(2)

    reset_tf_session()

    email = os.getenv('GMAIL')
    yag = yagmail.SMTP(email)

    top_n = {1: None, 2: None, 4: None, 8: None, 16: None, 32: None, 64: None}

    model_metrics = ModelMetrics(model_id, 32)

    model_metrics.create_codes()
    model_metrics.build_annoy_index()

    for key, value in top_n.items():
        print(F'getting scores top-{key} scores for {model_id}...')
        results = model_metrics.get_scores(key)
        top_n[key] = results
        print(results)
        break

    with open('results.json', 'r') as f:
        results_json = json.load(f)

    results = json.loads(results_json)
    results[model_id] = top_n

    results_json = json.dumps(results)

    mail_content = F'''
    Finished measuring accuracy for {model_id}.
    The results are:
    {results_json}
    '''

    yag.send(to='mizutaninikkou@gmail.com', subject='Finished Measuring Accuracy', contents=mail_content)
    with open('results.json', 'w') as f:
        json.dump(results_json, f)

if __name__ == "__main__":
    """ This is executed when run from the command line """

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'm:h', ['model=', 'help'])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    main(opts)