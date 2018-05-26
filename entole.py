import argparse
import tensorflow as tf
import numpy as np

import entole_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=20, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    # get data
    df, (train_x, train_y), test = entole_data.load_data()

    # create Feature columns
    feature_columns = []
    for key in train_x.keys():
        #print "The key is {}".format(key)
        #print "Unique values are {}".format(df[key].unique())
        new_col = tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                key = key,
                vocabulary_list = df[key].unique()
            )
        )
        #print "new_col {} is ok".format(new_col)
        feature_columns.append(new_col)
        # build the DNN
        classifier = tf.estimator.DNNClassifier(
            feature_columns = feature_columns,
            hidden_units = [7],
            n_classes = 3,
            label_vocabulary = entole_data.TYPE
    )

    results = {}
    for x in range(0, 10):
        # train the model
        classifier.train(
                input_fn = lambda:entole_data.train_input_fn(
                    train_x, 
                    train_y, 
                    args.batch_size
                ),
                steps = args.train_steps
        )

        # evaluate the model
        eval_result = classifier.evaluate(
                input_fn = lambda:entole_data.eval_input_fn(
                    train_x, 
                    train_y, 
                    args.batch_size
                )
        )

        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

        # Run the model on the test set
        predictions = classifier.predict(
                input_fn = lambda:entole_data.eval_input_fn(
                    test,
                    labels = None,
                    batch_size = args.batch_size
                )
        )

        # populate results
        for verse, pred_dict in zip(test['verse'].ravel(), predictions):
            if verse not in results:
                results[verse] = {}
                
            for p, t in zip(pred_dict['probabilities'], entole_data.TYPE):
                if t not in results[verse]:
                    results[verse][t] = []
                results[verse][t].append(100 * p)

    # print results
    for verse, result in results.iteritems():
        print('{}'.format(verse))
        for t, data in result.iteritems():
            print('{} - Average={:.3f}%, Variance={}'.format(t, np.mean(data), np.var(data)))
        print('\n')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.WARN)
    tf.app.run(main)
