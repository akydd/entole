import argparse
import tensorflow as tf

import load_data

parser = argparser.ArgumentParse()
parser.add_argument('--batch_size', default=20, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    # get data
    df, (train_x, train_y), test = entole_data.load_data()

    # Feature columns
    feature_columns = []
    # book
    #feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(
    #        key='book',
    #        vocabulary_list=["Matthew", "Mark", "Luke", "John", "Acts",
    #            "Romans", "1 Cor", "Eph", "Col", "1 Tim", "Titus", "Heb",
    #            "2 Peter", "1 John", "2 John", "Rev"]))

    for key in train_x.keys():
        feature_columns.append(tf.feature_columns.categorical_column_with_vocabulary_list(
        key=key,
        vocabulary_list = df[key].unique()))

    # build the DNN
    classifier = tf.estimator.DNNClassifier(
            feature_columns = feature_columns,
            hidden_units=[10, 10],
            n_classes = 3)

    # train the model
    classifier.train(
            input_fn=lambda:entole_data.train_input_fx(train_x, train_y, args.batch_size),
            steps =args.train_steps)

    # evaluate the model
    eval_result = classifier.evaluate(
            input_fn = lambda:enetole_data.eval_input_fn(train_x, train_y, args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Try the model for the test set
    predictions = classifier.predict(
            input_fn = lambda:entole_data.eval_input_fn(
                test,
                labels = None,
                batch_size = args.batch_size))

    template = ('\nComputed value is "{} ({:.1f}%)')

    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(entole_data.TYPE[class_id], 100 * probability))



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
