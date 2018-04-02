import tensorflow as tf
import pandas as pd

DATA = "entole.csv"

CSV_COLUMN_NAMES = ['book', 'chapter', 'verse', 'author', 'author_type',
        'traditional_author', 'traditional_author_type', 'audience_type',
        'speaker', 'listener', 'raw_type', 'text_type', 'num_type', 'reason']

TYPE = ['torah', 'other', 'decalogue']

def load_data(y_name='text_type'):
    """Returns the entole data set as df, (train_x, train_y), test."""
    df = pd.read_csv(DATA, names=CSV_COLUMN_NAMES, header=0)
    # we don't need these columns
    df.pop('reason')
    df.pop('raw_type')
    df.pop('num_type')

    # Split the entire data set into two subsets:
    # 1. Training subset has known values for the type
    train_mask = df[y_name].notnull()
    train = df[train_mask]
    train_x, train_y = train, train.pop(y_name)

    # 2. Testing subset has unknown values for the type
    test_mask = df[y_name].isnull()
    test = df[test_mask]
    test.pop(y_name)

    return df, (train_x, train_y), test

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evauluation or prediction"""
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    # convert the inputs to a Dataset
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    return dataset
