import tensorflow as tf
import pandas as pd

DATA = "entole.csv"

CSV_COLUMN_NAMES = ['book', 'chapter', 'verse', 'author', 'author type',
        'traditional author', 'traditional author type', 'audience type',
        'speaker', 'listener', 'raw type', 'type', 'reason']

COMMANDMENT_TYPE = ['torah', 'decalogue', 'other']

def load_data(y_name='commandment type'):
    """Returns the entole data set as (train_x, train_y), (test_x, test_y)."""
    df = pd.read_csv(DATA, names=CSV_COLUMN_NAMES, header=0)
    # we don't need these columns
    df.pop('reason')
    df.pop('raw type')

    # Split the entire data set into two subsets:
    # 1. Training subset has known values for the type
    train_mask = df['type'] != ''
    train = df[train_mask]
    train_x, train_y = train, train.pop(y_name)

    # 2. Testing subset has unknown values for the type
    test_mask = df['type'] == ''
    test = df[test_mask]
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)

