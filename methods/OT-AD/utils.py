import numpy as np


seed_dic = {'abu-airport-1': 5130, 'abu-airport-2': 5599, 'abu-airport-3': 1024,
             'abu-airport-4': 7105, 'abu-beach-1': 1018, 'abu-beach-2': 9126,
             'abu-beach-3': 6140, 'abu-beach-4': 7507, 'abu-urban-1': 7210,
             'abu-urban-2': 9905, 'abu-urban-3': 8180, 'abu-urban-4': 2413,
             'abu-urban-5': 2425, 'cri': 7305, 'hydice': 8123, 'sandiego': 5521}


def hyper_norm(data):
    """
    Normalize the input data according to the specified method, scaling values to [0, 1].
    """

    data = data.astype(np.float64)
    min_val = np.min(data)
    max_val = np.max(data)
    norm = data - min_val
    if max_val == min_val:
        norm = np.zeros(data.shape)
    else:
        norm /= (max_val - min_val)
    return norm


