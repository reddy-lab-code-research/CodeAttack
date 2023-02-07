import pandas as pd

def get_summarization_data(data_path):
    data = pd.read_json(data_path, lines=True, orient='records')
    columns = ['code','docstring']
    data = data[columns]
    features = data.values.tolist()
    return features


def get_translation_data(x, y):
    data_x = open(x).readlines()
    data_y = open(y).readlines()
    data = pd.DataFrame(data_x, columns=['input_code']) 
    data['output_code'] = data_y
    features = data.values.tolist()
    return features

