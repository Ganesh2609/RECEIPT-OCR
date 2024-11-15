import os
import random
import numpy as np
import torch
import torch_geometric.data
from graph import Grapher
from tqdm import tqdm
from bpemb import BPEmb
from sentence_transformers import SentenceTransformer
from torch_geometric.utils.convert import from_networkx

bpemb = BPEmb(lang='en')
sent_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')


def make_sent_bert_features(text):
    emb = sent_model.encode([text])[0]
    return emb


def get_data(save_fd):
    """
        Returns one big graph with unconnected graphs with the following:
        - x (Tensor, optional) – Node feature matrix with shape [num_nodes, num_node_features]. (default: None)
        - edge_index (LongTensor, optional) – Graph connectivity in COO format with shape [2, num_edges]. (default: None)
        - edge_attr (Tensor, optional) – Edge feature matrix with shape [num_edges, num_edge_features]. (default: None)
        - y (Tensor, optional) – Graph or node targets with arbitrary shape. (default: None)
        - validation mask, training mask and testing mask
    """
    data_fd = '../dataset/sroie-2019'
    path = '../dataset/sroie-2019/raw/box'
    files = [i.split('.')[0] for i in os.listdir(path)]
    files.sort()
    all_files = files[1:]

    list_of_graphs = []
    train_list_of_graphs, test_list_of_graphs = [], []

    files = all_files.copy()
    random.shuffle(files)

    # Get 550 receipts for training
    training, testing = files[:550], files[550:]

    for file in tqdm(all_files):
        connect = Grapher(file, data_fd)
        G, _, _ = connect.graph_formation()
        df = connect.relative_distance()
        invidual_data = from_networkx(G)
        df = df.drop('index', axis=1)

        feature_cols = ['rd_b', 'rd_r', 'rd_t', 'rd_l', 'line_number',
                        'n_upper', 'n_alpha', 'n_spaces', 'n_numeric', 'n_special']

        text_features = np.array(df['Object'].map(make_sent_bert_features)).tolist()
        numeric_features = df[feature_cols].values.astype(np.float32)

        features = np.concatenate((numeric_features, text_features), axis=1)
        features = torch.tensor(features)

        for col in df.columns:
            try:
                df[col] = df[col].str.strip()
            except AttributeError as e:
                pass

        df['labels'] = df['labels'].fillna('undefined')
        df.loc[df['labels'] == 'company', 'num_labels'] = 1
        df.loc[df['labels'] == 'address', 'num_labels'] = 2
        df.loc[df['labels'] == 'date', 'num_labels'] = 3
        df.loc[df['labels'] == 'total', 'num_labels'] = 4
        df.loc[df['labels'] == 'undefined', 'num_labels'] = 5

        # adding to test
        df.loc[df['labels'] == 'invoice', 'num_labels'] = 5

        assert df['num_labels'].isnull().values.any() == False, \
            f'labeling error! Invalid label(s) present in {file}.csv'
        labels = torch.tensor(df['num_labels'].values.astype(np.int64))
        text = df['Object'].values

        invidual_data.x = features
        invidual_data.y = labels
        invidual_data.text = text
        invidual_data.img_id = file

        if file in training:
            train_list_of_graphs.append(invidual_data)
        elif file in testing:
            test_list_of_graphs.append(invidual_data)

    train_data = torch_geometric.data.Batch.from_data_list(train_list_of_graphs)
    train_data.edge_attr = None
    test_data = torch_geometric.data.Batch.from_data_list(test_list_of_graphs)
    test_data.edge_attr = None

    torch.save(train_data, os.path.join(save_fd, 'train_data.dataset'))
    torch.save(test_data, os.path.join(save_fd, 'test_data.dataset'))


get_data(save_fd='../dataset/sroie-2019/processed')

