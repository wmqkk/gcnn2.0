import numpy as np
# import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle as pkl

import geopandas as gpd

from geopandas import GeoSeries
from shapely.geometry import LineString
import scipy.sparse as sp
import pandas as pd

from scipy.sparse.linalg.eigen.arpack import eigsh

# import networkx as nx

seed = 123
np.random.seed(seed)


def distance(p1, p2):
    return ((p1.X - p2.X) ** 2 + (p1.Y - p2.Y) ** 2) ** 0.5


def angle(p1, p2):
    dx = p1.X - p2.X
    dy = p1.Y - p2.Y
    if dx == 0:
        return 90
    else:
        return np.arctan(dy / dx) / np.pi * 180


class Point:
    def __init__(self, x, y):
        self.X = x
        self.Y = y


def input_data(num, path):
    all_data = []
    all_adj = []
    max_nums = 0
    for i in range(num):
        df_shape = gpd.read_file(path + str(i) + '.shp', encode='utf-8')
        lst = df_shape['geometry']
        point = []
        center = Point(0, 0)
        all_long = 0
        _distance = []
        _direction = []
        _angle = []
        if (lst.shape[0] > max_nums):
            max_nums = lst.shape[0]
        for j in range(lst.shape[0]):
            try:
                line = np.array(lst.iat[j].xy).T
            except:
                print(i, j)
            point.append((Point(line[0][0], line[0][1]), Point(line[1][0], line[1][1])))
            long = distance(Point(line[0][0], line[0][1]), Point(line[1][0], line[1][1]))
            center.X += (line[0][0] + line[1][0]) * long
            center.Y += (line[0][1] + line[1][1]) * long
            all_long += long
        center.X = 0.5 * center.X / all_long
        center.Y = 0.5 * center.Y / all_long
        shape = lst.shape[0]
        adj = np.zeros(shape=(shape, shape))
        for j in range(lst.shape[0]):
            _direction.append(angle(point[j][0], point[j][1]))
            _distance.append(0.5 * (distance(point[j][0], center) + distance(point[j][1], center)))
            _angle.append(angle(point[j][0], center))
            for k in range(j + 1, lst.shape[0]):
                if ((point[k][0].X == point[j][0].X and point[k][0].Y == point[j][0].Y)
                        or (point[k][0].X == point[j][1].X and point[k][0].Y == point[j][1].Y)
                        or (point[k][1].X == point[j][0].X and point[k][1].Y == point[j][0].Y)
                        or (point[k][1].X == point[j][1].X and point[k][1].Y == point[j][1].Y)):
                    adj[k][j] += 1
                    adj[j][k] += 1
        _degree = np.sum(adj, axis=0)
        scaler = preprocessing.StandardScaler()
        _degree = scaler.fit_transform(_degree.reshape(-1, 1))
        _distance = scaler.fit_transform(np.array(_distance).reshape(-1, 1))
        _direction = scaler.fit_transform(np.array(_direction).reshape(-1, 1))
        _angle = scaler.fit_transform(np.array(_angle).reshape(-1, 1))
        data = np.array([_degree, _distance, _direction, _angle]).T
        data = data.reshape(data.shape[1], data.shape[2])
        all_data.append(data)
        all_adj.append(adj)
    return all_data, all_adj, max_nums


def input_all_data(num, chebyshev_p):
    all_tree, tree_adj, tree_nums = input_data(num, './data/irregular/merge/')
    all_comb, comb_adj, comb_nums = input_data(num, './data/radial/merge/')
    all_road, road_adj, road_nums = input_data(num, './data/grid/merge/')
    max_nums = max(road_nums, tree_nums, comb_nums)
    all_data = all_road + all_tree + all_comb
    all_adj = road_adj + tree_adj + comb_adj
    for i in range(len(all_data)):
        shape = all_data[i].shape
        all_data[i] = np.pad(all_data[i], pad_width=((0, max_nums - shape[0]), (0, 0)), mode='constant')
        if (i < num):
            all_data[i] = [all_data[i], np.array([0, 1, 0])]
        elif (i < 2 * num):
            all_data[i] = [all_data[i], np.array([0, 0, 1])]
        else:
            all_data[i] = [all_data[i], np.array([1, 0, 0])]
        all_adj[i] = np.pad(all_adj[i], pad_width=((0, max_nums - shape[0]), (0, max_nums - shape[0])))
        all_data[i].append(chebyshev_polynomials(all_adj[i], chebyshev_p))
        # all_data[i].append([preprocess_adj(all_adj[i])])
        all_data[i].append(i)
    train_X, test_X = train_test_split(all_data, test_size=0.3)

    test_index = []
    feature = []
    train_y = []
    test_feature = []
    test_y = []
    train_support = []
    test_support = []
    for i in range(len(train_X)):
        feature.append(train_X[i][0])
        train_y.append(train_X[i][1])
        train_support.append(train_X[i][2])
    for i in range(len(test_X)):
        test_feature.append(test_X[i][0])
        test_y.append(test_X[i][1])
        test_support.append(test_X[i][2])
        test_index.append(test_X[i][3])
    pd.DataFrame(test_index).to_csv('test.csv')
    return feature, train_y, train_support, test_feature, test_y, test_support, test_index


def generate_sz():
    df_shape = gpd.read_file('./data/szs2s.shp', encode='utf-8')
    lst = df_shape['geometry']
    point = []
    center = Point(0, 0)
    all_long = 0
    _distance = []
    _direction = []
    _angle = []
    labels = []
    for grid in df_shape['grid']:
        labels.append(grid)
    for j in range(lst.shape[0]):
        try:
            line = np.array(lst.iat[j].xy).T
        except:
            print('wrong')
        point.append((Point(line[0][0], line[0][1]), Point(line[1][0], line[1][1])))
        long = distance(Point(line[0][0], line[0][1]), Point(line[1][0], line[1][1]))
        center.X += (line[0][0] + line[1][0]) * long
        center.Y += (line[0][1] + line[1][1]) * long
        all_long += long
    center.X = 0.5 * center.X / all_long
    center.Y = 0.5 * center.Y / all_long
    shape = lst.shape[0]
    col = []
    row = []
    adj_data = []
    for j in range(lst.shape[0]):
        _direction.append(angle(point[j][0], point[j][1]))
        _distance.append(0.5 * (distance(point[j][0], center) + distance(point[j][1], center)))
        _angle.append(angle(point[j][0], center))
        for k in range(j + 1, lst.shape[0]):
            if ((point[k][0].X == point[j][0].X and point[k][0].Y == point[j][0].Y)
                    or (point[k][0].X == point[j][1].X and point[k][0].Y == point[j][1].Y)
                    or (point[k][1].X == point[j][0].X and point[k][1].Y == point[j][0].Y)
                    or (point[k][1].X == point[j][1].X and point[k][1].Y == point[j][1].Y)):
                col.append(k)
                row.append(j)
                adj_data.append(1)
                col.append(j)
                row.append(k)
                adj_data.append(1)
    adj = sp.coo_matrix((adj_data, (row, col)), shape=(shape, shape))
    _degree = np.sum(adj, axis=0)
    scaler = preprocessing.StandardScaler()
    _degree = scaler.fit_transform(_degree.reshape(-1, 1))
    _distance = scaler.fit_transform(np.array(_distance).reshape(-1, 1))
    _direction = scaler.fit_transform(np.array(_direction).reshape(-1, 1))
    _angle = scaler.fit_transform(np.array(_angle).reshape(-1, 1))
    data = np.array([_degree, _distance, _direction, _angle]).T
    data = data.reshape(data.shape[1], data.shape[2])
    filename = 'ssz.inx'
    with open(filename, 'w') as file_object:
        for i in col:
            file_object.write(i.__str__() + '\n' )
    filename = 'ssz.iny'
    with open(filename, 'w') as file_object:
        for i in row:
            file_object.write(i.__str__() + '\n')
    filename = 'ssz.graph'
    with open(filename, 'w') as file_object:
        for i in adj_data:
            file_object.write(i.__str__() + '\n')
    filename = 'ssz.data'
    with open(filename, 'w') as file_object:
        for item in data:
            file_object.write(' '.join([str(i) for i in item]))
            file_object.write('\n')
    filename = 'ssz.label'
    with open(filename, 'w') as file_object:
        for i in labels:
            file_object.write(i.__str__() + '\n')



def input_sz():
    names = ['inx', 'iny', 'graph', 'label']
    object = []
    for i in range(len(names)):
        with open("./data/ssz.{}".format(names[i]), 'r') as f:
            temp = f.read().split('\n')
            temp.pop()
            temp = [int(i) for i in temp]
            object.append(temp)
    x, y, graph, label = tuple(object)

    with open("./data/ssz.data") as f:
        data = f.read().split('\n')
        data.pop()
        for i in range(len(data)):
            data[i] = [float(i) for i in data[i].split(' ')]

    shape = len(data)
    adj = sp.coo_matrix((graph,(x,y)),shape=(shape, shape))
    idx_train, idx_test = train_test_split(list(range(shape)), test_size=0.95)
    labels = preprocessing.OneHotEncoder(sparse=False).fit_transform(np.array(label).reshape(-1,1))
    train_mask = sample_mask(idx_train, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    y_train = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    return adj, data, y_train, y_test, train_mask, test_mask, labels



def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        cords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return cords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1), dtype='float64')
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    a = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return a


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    # print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        # s_lap = scaled_lap
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

# generate_sz()