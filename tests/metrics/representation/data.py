import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split


queries = [[0.3281, 0.3934, 0.3079, 0.3238],
           [0.0344, 0.8396, 0.1414, 0.7388],
           [0.587 , 0.1184, 0.1509, 0.3035]]

database = [[0.0256, 0.266 , 0.5239, 0.0042],
            [0.0513, 0.0375, 0.0321, 0.8175],
            [0.8611, 0.8352, 0.3209, 0.8839],
            [0.8433, 0.3853, 0.3332, 0.5728],
            [0.6903, 0.2962, 0.7524, 0.0826],
            [0.1255, 0.0154, 0.8745, 0.2216]]

# Query number to its distances =  1 - cosine_distance
# {
#     0: [0.3, 0.5, 0, 0.1, 0.2, 0.4],
#     1: [0.5, 0.3, 0.2, 0.4, 0.6, 0.7],
#     2: [0.7, 0.5, 0.1, 0, 0.2, 0.6],
# }

# Create relevant
# {
#     0: [4], # third 
#     1: [1, 0], # second and fourth
#     2: [3, 2, 5], # first, second and fourth
# }

# So Data would be sum of query and database list, do some shuffle.
# Shuffle must be done with condition that queries must be at first place by targets.
VECTORS = torch.tensor([
    queries[0], 
    database[4], 
    queries[1],
    queries[2],
    database[0],
    database[2],
    database[3],
    database[5],
    database[1]
])

# Create labels for Classification Dataset
TARGETS = torch.tensor([0, 0, 1, 2, 1, 2, 2, 2, 1])

# Need create queries_idxs and scores for Representation Dataset
QUERIES_IDX = torch.tensor([0, -1, 1, 2, -1, -1, -1, -1, -1])

SCORES = torch.tensor(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 2, 0],
        [0, 0, 1],
        [0, 0, 2],
        [0, 0, 4],
        [0, 4, 0],
    ]
)

# query 1 
# - relevant = [0]
# - closest  = [2, 3, 0, 4, 5, 1]

# query 2
# - relevant = [5, 1]
# - closest = [2, 3, 5, 0, 4, 1]

# query 3
# - relevant = [4, 3, 2]
# - closest = [2, 3, 0, 5, 4, 1]

# Precision
# k_1 = (0 + 0 + 1) / 3 = 1/3
# k_2 = (0 + 0 + 1) / 3 = 1/3
# k_3 = (1/3 + 1/3 + 2/3) / 3 = 4/9
# k_4 = (1/4 + 1/4 + 2/4) / 3 = 1/3
# k_5 = (1/5 + 2/5 + 2/5) / 3 = 1/3
# k_6 = (1/6 + 2/6 + 3/6) / 3 = 1/3

# Recall
# k_1 = (0 + 0 + 1/3) / 3 = 1/9
# k_2 = (0 + 0 + 2/3) / 3 = 2/9
# k_3 = (1 + 1/2 + 2/3) / 3 = 13/18
# k_4 = (1 + 1/2 + 2/3) / 3 = 13/18
# k_5 = (1 + 1/2 + 1) / 3 = 5/6
# k_6 = 1

# Average Precision
# k_1 = (0 + 0 + 1/3) / 3 = 1/9
# k_2 = (0 + 0 + 2/3) / 3 = 2/9
# k_3 = (1/3 + 1/6 + 2/3) / 3 = 7/18
# k_4 = (1/3 + 1/6 + 2/3) / 3 = 7/18
# k_5 = (1/3 + 1/6 + (2 + 3/5)/ 3) / 3 = (1/2 + 13/15) / 3 = 41/90
# k_6 = (1/3 + (1/3 + 2/6)/ 2 + 13/15) / 3 = (10/30 + 11/30 + 26/30) / 3 = 46/90

# NDCG For classification for classification it always be DCG because all score is 1
# DCG@k = sum_k score[k] / log_2(k + 1)
# k_1 = (0 + 0 + 0.25) / 3 = 0.08333333
# k_2 = (0 + 0 + 0.42985935) / 3 = 0.14328645
# k_3 = (0.5 + 0.38009377 + 0.39255721) / 3 = 0.42421699
# k_4 = (0.5 + 0.38009377 + 0.39255721) / 3 = 0.42421699
# k_5 = (0.5 + 0.38009377 + 0.6611183) / 3 = 0.51373735
# k_6 = (0.5 + 0.5154859 + 0.6611183) / 3 = 0.55886806

# 1-6 is top-k for retrieval
ANSWERS = {
    'precision': {
        1: 1/3,
        2: 1/3,
        3: 4/9,
        4: 1/3,
        5: 1/3,
        6: 1/3
    },
    'recall': {
        1: 1/9, 
        2: 2/9, 
        3: 13/18,
        4: 13/18,
        5: 5/6,
        6: 1
    },
    'average_precision': {
        1: 1/9,
        2: 2/9,
        3: 7/18,
        4: 7/18,
        5: 41/90,
        6: 46/90
    },
    'ndcg': {
        1: 0.08333333,
        2: 0.14328645,
        3: 0.42421699,
        4: 0.42421699,
        5: 0.51373735,
        6: 0.55886806
    }
}
