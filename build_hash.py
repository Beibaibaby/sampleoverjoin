from collections import defaultdict
from acyclic_join import *
import pickle

# def hash_j(j):
#     n = len(j.tables)
#     hs = [defaultdict(list) for _ in range(n-1)]
#     for i in range(n-1):
#         print(i)
#         for index, row in j.tables[i].iterrows():
#             key = row[j.keys[i]]
#             # print(key)
#             if len(hs[i][key]) == 0:
#                 for index_1, row_1 in j.tables[i+1].iterrows():
#                     if row_1[j.keys[i]] == key:
#                         hs[i][key].append(index_1)
#     return hs


def hash_j(j):
    n = len(j.tables)
    hs = [defaultdict(list) for _ in range(n-1)]
    for i in range(1,n):
        print(i)
        for index, row in j.tables[i].iterrows():
            # print(index)
            key = row[j.keys[i-1]]
            hs[i-1][key].append(index)
    return hs

def hash_j_pri(j):
    pri_keys = ['NationKey', 'S_SuppKey', 'CustKey', 'OrderKey', 'LineNumber']
    n = len(j.tables)
    hs = [defaultdict(list) for _ in range(n-1)]
    for i in range(1,n):
        print(i)
        for index, row in j.tables[i].iterrows():
            # print(index)
            key = row[j.keys[i-1]]
            pri = row[pri_keys[i]]
            hs[i-1][key].append(pri)
    return hs


# def main():
#     hs_1 = pickle.load(open("./iidjoin/acyclic_3/q1_hs.pkl", "rb"))
#     print(hs_1[3])


# if __name__ == '__main__':
#     main()