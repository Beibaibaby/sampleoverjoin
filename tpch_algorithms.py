import pandas as pd
import numpy as np
import random
from collections import defaultdict
import time

def fixed_sample(table, fixed, scale):

    table_fixed = table.iloc[:int(table.shape[0] * fixed), :]

    sample_frac = (scale - fixed) / (1 - fixed)
    table_sample = table.iloc[int(table.shape[0] * fixed):, :].sample(frac=sample_frac, replace=False)

    frames = [table_fixed, table_sample]
    sample = pd.concat(frames)

    return sample

def process_tpch(fixed, scale):

    # Qx: nation, supplier, customer, orders, lineitem
    nation = pd.read_table('./tpch_1/nation.tbl', index_col=False, names=['NationKey','NationName','RegionKey','Comment'], delimiter = '|').iloc[:, :-1]
    supplier = pd.read_table('./tpch_1/supplier.tbl', index_col=False, names=['SuppKey','SuppName','Address','NationKey','Phone','Acctbl','Comment'], delimiter = '|').iloc[:, :-1]
    customer = pd.read_table('./tpch_1/customer.tbl', index_col=False, names=['CustKey','CustName','Address','NationKey','Phone','Acctbal','MktSegment','Comment'], delimiter = '|').iloc[:, :-1]
    orders = pd.read_table('./tpch_1/orders.tbl', index_col=False, names=['OrderKey','CustKey','OrderStatus','TotalPrice','OrderDate','OrderPriority','Clerk','ShipPriority','Comment'], delimiter = '|').iloc[:, :-1]
    lineitem = pd.read_table('./tpch_1/lineitem.tbl', index_col=False, names=['OrderKey','PartKey','SuppKey','LineNumber','Quantity','ExtendedPrice','Discount','Tax','ReturnFlag',
    'LineStatus','ShipDate','CommitDate','ReceiptDate','ShipinStruct','ShipMode','Comment'], delimiter = '|').iloc[:, :-1]

    # Change rows to random order
    nation.sample(frac=1)
    supplier.sample(frac=1)
    customer.sample(frac=1)
    orders.sample(frac=1)
    lineitem.sample(frac=1)

    # should adjust percentage and scale according to table size
    nation_sample = fixed_sample(nation, fixed, scale)
    supplier_sample = fixed_sample(supplier, fixed, scale)
    customer_sample = fixed_sample(customer, fixed, scale)
    orders_sample = fixed_sample(orders, fixed, scale)
    lineitem_sample = fixed_sample(lineitem, fixed, scale)

    return nation_sample,supplier_sample,customer_sample,orders_sample,lineitem_sample


class chain_join:
    def __init__(self, tables, keys):
        self.tables = tables
        self.keys = keys

    def f_join(self):
        result = self.tables[0]
        for i in range(1,len(self.tables)):
            # print(len(result))
            result = pd.merge(result, self.tables[i], on = self.keys[i-1], how = 'inner')
        return result

    # extended olken's |R_1|* prod(M)
    def e_size(self):
        size = t_size(self.tables[0])
        for i in range(1,len(self.tables)):
            size *= max_d(self.tables[i], self.keys[i-1])
        return size


# return max degree
def max_d(table, attribute):
    return table[attribute].value_counts().max()


# return table size
def t_size(table):
    return table.shape[0]


def exact_olp(j1, j2):
    # print(len(intersection(join_1_f, join_2_f)))
    inter = pd.merge(j1, j2, how = 'inner').drop_duplicates()
    # print(len(join_1_f.intersection(join_2_f)))
    return inter.shape[0]


def e_o_olken(j1, j2): 
    K = 0 
    N_1_1 = j1.tables[0].shape[0] * max_d(j1.tables[1], j1.keys[0])
    N_2_1 = j2.tables[0].shape[0] * max_d(j2.tables[1], j2.keys[0])
    K = min(N_1_1, N_2_1)
    # print(K)
    for i in range(1,len(j1.keys)):
        K *= min(max_d(j1.tables[i+1], j1.keys[i]), max_d(j2.tables[i+1], j2.keys[i]))
    return K


def e_o_value(j1, j2):  
    K = 0

    uniq_1_1 = j1.tables[0][j1.keys[0]].unique()
    uniq_1_2 = j1.tables[1][j1.keys[0]].unique()
    uniq_1 = uniq_1_1[np.in1d(uniq_1_1,uniq_1_2)]
    # print(uniq_1)
    uniq_2_1 = j2.tables[0][j1.keys[0]].unique()
    uniq_2_2 = j2.tables[1][j1.keys[0]].unique()
    uniq_2 = uniq_2_1[np.in1d(uniq_2_1,uniq_2_2)]
    # print(uniq_2)
    uniq = uniq_1[np.in1d(uniq_1,uniq_2)]
    # print(uniq)

    key_1 = j1.keys[0]
    
    for v in uniq:
        d_1_1 = j1.tables[0][j1.tables[0][key_1] == v].shape[0]
        # print(d_1_1)
        d_1_2 = j1.tables[1][j1.tables[1][key_1] == v].shape[0]
        # print(d_1_2)
        d_2_1 = j2.tables[0][j2.tables[0][key_1] == v].shape[0]
        # print(d_2_1)
        d_2_2 = j2.tables[1][j2.tables[1][key_1] == v].shape[0]
        # print(d_2_2)

        K += min(d_1_1*d_1_2, d_2_1*d_2_2)
    # print(K)
    for i in range(1,len(j1.keys)):
        K *= min(max_d(j1.tables[i+1], j1.keys[i]), max_d(j2.tables[i+1], j2.keys[i]))
    return K


def A_i_1(joins, idx, O):
    m = np.max(O[idx, :])
    return joins[idx].e_size() - m


# one-to-one join
def hash_t(table, keys, k):
    h_t = [[]] * k
    h_t = defaultdict(list)
    for i, row in table.iterrows():
        # print("key:", row[keys])
        # print(table.iloc[[i]].columns.tolist())
        # print("i: ", i)
        h_t[row[keys] % k].append(table.loc[[i]])
    return h_t

def hash_j(join, k):
    h_t_1 = hash_t(join.tables[0], join.keys[0], k) 
    h_j = [None] * len(join.tables)
    # print(h_j)
    h_j[0] = h_t_1
    for i in range(1,len(join.tables)):
        print("iter in hash ", i)
        h_j[i] = hash_t(join.tables[i], join.keys[i-1], k)
    return h_j

def hash_s(t, keys, k):
    return t[keys].values[0] % k

def hash_pre(joins, k):
    h_js = [None] * len(joins)
    for i in range(0, len(joins)):
        h_js[i] = hash_j(joins[i], k)
    return h_js


# sample from single join
def sample_from_s_join(join, h_j, k):
    count = 0
    product = []

    while count < len(h_j):
        if count == 0:
            l = list(h_j[0])
            bucket = random.sample(l, 1)[0]
            t = random.sample(h_j[0][bucket], 1)[0]
            product.append(len(h_j[0][bucket]))
            count += 1
        else:
            key_attribute = join.keys[count-1]
            key_value = hash_s(t, key_attribute, k)
            s = random.sample(h_j[count][key_value], 1)[0]
            if s[key_attribute].values[0] == t[key_attribute].values[0]:
                t = pd.merge(t, s, on = key_attribute, how = 'inner')
                d_s = (join.tables[count][key_attribute] == (s[key_attribute].values[0])).sum()
                product.append(d_s)
                count += 1
            else:
                product.clear()
                count = 0

    return t, np.prod(product)


# sample from union of joins
def sample_from_u_join(joins, h_js, n, k, sample_start):
    time_store = []

    S = pd.DataFrame()
    N = len(joins)

    # store join size
    J = np.zeros(N)
    # store overlapping size
    O = np.zeros((N,N))

    for i in range(0, N-1):
        J[i] = joins[i].e_size()
        for j in range (i+1, N):
            O[i][j] = e_o_value(joins[i], joins[j])
    J[N-1] = joins[N-1].e_size()

    # store A_i^1
    A = np.zeros(N)

    for i in range(0, N):
        A[i] = A_i_1(joins, i, O)
    
    # probability of choosing J_i
    P = np.zeros(N)
    for i in range(0, N):
        P[i] = (2 * A[i] + np.sum(O, axis=0)[i]) / (2 * np.sum(A) + np.sum(O))

    C = np.sum(J)

    while len(S) < n:
        i = np.random.choice(np.arange(0, N), p = P)
        r_j = random.random()
        p_j = (J[i] * (2 * np.sum(A) + np.sum(O))) / (C * (2 * A[i] + np.sum(O, axis=0)[i]))
        if r_j <= p_j:
            t, prod_bkts = sample_from_s_join(joins[i], h_js[i], k)
            p_t = k * prod_bkts / J[i]
            r_t = random.random()
            if r_t <= p_t:
                S = S.append(t)
                print(len(S))
                if(len(S) % 10 == 0):
                    cur_time = time.perf_counter()
                    time_store.append(cur_time - sample_start)
            else:
                continue
        else:
            continue
    
    return S, time_store



# baseline
def sample_from_join(join_u, n):
    return join_u.sample(n)


def main():
    import argparse
    # import os

    parser = argparse.ArgumentParser(description='Random sampling over union of joins')
    parser.add_argument('--scale', type=float, default=0.3, help='less than 1; scale of the 1g dataset')
    parser.add_argument('--overlap', type=float, default=0.1, help='percentage of overlapping data between joins')
    parser.add_argument('--n', type=int, default=1000, help='number of target samples')
    parser.add_argument('--k', type=int, default=10000, help='Number of buckets after hashing')
    parser.add_argument('--value_base', action='store_true', default=True, help='If provided, use value based alg to calculate overlap; otherwise use extended olken')
    args = parser.parse_args()

    scale = args.scale
    overlap = args.overlap

    n = args.n
    k = args.k

    nation_sample_1,supplier_sample_1,customer_sample_1,orders_sample_1,lineitem_sample_1 = process_tpch(overlap, scale)
    nation_sample_2,supplier_sample_2,customer_sample_2,orders_sample_2,lineitem_sample_2 = process_tpch(overlap, scale)
    nation_sample_3,supplier_sample_3,customer_sample_3,orders_sample_3,lineitem_sample_3 = process_tpch(overlap, scale)

    tables_1 = [nation_sample_1, supplier_sample_1, customer_sample_1, orders_sample_1,lineitem_sample_1]
    tables_2 = [nation_sample_2, supplier_sample_2, customer_sample_2, orders_sample_2,lineitem_sample_2]
    tables_3 = [nation_sample_3, supplier_sample_3, customer_sample_3, orders_sample_3,lineitem_sample_3]
    keys = ['NationKey', 'NationKey', 'CustKey', 'OrderKey']

    print("step 1 over")


    join_1 = chain_join(tables_1, keys)
    join_2 = chain_join(tables_2, keys)
    join_3 = chain_join(tables_3, keys)

    print("step 2 over")


    # hash
    hash_start = time.perf_counter()
    hashed_joins = hash_pre([join_1, join_2, join_3], k)
    hash_end = time.perf_counter()
    print(f"Hash all tables in {hash_end - hash_start:0.4f} seconds")


    # sampling over union of joins
    sample_start = time.perf_counter()
    online_sample_result, time_sample = sample_from_u_join([join_1, join_2], hashed_joins, n, k, sample_start)
    print(time_sample)

    # baseline
    base_start = time.perf_counter()
    join_1_f = join_1.f_join()
    join_2_f = join_2.f_join()
    join_3_f = join_3.f_join()
    print("step 3 over")
    full_frames = [join_1_f, join_2_f, join_3_f]
    join_f = pd.concat(full_frames)
    base_sample_result = sample_from_join(join_f, n)
    base_end = time.perf_counter()
    print(f"baseline in {base_end - base_start:0.4f} seconds")


if __name__ == '__main__':
    main()
