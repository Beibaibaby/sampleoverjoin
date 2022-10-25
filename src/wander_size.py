import random
import re
from acyclic_join import *
from build_hash import *
import scipy.stats as sps
import math
import pandas as pd
import numpy as np
import pickle
from equi_chain_overlap import *

def random_walk(j, hs):
    """
    :param j: join j
    :param hs: hash table for join keys
    :return: joined tuple and probability
    """ 
    pri_keys = ['NationKey', 'S_SuppKey', 'CustKey', 'OrderKey', 'LineNumber']
    ts = []
    t = j.tables[0].sample(n=1)
    ts.append(t)
    p = 1 / j.tables[0].shape[0]
    for table_index in range(1, len(j.tables)):
        bucket = hs[table_index-1][ts[table_index-1][j.keys[table_index-1]].values[0]]
        if len(bucket) == 0:
            return t, 0
        else:
            temp_t_index = random.sample(bucket, 1)
            # temp_t = j.tables[table_index].iloc[temp_t_index]
            # print(temp_t_index)
            # print(j.tables[table_index][pri_keys[table_index]])
            temp_t = j.tables[table_index].loc[j.tables[table_index][pri_keys[table_index]] == temp_t_index[0]]
            # df['column_name'] == some_value
            ts.append(temp_t)
            # print(t)
            # print(temp_t)
            t = pd.merge(t, temp_t, on = j.keys[table_index-1], how = 'inner')
            # print(t)
            p *= 1 / len(bucket)
            # print(t)
    return t, p


# def random_walk(j, hs):
#     """
#     :param j: join j
#     :param hs: hash table for join keys
#     :return: joined tuple and probability
#     """ 
#     ts = []
#     t = j.tables[0].sample(n=1)
#     ts.append(t)
#     p = 1 / j.tables[0].shape[0]
#     for table_index in range(1, len(j.tables)):
#         bucket = hs[table_index-1][ts[table_index-1][j.keys[table_index-1]].values[0]]
#         if len(bucket) == 0:
#             return t, 0
#         else:
#             temp_t_index = random.sample(bucket, 1)
#             temp_t = j.tables[table_index].iloc[temp_t_index]
#             ts.append(temp_t)
#             # print(t)
#             # print(temp_t)
#             t = pd.merge(t, temp_t, on = j.keys[table_index-1], how = 'inner')
#             # print(t)
#             p *= 1 / len(bucket)
#             # print(t)
#     return t, p


def calc_eps(sigma, z_alpha, recip_ps):
    """
    :param alpha: prompt by the user
    :param ps: list of probabilities for sampled tuples
    :return: confidence interval epsilon
    """ 
    m = len(recip_ps)
    epsilon = z_alpha * sigma / math.sqrt(m)
    return epsilon


def calc_pr(epsilon, e_size, recip_ps):
    count_true = 0
    count_false = 0
    for estimator in recip_ps:
        if abs(estimator - e_size) <= epsilon:
            count_true += 1
        else:
            count_false += 1
    # print("count true: ", count_true)
    # print("count false: ", count_false)
    return count_true / (count_true + count_false)


def wander_e_size(alpha, join, hs):
    eps = 0
    ts = []
    ps = []
    # while (True):
    max_pr = 0
    max_accu_size = 0
    for i in range(1000):
    # while(True):
        t,p = random_walk(join, hs)
        ts.append(t)
        ps.append(p)
        
        if np.sum(ps) > 0:
            # print(np.sum(ps))
            recip_ps = []
            for p in ps:
                if p == 0:
                    recip_ps.append(0)
                else:
                    recip_ps.append(1 / p)

            e_size = np.mean(recip_ps)
            # print("Estimated size: ", e_size)

            sigma = np.std(recip_ps)
            # print("sigma: ", sigma)
            # z_alpha  = sps.norm.ppf((alpha + 1) / 2, loc=e_size, scale=sigma)
            z_alpha  = sps.norm.ppf((alpha + 1) / 2, loc=0, scale=1)
            # print("z_alpha", z_alpha)

            eps = calc_eps(sigma, z_alpha, recip_ps)
            # print("eps: ", eps) 

            pr = calc_pr(eps, e_size, recip_ps)
            # print("pr: ", pr)

            if pr > max_pr and pr < 1:
                max_accu_size = e_size
                max_pr = pr

            # print("pr: ", max_pr)

            if pr > alpha and pr < 1:
                break
    print(max_accu_size)
    return max_accu_size, max_pr


def main():
    # supplier_sample_1,nation_sample_1,customer_sample_1,orders_sample_1,lineitem_sample_1 = process_tpch(0.05, 0.05)

    # nation_sample_1.to_csv(r'./tpch_wander_test/n1.csv')
    # supplier_sample_1.to_csv(r'./tpch_wander_test/s1.csv')
    # customer_sample_1.to_csv(r'./tpch_wander_test/c1.csv')
    # orders_sample_1.to_csv(r'./tpch_wander_test/o1.csv')
    # lineitem_sample_1.to_csv(r'./tpch_wander_test/l1.csv')

    nation_sample_1 = pd.read_csv('./tpch_wander_test/n1.csv' ,index_col=0)
    supplier_sample_1 = pd.read_csv('./tpch_wander_test/s1.csv' ,index_col=0)
    customer_sample_1 = pd.read_csv('./tpch_wander_test/c1.csv' ,index_col=0)
    orders_sample_1 = pd.read_csv('./tpch_wander_test/o1.csv' ,index_col=0)
    lineitem_sample_1 = pd.read_csv('./tpch_wander_test/l1.csv' ,index_col=0)
    tables_1 = [supplier_sample_1, nation_sample_1, customer_sample_1, orders_sample_1,lineitem_sample_1]
    keys = ['NationKey', 'NationKey', 'CustKey', 'OrderKey']

    join = chain_join(tables_1, keys)
    print("Join created")

    # join_f = join.f_join()
    # print(join_f.shape[0])
    
    hs = hash_j(join)
    print("Hash success")

    f = open("./tpch_wander_test/hs.pkl","wb")
    pickle.dump(hs,f)
    f.close()

    # hs  = pickle.load(open("./tpch_wander_test/hs1.pkl", "rb"))
    # print("hash successfully loaded")

    alpha = 0.9
    
    max_accu_size, max_pr = wander_e_size(alpha, join, hs)
    print("size: ", max_accu_size)
    print("confidence level:", max_pr)

if __name__ == '__main__':
    main()
