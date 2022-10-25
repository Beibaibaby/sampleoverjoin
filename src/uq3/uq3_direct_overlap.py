import json
import pandas as pd
# from equi_chain_overlap import *
from acyclic_join import *
from build_hash import *
import pickle
from sample_from_disjoint import *
from sample_union_bernoulli import *
from uq3_preprocess import *

def e_size(j):
    uniq_1 = j.tables[0][j.keys[0]].unique()
    uniq_2 = j.tables[1][j.keys[0]].unique()
    uniq = uniq_1[np.in1d(uniq_1,uniq_2)]
    key_1 = j.keys[0]
    size = 0
    for v in uniq:
        size += (j.tables[0][j.tables[0][key_1] == v].shape[0] *
            j.tables[1][j.tables[1][key_1] == v].shape[0])
    for i in range(1,len(j.keys)):
        size *= max_d(j.tables[i+1], j.keys[i])
    return size


# return max degree
def max_d(table, attribute):
    return table[attribute].value_counts().max()


# return table size
def t_size(table):
    return table.shape[0]


def gen_os(js):
    #find intersection of values in first join attribute in all tables

    ans = p_set(list(range(len(js))))
    
    Os = []
    for subset in ans:
        Os.append(gen_o(js,subset))
    
    return ans, Os

def get_int_values(js, index, subset):
    uniq = js[subset[0]].tables[index][js[subset[0]].keys[index]].unique()
    for i in subset:
        uniq_1 = js[i].tables[index][js[i].keys[index]].unique()
        uniq_2 = js[i].tables[index+1][js[i].keys[index]].unique()
        uniq_1_2 = uniq_1[np.in1d(uniq_1,uniq_2)]
        uniq = uniq[np.in1d(uniq,uniq_1_2)]
    return uniq


# generate powerset
def p_set(nums):
    ans_all = [[]]

    for n in nums:
        ans_all += [a+[n] for a in ans_all]
        
    ans = []
    for i in ans_all:
        if len(i) > 1: 
            ans.append(i)
        
    return ans

def gen_o(js,subset):
    key_1 = js[0].keys[0]
    K = 0
    uniq = get_int_values(js, 0, subset)

    sizes = []
    for v in uniq:
        for i in subset:
            sizes.append(js[i].tables[0][js[i].tables[0][key_1] == v].shape[0] *
            js[i].tables[1][js[i].tables[1][key_1] == v].shape[0])
        K += min(sizes)

    for k in range(1,len(js[0].keys)):
        max_ds = []
        uniq = get_int_values(js, k, subset)
        for i in subset:
            if (js[i].join_type[k] == False):
                max_ds.append(1)
            else:
                max_ds.append(max_d_in_set(js[i].tables[k+1], js[i].keys[k], uniq))
        K *= min(max_ds)
    
    return K

def max_d_in_set(table, attribute, value_set):
    values = table[attribute].value_counts(dropna=False).keys().tolist()
    counts = table[attribute].value_counts(dropna=False).tolist()
    value_dict = dict(zip(values, counts))
    for v in values:
        if v in value_set:
            return value_dict[v]

def exact_olp(f_js):
    ans = p_set(list(range(len(f_js))))
    
    Os = []
    for subset in ans:
        frames = []
        for i in range(len(subset)):
            frames.append(f_js[subset[i]])
        disjoint = pd.concat(frames)
        inter = disjoint.value_counts()[disjoint.value_counts() == len(subset)]
        Os.append(inter.shape[0])
    
    return Os

def calc_As(js, Os, ans):
    n = len(js)
    As = [ [0]*n for i in range(n)]
    for j in range(len(js)):
        As[j][n-1] = Os[len(Os)-1]
        for k in range(n-1, 0, -1):
            A = 0 
            count = 0
            for index in range(len(ans)):
                if (len(ans[index]) == k) and (j in ans[index]):
                    A += Os[index]
                    count += 1
            if (k == 1): A += e_size(js[j])
            # Calculate A
            for r in range(k+1, n+1):
                A -= (math.comb(r-1, k-1) * As[j][r-1])
            As[j][k-1] = A
    return As

def uq3_calc_U(js, norm_js):

    ans, Os = gen_os(norm_js)
    As = calc_As(js, Os, ans)
    U = 0

    As_T = np.array(As).T.tolist()
    for k in range(len(As)):
        U += (1/(k+1) * np.sum(As_T[k]))
    return U