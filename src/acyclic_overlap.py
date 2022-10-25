import pandas as pd
import numpy as np
import math
# from collections import defaultdict
from acyclic_join import *

# estimate size
def acyc_e_size(j):
    
    root = j.root
    first_child = root.childs[0]
    
    uniq_1 = root.table[root.key].unique()
    uniq_2 = first_child.table[root.key].unique()
    uniq = uniq_1[np.in1d(uniq_1,uniq_2)]

    size = 0
    for v in uniq:
        size += (root.table[root.table[root.key] == v].shape[0] *
            first_child.table[first_child.table[root.key] == v].shape[0])
    
    size *= get_M(first_child, size)

    return size


# get max degree of childs
def get_M(root, size):
    if len(root.childs) == 0:
        return max_d(root.table, root.parent.key)
    else:
        sum = 0
        for child in root.childs:
            sum += size * get_M(child, size)
        return sum
    
    
# return max degree
def max_d(table, attribute):
    return table[attribute].value_counts().max()


# return table size
def t_size(table):
    return table.shape[0]


def acyc_gen_os(tables, Ms, keys):
    #find intersection of values in first join attribute in all tables

    ans = p_set(list(range(len(tables))))
    
    Os = []
    for subset in ans:
        Os.append(acyc_gen_o(tables, Ms, subset, keys))
    
    return ans, Os


def get_int_values(tables, index, subset, keys):
    uniq = tables[subset[0]][index][keys[index]].unique()
    for i in subset:
        uniq_1 = tables[i][index][keys[index]].unique()
        uniq_2 = tables[i][index+1][keys[index]].unique()
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

def acyc_gen_o(tables, Ms, subset, keys):
    key_1 = keys[0]
    K = 0
    uniq = get_int_values(tables, 0, subset, keys)
    
    sizes = []
    for v in uniq:
        for i in subset:
            sizes.append(tables[i][0][tables[i][0][key_1] == v].shape[0] *
            tables[i][1][tables[i][1][key_1] == v].shape[0])
        K += min(sizes)
    
    for m in Ms:
        K *= min(m)
    
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
            if (k == 1): A += acyc_e_size(js[j])
            # Calculate A
            for r in range(k+1, n+1):
                A -= (math.comb(r-1, k-1) * As[j][r-1])
            As[j][k-1] = A
    return As

def calc_U(js, tables, Ms, keys):

    ans, Os = acyc_gen_os(tables, Ms, keys)
    As = calc_As(js, Os, ans)
    U = 0

    As_T = np.array(As).T.tolist()
    for k in range(len(As)):
        U += (1/(k+1) * np.sum(As_T[k]))
    return U