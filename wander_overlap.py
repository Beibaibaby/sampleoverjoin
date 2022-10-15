import pandas as pd
import numpy as np
import math
# from collections import defaultdict
from acyclic_join import *
from wander_size import *
from tpch_3_chain_5 import *

def wander_gen_os(js, hss):
    #find intersection of values in first join attribute in all tables

    ans = p_set(list(range(len(js))))
    
    Os = []
    for subset in ans:
        Os.append(gen_o(js,subset, hss))
    
    return ans, Os

# generate powerset
def wander_p_set(nums):
    ans_all = [[]]

    for n in nums:
        ans_all += [a+[n] for a in ans_all]
        
    ans = []
    for i in ans_all:
        if len(i) > 1: 
            ans.append(i)
        
    return ans

def wander_gen_o(js, subset, hss):
    olp = 0
    len_s = len(subset)
    # here ts is already S'
    ts = [[] for i in range(len_s)]
    ps = [[] for i in range(len_s)]
    count = [[] for i in range(len_s)]
    recip_ps = [[] for i in range(len_s)]
    it = 0
    sum_j_s = 0
    # while (True):
    for i in range(5000):
        it += 1
        for j_index in subset:
            t, p = random_walk(js[j_index], hss[j_index])
            print(p)
            if p > 0:
                count_t = round(1/p)
                count[j_index].append(count_t)
                ts[j_index].append(t)
                ps[j_index].append(p)
                for p in ps[j_index]:
                    recip_ps[j_index].append(1 / p)
                sum_j_s += np.mean(recip_ps[j_index]) / np.sum(count[j_index])
            # print(ts)
        inter_num = calc_sample_olp(ts, count)
        olp = sum_j_s * np.sum(inter_num) / 2
        print("overlap: ", olp)
        # if olp > 0:
        #     break
    # print("iterations: ", it)
    # print(ts[0])
    print(ts[1])

    return olp

def calc_sample_olp(ts, count):
    inter_num = []
    for t_index in range(len(ts[0])):
        exist = True
        min_count = count[0][t_index] 
        j_index = 1
        while(exist and j_index < len(ts)):   
            for t_temp_index in range(len(ts[j_index])):
                # print("here")
                if ts[j_index][t_temp_index].equals(ts[0][t_index]):
                    # print("here")
                    min_count = min(count[j_index][t_temp_index], min_count)
                    break
                if t_temp_index == len(ts[j_index])-1:
                    exist = False
        # print(exist)
            j_index += 1
        if exist: inter_num.append(min_count)
    return inter_num


def wander_calc_As(js, Os, ans):
    n = len(js)
    As = [ [0]*n for i in range(n)]
    for j in range(len(js)):
        As[j][n-1] = Os[len(Os)-1]
        for k in range(n-1, 0, -1):
            # print("k: ", k)
            A = 0 
            count = 0
            for index in range(len(ans)):
                if (len(ans[index]) == k) and (j in ans[index]):
                    A += Os[index]
                    count += 1
            if (k == 1): A += e_size(js[j])
            # if (k == 1): A += exact_j[j]
            # Calculate A
            for r in range(k+1, n+1):
                # print(math.comb(r-1, k-1))
                # print("As[r-1]", As[j][r-1])
                A -= (math.comb(r-1, k-1) * As[j][r-1])
            As[j][k-1] = A
    return As

def wander_calc_U(js):

    ans, Os = gen_os(js)
    As = calc_As(js, Os, ans)
    U = 0
    # for j in range(len(As)):
    #     for k in range(len(As[j])):
    #         U += (1/(k+1) * As[j][k])

    As_T = np.array(As).T.tolist()
    # print(As_T)
    for k in range(len(As)):
        # print(np.sum(As[k]))
        U += (1/(k+1) * np.sum(As_T[k]))
    return U

def exact_olp(f_js):

    frames = []
    for i in range(len(f_js)):
        frames.append(f_js[i])
    disjoint = pd.concat(frames)
    inter = disjoint.value_counts()[disjoint.value_counts() == len(f_js)]
    
    return inter.shape[0]

def main():
    scale = 0.003
    overlap = 0.003

    # nation = pd.read_table('./tpch_5/nation.tbl', index_col=False, names=['NationKey','NationName','RegionKey','Comment'], delimiter = '|').iloc[:, :-1]
    # supplier = pd.read_table('./tpch_5/supplier.tbl', index_col=False, names=['SuppKey','SuppName','Address','NationKey','Phone','Acctbl','Comment'], delimiter = '|').iloc[:, :-1]
    # customer = pd.read_table('./tpch_5/customer.tbl', index_col=False, names=['CustKey','CustName','Address','NationKey','Phone','Acctbal','MktSegment','Comment'], delimiter = '|').iloc[:, :-1]
    # orders = pd.read_table('./tpch_5/orders.tbl', index_col=False, names=['OrderKey','CustKey','OrderStatus','TotalPrice','OrderDate','OrderPriority','Clerk','ShipPriority','Comment'], delimiter = '|').iloc[:, :-1]
    # lineitem = pd.read_table('./tpch_5/lineitem.tbl', index_col=False, names=['OrderKey','PartKey','SuppKey','LineNumber','Quantity','ExtendedPrice','Discount','Tax','ReturnFlag',
    # 'LineStatus','ShipDate','CommitDate','ReceiptDate','ShipinStruct','ShipMode','Comment'], delimiter = '|').iloc[:, :-1]

    # print("read successfully")

    # nation_sample_1,supplier_sample_1,customer_sample_1,orders_sample_1,lineitem_sample_1 = process_tpch(overlap, scale, nation, supplier, customer, orders, lineitem)
    # nation_sample_1.to_csv(r'./tpch_wander_test/n1.csv')
    # supplier_sample_1.to_csv(r'./tpch_wander_test/s1.csv')
    # customer_sample_1.to_csv(r'./tpch_wander_test/c1.csv')
    # orders_sample_1.to_csv(r'./tpch_wander_test/o1.csv')
    # lineitem_sample_1.to_csv(r'./tpch_wander_test/l1.csv')

    # print("1 done")

    # nation_sample_2,supplier_sample_2,customer_sample_2,orders_sample_2,lineitem_sample_2 = process_tpch(overlap, scale, nation, supplier, customer, orders, lineitem)
    # nation_sample_2.to_csv(r'./tpch_wander_test/n2.csv')
    # supplier_sample_2.to_csv(r'./tpch_wander_test/s2.csv')
    # customer_sample_2.to_csv(r'./tpch_wander_test/c2.csv')
    # orders_sample_2.to_csv(r'./tpch_wander_test/o2.csv')
    # lineitem_sample_2.to_csv(r'./tpch_wander_test/l2.csv')

    # print("2 done")

    nation_sample_1 = pd.read_csv('./tpch_wander_test/n1.csv' ,index_col=0)
    supplier_sample_1 = pd.read_csv('./tpch_wander_test/s1.csv' ,index_col=0)
    customer_sample_1 = pd.read_csv('./tpch_wander_test/c1.csv' ,index_col=0)
    orders_sample_1 = pd.read_csv('./tpch_wander_test/o1.csv' ,index_col=0)
    lineitem_sample_1 = pd.read_csv('./tpch_wander_test/l1.csv' ,index_col=0)
    tables_1 = [supplier_sample_1, nation_sample_1, customer_sample_1, orders_sample_1,lineitem_sample_1]

    # nation_sample_2 = pd.read_csv('./tpch_wander_test/n2.csv' ,index_col=0)
    # supplier_sample_2 = pd.read_csv('./tpch_wander_test/s2.csv' ,index_col=0)
    # customer_sample_2 = pd.read_csv('./tpch_wander_test/c2.csv' ,index_col=0)
    # orders_sample_2 = pd.read_csv('./tpch_wander_test/o2.csv' ,index_col=0)
    # lineitem_sample_2 = pd.read_csv('./tpch_wander_test/l2.csv' ,index_col=0)
    # tables_2 = [nation_sample_2, supplier_sample_2, customer_sample_2, orders_sample_2,lineitem_sample_2]

    keys = ['NationKey', 'NationKey', 'CustKey', 'OrderKey']

    join_1 = chain_join(tables_1, keys)
    # join_2 = chain_join(tables_2, keys)
    print("Join created")

    # hs_1 = hash_j(join_1)
    # print("Hash success")

    # f = open("./tpch_wander_test/hs1.pkl","wb")
    # pickle.dump(hs_1,f)
    # f.close()

    # hs_2 = hash_j(join_2)
    # print("Hash success")

    # f = open("./tpch_wander_test/hs2.pkl","wb")
    # pickle.dump(hs_2,f)
    # f.close()

    hs_1 = pickle.load(open("./tpch_wander_test/hs1.pkl", "rb"))
    # hs_2 = pickle.load(open("./tpch_wander_test/hs2.pkl", "rb"))
    print("successfully loaded")

    # alpha = 0.85
    
    # join_1_f = join_1.f_join()
    # join_2_f = join_2.f_join()
    # print("exact: ", exact_olp([join_1_f, join_1_f]))
    # 8112

    wander_gen_o([join_1, join_1], [0,1], [hs_1, hs_1])

if __name__ == '__main__':
    main()