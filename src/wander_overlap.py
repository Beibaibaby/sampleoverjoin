import time
import pandas as pd
import numpy as np
import math
# from collections import defaultdict
from acyclic_join import *
from wander_size import *

# online overlap estimation
def olp_random_walk(j, hs):
    pri_keys = ['NationKey', 'S_SuppKey', 'CustKey', 'OrderKey', 'LineNumber']
    """
    :param j: join j
    :param hs: hash table for join keys
    :return: joined tuple and probability
    """ 
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

def wander_gen_o(js, hss, alpha):
    pri_keys = ['NationKey', 'S_SuppKey', 'CustKey', 'OrderKey', 'LineNumber']
    ts = []
    ps = []
    count = []
    recip_ps = []
    inter_count = []
    olp = 0
    olps = []
    ratios = []
    
    max_pr = 0
    max_accu_olp = 0

    # while (max_pr < alpha):
    for i in range(1000):
        t, p = olp_random_walk(js[0], hss[0])
        # print(t)
        # print("p: ", p)
        ts.append(t)
        ps.append(p)
        if p == 0:
            recip_ps.append(0)
            count.append(0)
        else:
            recip_ps.append(1/p)
            count_t = round(1/p)
            count.append(count_t)
            # max_p = p
            # check overlap
            # join path j_index
            find = True
            for j_index in range(1,len(js)):
                # temp_p = 1 / js[j_index].tables[0].shape[0]
                # table h_index
                for h_index in range(len(hss[0])):
                    # key
                    # print(t)
                    t_value = t[js[0].keys[h_index]].values[0]
                    if t_value in hss[j_index][h_index]:
                        # t_next_value = t[js[0].keys[h_index+1]].values[0]
                        t_next_value = t[pri_keys[h_index+1]].values[0]
                        temp_bucket = hss[j_index][h_index][t_value]
                        # value
                        if t_next_value in temp_bucket:
                            continue
                           # temp_p *= 1 / len(temp_bucket)
                        else: 
                            find = False
                            break
                    else: 
                        find = False
                        break
                if find is False: 
                    break
            # else:
                # max_p = max(max_p, temp_p)
            if find:
                inter_count.append(count_t)
            # print("find? ", find)
        # print(np.sum(inter_count))
        # print(np.sum(count))
        binom_ratio = np.sum(inter_count) / np.sum(count) 
        olp = np.mean(recip_ps) * binom_ratio
        eps = calc_conf(recip_ps, binom_ratio, alpha)
        
        olps.append(olp)
        ratios.append(binom_ratio)
        # print("olp: ", olp)
        
        pr = olp_calc_pr(eps, binom_ratio, ratios)
            # print("pr: ", pr)

        if pr > max_pr and olp > 0:
            max_accu_olp = olp
            max_pr = pr

        # print("pr: ", pr)
    print(max_accu_olp)
    return max_accu_olp, max_pr


def olp_calc_pr(epsilon, ratio, ratios):
    count_true = 0
    count_false = 0
    for estimator in ratios:
        if abs(estimator - ratio) <= epsilon:
            count_true += 1
        else:
            count_false += 1
    # print("count true: ", count_true)
    # print("count false: ", count_false)
    return count_true / (count_true + count_false)


def calc_conf(recip_ps, binom_ratio, alpha):
    z_alpha  = sps.norm.ppf((alpha + 1) / 2, loc=0, scale=1)
    # size_sigma = np.std(recip_ps)
    # size = np.mean(recip_ps)
    # var = size_sigma * binom_ratio * (1 - binom_ratio) + size_sigma * binom_ratio + size * binom_ratio * (1 - binom_ratio)
    var = binom_ratio * (1 - binom_ratio)
    sig = math.sqrt(var)
    eps = z_alpha * sig / math.sqrt(len(recip_ps))
    
    return eps

def gen_os(js, hs, alpha):
    #find intersection of values in first join attribute in all tables

    ans = p_set(list(range(len(js))))
    
    Os = []
    Ps = []
    for subset in ans:
        new_js = []
        new_hs = []
        for index in subset:
            new_js.append(js[index])
            new_hs.append(hs[index])
        max_accu_olp, max_pr = wander_gen_o(new_js, new_hs, alpha)
        Os.append(max_accu_olp)
        Ps.append(max_pr)
    return ans, Os, Ps

def calc_As(js, Os, ans, e_j):
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
            if (k == 1): A += e_j[j]
            # if (k == 1): A += e_size(js[j])
            # if (k == 1): A += exact_j[j]
            # Calculate A
            for r in range(k+1, n+1):
                # print(math.comb(r-1, k-1))
                # print("As[r-1]", As[j][r-1])
                A -= (math.comb(r-1, k-1) * As[j][r-1])
            As[j][k-1] = A
    return As

def calc_U(js, hs, alpha, e_j):

    ans, Os, ps = gen_os(js, hs, alpha)
    print(ans)
    print(Os)
    print(ps)
    As = calc_As(js, Os, ans, e_j)
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


def main():
    scale = 0.2
    overlap = 0.02
    
    n = 10000
    k = 0

    # nation = pd.read_table('./tpch_1/nation.tbl', index_col=False, names=['NationKey','NationName','RegionKey','Comment'], delimiter = '|').iloc[:, :-1]
    # supplier = pd.read_table('./tpch_1/supplier.tbl', index_col=False, names=['S_SuppKey','SuppName','Address','NationKey','Phone','Acctbl','Comment'], delimiter = '|').iloc[:, :-1]
    # customer = pd.read_table('./tpch_1/customer.tbl', index_col=False, names=['CustKey','CustName','Address','NationKey','Phone','Acctbal','MktSegment','Comment'], delimiter = '|').iloc[:, :-1]
    # orders = pd.read_table('./tpch_1/orders.tbl', index_col=False, names=['OrderKey','CustKey','OrderStatus','TotalPrice','OrderDate','OrderPriority','Clerk','ShipPriority','Comment'], delimiter = '|').iloc[:, :-1]
    # lineitem = pd.read_table('./tpch_1/lineitem.tbl', index_col=False, names=['OrderKey','PartKey','L_SuppKey','LineNumber','Quantity','ExtendedPrice','Discount','Tax','ReturnFlag',
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

    # nation_sample_3,supplier_sample_3,customer_sample_3,orders_sample_3,lineitem_sample_3 = process_tpch(overlap, scale, nation, supplier, customer, orders, lineitem)
    # nation_sample_3.to_csv(r'./tpch_wander_test/n3.csv')
    # supplier_sample_3.to_csv(r'./tpch_wander_test/s3.csv')
    # customer_sample_3.to_csv(r'./tpch_wander_test/c3.csv')
    # orders_sample_3.to_csv(r'./tpch_wander_test/o3.csv')
    # lineitem_sample_3.to_csv(r'./tpch_wander_test/l3.csv')

    # print("3 done")

    # nation_sample_4,supplier_sample_4,customer_sample_4,orders_sample_4,lineitem_sample_4 = process_tpch(overlap, scale, nation, supplier, customer, orders, lineitem)
    # nation_sample_4.to_csv(r'./tpch_wander_test/n4.csv')
    # supplier_sample_4.to_csv(r'./tpch_wander_test/s4.csv')
    # customer_sample_4.to_csv(r'./tpch_wander_test/c4.csv')
    # orders_sample_4.to_csv(r'./tpch_wander_test/o4.csv')
    # lineitem_sample_4.to_csv(r'./tpch_wander_test/l4.csv')

    # print("4 done")

    # nation_sample_5,supplier_sample_5,customer_sample_5,orders_sample_5,lineitem_sample_5 = process_tpch(overlap, scale, nation, supplier, customer, orders, lineitem)
    # nation_sample_5.to_csv(r'./tpch_wander_test/n5.csv')
    # supplier_sample_5.to_csv(r'./tpch_wander_test/s5.csv')
    # customer_sample_5.to_csv(r'./tpch_wander_test/c5.csv')
    # orders_sample_5.to_csv(r'./tpch_wander_test/o5.csv')
    # lineitem_sample_5.to_csv(r'./tpch_wander_test/l5.csv')

    # print("5 done")

    nation_sample_1 = pd.read_csv('./tpch_wander_test/n1.csv' ,index_col=0)
    supplier_sample_1 = pd.read_csv('./tpch_wander_test/s1.csv' ,index_col=0)
    customer_sample_1 = pd.read_csv('./tpch_wander_test/c1.csv' ,index_col=0)
    orders_sample_1 = pd.read_csv('./tpch_wander_test/o1.csv' ,index_col=0)
    lineitem_sample_1 = pd.read_csv('./tpch_wander_test/l1.csv' ,index_col=0)

    nation_sample_2 = pd.read_csv('./tpch_wander_test/n2.csv' ,index_col=0)
    supplier_sample_2 = pd.read_csv('./tpch_wander_test/s2.csv' ,index_col=0)
    customer_sample_2 = pd.read_csv('./tpch_wander_test/c2.csv' ,index_col=0)
    orders_sample_2 = pd.read_csv('./tpch_wander_test/o2.csv' ,index_col=0)
    lineitem_sample_2 = pd.read_csv('./tpch_wander_test/l2.csv' ,index_col=0)

    nation_sample_3 = pd.read_csv('./tpch_wander_test/n3.csv' ,index_col=0)
    supplier_sample_3 = pd.read_csv('./tpch_wander_test/s3.csv' ,index_col=0)
    customer_sample_3 = pd.read_csv('./tpch_wander_test/c3.csv' ,index_col=0)
    orders_sample_3 = pd.read_csv('./tpch_wander_test/o3.csv' ,index_col=0)
    lineitem_sample_3 = pd.read_csv('./tpch_wander_test/l3.csv' ,index_col=0)

    # nation_sample_4 = pd.read_csv('./tpch_wander_test/n4.csv' ,index_col=0)
    # supplier_sample_4 = pd.read_csv('./tpch_wander_test/s4.csv' ,index_col=0)
    # customer_sample_4 = pd.read_csv('./tpch_wander_test/c4.csv' ,index_col=0)
    # orders_sample_4 = pd.read_csv('./tpch_wander_test/o4.csv' ,index_col=0)
    # lineitem_sample_4 = pd.read_csv('./tpch_wander_test/l4.csv' ,index_col=0)

    # nation_sample_5 = pd.read_csv('./tpch_wander_test/n5.csv' ,index_col=0)
    # supplier_sample_5 = pd.read_csv('./tpch_wander_test/s5.csv' ,index_col=0)
    # customer_sample_5 = pd.read_csv('./tpch_wander_test/c5.csv' ,index_col=0)
    # orders_sample_5 = pd.read_csv('./tpch_wander_test/o5.csv' ,index_col=0)
    # lineitem_sample_5 = pd.read_csv('./tpch_wander_test/l5.csv' ,index_col=0)

    tables_1 = [nation_sample_1, supplier_sample_1, customer_sample_1, orders_sample_1,lineitem_sample_1]
    tables_2 = [nation_sample_2, supplier_sample_2, customer_sample_2, orders_sample_2,lineitem_sample_2]
    tables_3 = [nation_sample_3, supplier_sample_3, customer_sample_3, orders_sample_3,lineitem_sample_3]
    # tables_4 = [nation_sample_4, supplier_sample_4, customer_sample_4, orders_sample_4,lineitem_sample_4]
    # tables_5 = [nation_sample_5, supplier_sample_5, customer_sample_5, orders_sample_5,lineitem_sample_5]
    keys = ['NationKey', 'NationKey', 'CustKey', 'OrderKey']

    print("step 1 over")

    join_1 = chain_join(tables_1, keys)
    join_2 = chain_join(tables_2, keys)
    join_3 = chain_join(tables_3, keys)
    # join_4 = chain_join(tables_4, keys)
    # join_5 = chain_join(tables_5, keys)

    print("step 2 over")

    # hs_1 = hash_j_pri(join_1)
    # print("Hash success")

    # f = open("./tpch_wander_test/q1_hs.pkl","wb")
    # pickle.dump(hs_1,f)
    # f.close()

    # hs_2 = hash_j_pri(join_2)
    # print("Hash success")

    # f = open("./tpch_wander_test/q2_hs.pkl","wb")
    # pickle.dump(hs_2,f)
    # f.close()

    # hs_3 = hash_j_pri(join_3)
    # print("Hash success")

    # f = open("./tpch_wander_test/q3_hs.pkl","wb")
    # pickle.dump(hs_3,f)
    # f.close()

    # hs_4 = hash_j_pri(join_4)
    # print("Hash success")

    # f = open("./tpch_wander_test/q4_hs.pkl","wb")
    # pickle.dump(hs_4,f)
    # f.close()

    # hs_5 = hash_j_pri(join_5)
    # print("Hash success")

    # f = open("./tpch_wander_test/q5_hs.pkl","wb")
    # pickle.dump(hs_5,f)
    # f.close()
    
    hs_1 = pickle.load(open("./tpch_wander_test/q1_hs.pkl", "rb"))
    hs_2 = pickle.load(open("./tpch_wander_test/q2_hs.pkl", "rb"))
    hs_3 = pickle.load(open("./tpch_wander_test/q3_hs.pkl", "rb"))
    # hs_4 = pickle.load(open("./tpch_wander_test/q4_hs.pkl", "rb"))
    # hs_5 = pickle.load(open("./tpch_wander_test/q5_hs.pkl", "rb"))
    print("hash successfully loaded")

    alpha = 0.9
    js = [join_1, join_2, join_3]
    hs = [hs_1, hs_2, hs_3]
    
    direct_start = time.perf_counter()
    
    e_j_1 = wander_e_size(alpha, join_1, hs_1)
    e_j_2 = wander_e_size(alpha, join_2, hs_2)
    e_j_3 = wander_e_size(alpha, join_3, hs_3)
    # e_j_4 = wander_e_size(alpha, join_4, hs_4)
    # e_j_5 = wander_e_size(alpha, join_5, hs_5)
    
    e_j = [e_j_1, e_j_2, e_j_3]
    u_size = calc_U(js, hs, alpha, e_j)
    print(u_size)
    
    print("ratio: ")
    print(e_j / u_size)
    
    direct_end = time.perf_counter()
    print(f"direct in {direct_end - direct_start:0.4f} seconds")
    
    # print("exact: ", exact_olp([join_1_f, join_2_f, join_3_f]))
    # 0.1: exact:  [144031, 138224, 145540, 31857]
    
    # max_accu_size, max_pr = wander_e_size(0.95, join_1, hs_1)
    # print("size: ", max_accu_size)
    # print("confidence level:", max_pr)
    
    # max_accu_olp, max_pr = wander_gen_o([join_1, join_2, join_3], [hs_1, hs_2, hs_3], alpha)
    # print(max_accu_olp)
    # print(max_pr)
    # print(inter_join)
    
    
    
    
    
    
    base_start = time.perf_counter()
    
    join_1_f = join_1.f_join()
    join_2_f = join_2.f_join()
    join_3_f = join_3.f_join()
    # join_4_f = join_4.f_join()
    # join_5_f = join_5.f_join()
    
    exact_j = [join_1_f.shape[0], join_2_f.shape[0], join_3_f.shape[0]]
    print("Exact join sizes:")
    print(exact_j)
    
    exact_o = exact_olp([join_1_f, join_2_f, join_3_f])
    print("Exact overlap sizes:")
    print(exact_o)

    exact_union = pd.concat([join_1_f, join_2_f, join_3_f]).drop_duplicates() 
    print("Exact union size:")
    print(exact_union.shape[0])
    
    print("ratio: ")
    ratio_exact = []
    for j in exact_j:
        ratio_exact.append(j / exact_union.shape[0])
    print(ratio_exact)
    
    base_end = time.perf_counter()
    print(f"baseline in {base_end - base_start:0.4f} seconds")

if __name__ == '__main__':
    main()