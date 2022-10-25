import json
import pandas as pd
from acyclic_join import *
from build_hash import *
import pickle
# from sample_from_disjoint import *
# from sample_union_bernoulli import *
from uq3_direct_overlap import *
from uq3_sample import *
import time

def hash_j_uq3(j, pri_keys):
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


def fixed_sample(table, fixed, scale):

    table_fixed = table.iloc[:int(table.shape[0] * fixed), :]

    sample_frac = (scale - fixed) / (1 - fixed)
    table_sample = table.iloc[int(table.shape[0] * fixed):, :].sample(frac=sample_frac, replace=False)

    frames = [table_fixed, table_sample]
    sample = pd.concat(frames)

    return sample


def process_tpch(fixed, scale):

    # Qx: nation, supplier, customer, orders, lineitem
    supplier = pd.read_table('./tpch_3/supplier.tbl', index_col=False, 
                names=['SuppKey','SuppName','Address','NationKey','Phone','Acctbl','Comment'], delimiter = '|').iloc[:, :-1]
    customer = pd.read_table('./tpch_3/customer.tbl', index_col=False, 
                names=['CustKey','CustName','Address','NationKey','Phone','Acctbal','MktSegment','Comment'], delimiter = '|').iloc[:, :-1]
    orders = pd.read_table('./tpch_3/orders.tbl', index_col=False, 
                names=['OrderKey','CustKey','OrderStatus','TotalPrice','OrderDate','OrderPriority','Clerk','ShipPriority','Comment'], delimiter = '|').iloc[:, :-1]
    # Change rows to random order
    supplier.sample(frac=1)
    customer.sample(frac=1)
    orders.sample(frac=1)

    # should adjust percentage and scale according to table size
    supplier_sample = fixed_sample(supplier, fixed, scale).reset_index(drop=True)
    customer_sample = fixed_sample(customer, fixed, scale).reset_index(drop=True)
    orders_sample = fixed_sample(orders, fixed, scale).reset_index(drop=True)

    return supplier_sample, customer_sample, orders_sample

def to_q1(tables):
    keys = ['NationKey', 'CustKey']
    join_q1 = chain_join(tables, keys)
    return join_q1


def to_q2(tables):

    origin_supplier = tables[0]
    origin_costomer = tables[1]

    # SUPPKEY, NATIONKEY, PHONE, ACCTBAL
    supplier_1 = origin_supplier.iloc[:, [0,3,4,5]]

    # CUSTKEY, NAME, ADDRESS
    customer_1 = origin_costomer.iloc[:, [0,1,2]]

    # CUSTKEY, NATIONKEY, PHONE, ACCTBAL, MKTSEGMENT
    customer_2 = origin_costomer.iloc[:, [0,3,4,5,6]]

    # SUPPKEY NAME ADDRESS
    supplier_2 = origin_supplier.iloc[:, [0,1,2]]

    orders = tables[2]

    # new_tables = [customer_1, customer_2, supplier_1, supplier_2, orders]
    # new_keys = ['CustKey', 'NationKey', 'SuppKey', 'CustKey']
    
    new_tables = [supplier_2, supplier_1, customer_2, customer_1, orders]
    new_keys = ['SuppKey', 'NationKey', 'CustKey', 'CustKey']

    join_q2 = chain_join(new_tables, new_keys)
    return join_q2



def to_q3(tables):
    
    supplier = tables[0]
    customer = tables[1]
    
    origin_orders = tables[2]
    
    # ORDERKEY, ORDERPRIORITY, CLERK, SHIPPRIORITY
    orders_1 = origin_orders.iloc[:, [0,5,6,7]]
    
    # ORDERKEY, CUSTKEY, ORDERSTATUS, TOTALPRICE, ORDERDATE
    orders_2 = origin_orders.iloc[:, [0,1,2,3,4]]
    
    tables = [supplier, customer, orders_2, orders_1]
    keys = ['NationKey', 'CustKey', 'OrderKey']
    
    join_q3 = chain_join(tables, keys)
    
    return join_q3

def q1_to_norm(join_q1):
    
    origin_supplier = join_q1.tables[0]
    origin_costomer = join_q1.tables[1]
    origin_orders = join_q1.tables[2]
    
    # SUPPKEY NAME ADDRESS
    supplier_2 = origin_supplier.iloc[:, [0,1,2]]
    
    # SUPPKEY, NATIONKEY, PHONE, ACCTBAL
    supplier_1 = origin_supplier.iloc[:, [0,3,4,5]]
    
    # CUSTKEY, NATIONKEY, PHONE, ACCTBAL, MKTSEGMENT
    customer_2 = origin_costomer.iloc[:, [0,3,4,5,6]]

    # CUSTKEY, NAME, ADDRESS
    customer_1 = origin_costomer.iloc[:, [0,1,2]]
    
    # ORDERKEY, CUSTKEY, ORDERSTATUS, TOTALPRICE, ORDERDATE
    orders_2 = origin_orders.iloc[:, [0,1,2,3,4]]

    # ORDERKEY, ORDERPRIORITY, CLERK, SHIPPRIORITY
    orders_1 = origin_orders.iloc[:, [0,5,6,7]]
    
    tables = [supplier_2, supplier_1, customer_2, customer_1, orders_2, orders_1]
    keys = ['SuppKey', 'NationKey', 'CustKey', 'CustKey', 'OrderKey']
    join_type = [False, True, False, True, False]
    
    norm_join_q1 = norm_chain_join(tables, keys, join_type)
    
    return norm_join_q1


def q2_to_norm(join_q2):
    
    origin_orders = join_q2.tables[4]
    
    # SUPPKEY NAME ADDRESS
    supplier_2 = join_q2.tables[0]
    
    # SUPPKEY, NATIONKEY, PHONE, ACCTBAL
    supplier_1 = join_q2.tables[1]
    
    # CUSTKEY, NATIONKEY, PHONE, ACCTBAL, MKTSEGMENT
    customer_2 = join_q2.tables[2]
    
    # CUSTKEY, NAME, ADDRESS
    customer_1 = join_q2.tables[3]
    
    # ORDERKEY, CUSTKEY, ORDERSTATUS, TOTALPRICE, ORDERDATE
    orders_2 = origin_orders.iloc[:, [0,1,2,3,4]]

    # ORDERKEY, ORDERPRIORITY, CLERK, SHIPPRIORITY
    orders_1 = origin_orders.iloc[:, [0,5,6,7]]
    
    tables = [supplier_2, supplier_1, customer_2, customer_1, orders_2, orders_1]
    keys = ['SuppKey', 'NationKey', 'CustKey', 'CustKey', 'OrderKey']
    join_type = [True, True, True, True, False]
    
    norm_join_q2 = norm_chain_join(tables, keys, join_type)
    
    return norm_join_q2

def q3_to_norm(join_q3):
    
    origin_supplier = join_q3.tables[0]
    origin_costomer = join_q3.tables[1]
    
    # SUPPKEY NAME ADDRESS
    supplier_2 = origin_supplier.iloc[:, [0,1,2]]
    
    # SUPPKEY, NATIONKEY, PHONE, ACCTBAL
    supplier_1 = origin_supplier.iloc[:, [0,3,4,5]]
    
    # CUSTKEY, NATIONKEY, PHONE, ACCTBAL, MKTSEGMENT
    customer_2 = origin_costomer.iloc[:, [0,3,4,5,6]]

    # CUSTKEY, NAME, ADDRESS
    customer_1 = origin_costomer.iloc[:, [0,1,2]]
    
    # ORDERKEY, CUSTKEY, ORDERSTATUS, TOTALPRICE, ORDERDATE
    orders_2 = join_q3.tables[2]

    # ORDERKEY, ORDERPRIORITY, CLERK, SHIPPRIORITY
    orders_1 = join_q3.tables[3]
    
    tables = [supplier_2, supplier_1, customer_2, customer_1, orders_2, orders_1]
    keys = ['SuppKey', 'NationKey', 'CustKey', 'CustKey', 'OrderKey']
    join_type = [False, True, False, True, True]
    
    norm_join_q3 = norm_chain_join(tables, keys, join_type)
    
    return norm_join_q3
    
    
def sort_by_index(tables):
    new_tables = []
    for table in tables:
        new_table = table.reset_index(drop=True)
        new_tables.append(new_table)
    return new_tables

# baseline
def sample_from_join(join_u, n):
    return join_u.sample(n)

def main():
    # import argparse
    
    # parser = argparse.ArgumentParser(description='Random sampling over union of joins')
    # parser.add_argument('--scale', type=float, default=0.001, help='less than 1; scale of the 1g dataset')
    # parser.add_argument('--overlap', type=float, default=0.0002, help='percentage of overlapping data between joins')
    # args = parser.parse_args()
    
    # scale = args.scale
    # overlap = args.overlap
    
    scale = 0.6
    overlap = 0.1

    n = 100000
    k = 0

    # supplier_sample_1,customer_sample_1,orders_sample_1 = process_tpch(overlap, scale)
    # supplier_sample_1.to_csv(r'./uq3_3_3/s1.csv')
    # customer_sample_1.to_csv(r'./uq3_3_3/c1.csv')
    # orders_sample_1.to_csv(r'./uq3_3_3/o1.csv')

    # supplier_sample_2,customer_sample_2,orders_sample_2 = process_tpch(overlap, scale)
    # supplier_sample_2.to_csv(r'./uq3_3_3/s2.csv')
    # customer_sample_2.to_csv(r'./uq3_3_3/c2.csv')
    # orders_sample_2.to_csv(r'./uq3_3_3/o2.csv')

    # supplier_sample_3,customer_sample_3,orders_sample_3 = process_tpch(overlap, scale)
    # supplier_sample_3.to_csv(r'./uq3_3_3/s3.csv')
    # customer_sample_3.to_csv(r'./uq3_3_3/c3.csv')
    # orders_sample_3.to_csv(r'./uq3_3_3/o3.csv')

    supplier_sample_1 = pd.read_csv('./uq3_3_3/s1.csv' ,index_col=0)
    customer_sample_1 = pd.read_csv('./uq3_3_3/c1.csv' ,index_col=0)
    orders_sample_1 = pd.read_csv('./uq3_3_3/o1.csv' ,index_col=0)

    supplier_sample_2 = pd.read_csv('./uq3_3_3/s2.csv' ,index_col=0)
    customer_sample_2 = pd.read_csv('./uq3_3_3/c2.csv' ,index_col=0)
    orders_sample_2 = pd.read_csv('./uq3_3_3/o2.csv' ,index_col=0)

    supplier_sample_3 = pd.read_csv('./uq3_3_3/s3.csv' ,index_col=0)
    customer_sample_3 = pd.read_csv('./uq3_3_3/c3.csv' ,index_col=0)
    orders_sample_3 = pd.read_csv('./uq3_3_3/o3.csv' ,index_col=0)

    tables_1 = [supplier_sample_1, customer_sample_1, orders_sample_1]
    tables_2 = [supplier_sample_2, customer_sample_2, orders_sample_2]
    tables_3 = [supplier_sample_3, customer_sample_3, orders_sample_3]

    print("step 1 over")

    join_1 = to_q1(tables_1)
    join_2 = to_q2(tables_2)
    join_3 = to_q3(tables_3)

    # print("step 2 over")

    # hs_1 = hash_j_uq3_index(join_1)
    # print("Hash success")

    # f = open("./uq3_3_3/q1_hs.pkl","wb")
    # pickle.dump(hs_1,f)
    # f.close()

    # hs_2 = hash_j_uq3_index(join_2)
    # print("Hash success")

    # f = open("./uq3_3_3/q2_hs.pkl","wb")
    # pickle.dump(hs_2,f)
    # f.close()

    # hs_3 = hash_j_uq3_index(join_3)
    # print("Hash success")

    # f = open("./uq3_3_3/q3_hs.pkl","wb")
    # pickle.dump(hs_3,f)
    # f.close()
    
    
    # hs_1_pri = hash_j_uq3(join_1, ['SuppKey', 'CustKey', 'OrderKey'])
    # print("Hash success")

    # f = open("./uq3_3_3/q1_hs_pri.pkl","wb")
    # pickle.dump(hs_1_pri,f)
    # f.close()

    # hs_2_pri = hash_j_uq3(join_2, ['SuppKey', 'SuppKey', 'CustKey', 'CustKey', 'OrderKey'])
    # print("Hash success")

    # f = open("./uq3_3_3/q2_hs_pri.pkl","wb")
    # pickle.dump(hs_2_pri,f)
    # f.close()

    # hs_3_pri = hash_j_uq3(join_3, ['SuppKey', 'CustKey', 'OrderKey', 'OrderKey'])
    # print("Hash success")

    # f = open("./uq3_3_3/q3_hs_pri.pkl","wb")
    # pickle.dump(hs_3_pri,f)
    # f.close()
    
    
    hs_1 = pickle.load(open("./uq3_3_3/q1_hs.pkl", "rb"))
    hs_2 = pickle.load(open("./uq3_3_3/q2_hs.pkl", "rb"))
    hs_3 = pickle.load(open("./uq3_3_3/q3_hs.pkl", "rb"))
    
    # hs_1_pri = pickle.load(open("./uq3_3_3/q1_hs_pri.pkl", "rb"))
    # hs_2_pri = pickle.load(open("./uq3_3_3/q2_hs_pri.pkl", "rb"))
    # hs_3_pri = pickle.load(open("./uq3_3_3/q3_hs_pri.pkl", "rb"))
    
    # ----------------------------------- normalization ----------------------------------- 
    # norm_join_1 = q1_to_norm(join_1)
    # norm_join_2 = q2_to_norm(join_2)
    # norm_join_3 = q3_to_norm(join_3)

    # # ----------------------------------- Estimation ----------------------------------- 
    
    # j_size = [e_size(join_1), e_size(join_2), e_size(join_3)]
    # print("Estimated join sizes:")
    # print(j_size)

    # ans, Os = gen_os([norm_join_1, norm_join_2, norm_join_3])
    # print("Estimated overlap sizes: ")
    # print(ans)
    # print(Os)
    
    # direct_start = time.perf_counter()
    # # As = calc_As([join_1, join_2, join_3], Os, ans)
    # e_union = calc_U([join_1, join_2, join_3], [norm_join_1, norm_join_2, norm_join_3])
    # print("Estimated union size: ")
    # print(e_union)
    
    # print("ratio: ")
    # print(j_size / e_union)
    
    # direct_end = time.perf_counter()
    # print(f"direct in {direct_end - direct_start:0.4f} seconds")

    # base_start = time.perf_counter()
    
    # join_1_f = join_1.f_join()
    # join_2_f = join_2.f_join()
    # join_3_f = join_3.f_join()
    
    # exact_j = [join_1_f.shape[0], join_2_f.shape[0], join_3_f.shape[0]]
    # print("Exact join sizes:")
    # print(exact_j)
    
    # # exact_o = exact_olp([join_1_f, join_2_f, join_3_f])
    # # print("Exact overlap sizes:")
    # # print(exact_o)

    # exact_union = pd.concat([join_1_f, join_2_f, join_3_f]).drop_duplicates() 
    # print("Exact union size:")
    # print(exact_union.shape[0])
    
    # print("ratio: ")
    # ratio_exact = []
    # for j in exact_j:
    #     ratio_exact.append(j / exact_union.shape[0])
    # print(ratio_exact)
    
    # base_end = time.perf_counter()
    # print(f"baseline in {base_end - base_start:0.4f} seconds")
    
    # real_sample = exact_union.sample(1000)
    
    # # base_sample_result = sample_from_join(join_f, n)
    # base_end = time.perf_counter()
    # print(f"baseline in {base_end - base_start:0.4f} seconds")
    
    # real_sample.to_csv(r'./uq3_3_3/real.csv')

    

    # ----------------------------------- sample from disjoint ----------------------------------- 


    olken_S, olken_time = olken_sample_from_disjoint([join_1, join_2, join_3], n, [hs_1, hs_2, hs_3])
    print(olken_time)
    olken_S.to_csv(r'./uq3_3_3/olken_sample_100000.csv')
    with open('./uq3_3_3/olken_time_100000.txt', 'w') as f:
        f.write(json.dumps(olken_time))

    # baseline
    # base_start = time.perf_counter()
    # join_1_f = join_1.f_join()
    # join_2_f = join_2.f_join()
    # join_3_f = join_3.f_join()
    # cur_time = time.perf_counter()
    # print(f"join in {cur_time - base_start:0.4f} seconds")

if __name__ == '__main__':
    main()