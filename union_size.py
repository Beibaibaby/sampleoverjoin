import json
import pandas as pd
from equi_chain_overlap import *
from acyclic_join import *
from build_hash import *
import pickle
from sample_from_disjoint import *
from sample_union_bernoulli import *

def fixed_sample(table, fixed, scale):

    table_fixed = table.iloc[:int(table.shape[0] * fixed), :]

    sample_frac = (scale - fixed) / (1 - fixed)
    table_sample = table.iloc[int(table.shape[0] * fixed):, :].sample(frac=sample_frac, replace=False)

    frames = [table_fixed, table_sample]
    sample = pd.concat(frames)

    return sample

def process_tpch(fixed, scale, nation, supplier, customer, orders, lineitem):

    # Change rows to random order
    # nation.sample(frac=1)

    # should adjust percentage and scale according to table size
    nation_sample = nation
    supplier_sample = fixed_sample(supplier, fixed, scale).reset_index(drop=True)
    customer_sample = fixed_sample(customer, fixed, scale).reset_index(drop=True)
    orders_sample = fixed_sample(orders, fixed, scale).reset_index(drop=True)
    lineitem_sample = fixed_sample(lineitem, fixed, scale).reset_index(drop=True)

    return nation_sample, supplier_sample, customer_sample,orders_sample,lineitem_sample
    # return lineitem_sample,orders_sample,customer_sample,supplier_sample,nation_sample


def sort_by_index(tables):
    new_tables = []
    for table in tables:
        new_table = table.reset_index(drop=True)
        new_tables.append(new_table)
    return new_tables

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

    n = 10000
    k = 0

    # Qx: nation, supplier, customer, orders, lineitem 
    nation = pd.read_table('./tpch_1/nation.tbl', index_col=False, names=['NationKey','NationName','RegionKey','Comment'], delimiter = '|').iloc[:, :-1]
    supplier = pd.read_table('./tpch_1/supplier.tbl', index_col=False, names=['SuppKey','SuppName','Address','NationKey','Phone','Acctbl','Comment'], delimiter = '|').iloc[:, :-1]
    customer = pd.read_table('./tpch_1/customer.tbl', index_col=False, names=['CustKey','CustName','Address','NationKey','Phone','Acctbal','MktSegment','Comment'], delimiter = '|').iloc[:, :-1]
    orders = pd.read_table('./tpch_1/orders.tbl', index_col=False, names=['OrderKey','CustKey','OrderStatus','TotalPrice','OrderDate','OrderPriority','Clerk','ShipPriority','Comment'], delimiter = '|').iloc[:, :-1]
    lineitem = pd.read_table('./tpch_1/lineitem.tbl', index_col=False, names=['OrderKey','PartKey','SuppKey','LineNumber','Quantity','ExtendedPrice','Discount','Tax','ReturnFlag',
    'LineStatus','ShipDate','CommitDate','ReceiptDate','ShipinStruct','ShipMode','Comment'], delimiter = '|').iloc[:, :-1]

    print("read successfully")
    
    supplier.sample(frac=1)
    customer.sample(frac=1)
    orders.sample(frac=1)
    lineitem.sample(frac=1)

    nation_sample_1,supplier_sample_1,customer_sample_1,orders_sample_1,lineitem_sample_1 = process_tpch(overlap, scale, nation, supplier, customer, orders, lineitem)
    # nation_sample_1.to_csv(r'./tpch_1_chain_5/n1.csv')
    # supplier_sample_1.to_csv(r'./tpch_1_chain_5/s1.csv')
    # customer_sample_1.to_csv(r'./tpch_1_chain_5/c1.csv')
    # orders_sample_1.to_csv(r'./tpch_1_chain_5/o1.csv')
    # lineitem_sample_1.to_csv(r'./tpch_1_chain_5/l1.csv')

    print("1 done")

    nation_sample_2,supplier_sample_2,customer_sample_2,orders_sample_2,lineitem_sample_2 = process_tpch(overlap, scale, nation, supplier, customer, orders, lineitem)
    # nation_sample_2.to_csv(r'./tpch_1_chain_5/n2.csv')
    # supplier_sample_2.to_csv(r'./tpch_1_chain_5/s2.csv')
    # customer_sample_2.to_csv(r'./tpch_1_chain_5/c2.csv')
    # orders_sample_2.to_csv(r'./tpch_1_chain_5/o2.csv')
    # lineitem_sample_2.to_csv(r'./tpch_1_chain_5/l2.csv')

    print("2 done")

    nation_sample_3,supplier_sample_3,customer_sample_3,orders_sample_3,lineitem_sample_3 = process_tpch(overlap, scale, nation, supplier, customer, orders, lineitem)
    # nation_sample_3.to_csv(r'./tpch_1_chain_5/n3.csv')
    # supplier_sample_3.to_csv(r'./tpch_1_chain_5/s3.csv')
    # customer_sample_3.to_csv(r'./tpch_1_chain_5/c3.csv')
    # orders_sample_3.to_csv(r'./tpch_1_chain_5/o3.csv')
    # lineitem_sample_3.to_csv(r'./tpch_1_chain_5/l3.csv')

    print("3 done")

    nation_sample_4,supplier_sample_4,customer_sample_4,orders_sample_4,lineitem_sample_4 = process_tpch(overlap, scale, nation, supplier, customer, orders, lineitem)
    # nation_sample_4.to_csv(r'./tpch_1_chain_5/n4.csv')
    # supplier_sample_4.to_csv(r'./tpch_1_chain_5/s4.csv')
    # customer_sample_4.to_csv(r'./tpch_1_chain_5/c4.csv')
    # orders_sample_4.to_csv(r'./tpch_1_chain_5/o4.csv')
    # lineitem_sample_4.to_csv(r'./tpch_1_chain_5/l4.csv')

    print("4 done")

    nation_sample_5,supplier_sample_5,customer_sample_5,orders_sample_5,lineitem_sample_5 = process_tpch(overlap, scale, nation, supplier, customer, orders, lineitem)
    # nation_sample_5.to_csv(r'./tpch_1_chain_5/n5.csv')
    # supplier_sample_5.to_csv(r'./tpch_1_chain_5/s5.csv')
    # customer_sample_5.to_csv(r'./tpch_1_chain_5/c5.csv')
    # orders_sample_5.to_csv(r'./tpch_1_chain_5/o5.csv')
    # lineitem_sample_5.to_csv(r'./tpch_1_chain_5/l5.csv')

    print("5 done")

    # nation_sample_1 = pd.read_csv('./tpch_1_chain_5/n1.csv' ,index_col=0)
    # supplier_sample_1 = pd.read_csv('./tpch_1_chain_5/s1.csv' ,index_col=0)
    # customer_sample_1 = pd.read_csv('./tpch_1_chain_5/c1.csv' ,index_col=0)
    # orders_sample_1 = pd.read_csv('./tpch_1_chain_5/o1.csv' ,index_col=0)
    # lineitem_sample_1 = pd.read_csv('./tpch_1_chain_5/l1.csv' ,index_col=0)

    # nation_sample_2 = pd.read_csv('./tpch_1_chain_5/n2.csv' ,index_col=0)
    # supplier_sample_2 = pd.read_csv('./tpch_1_chain_5/s2.csv' ,index_col=0)
    # customer_sample_2 = pd.read_csv('./tpch_1_chain_5/c2.csv' ,index_col=0)
    # orders_sample_2 = pd.read_csv('./tpch_1_chain_5/o2.csv' ,index_col=0)
    # lineitem_sample_2 = pd.read_csv('./tpch_1_chain_5/l2.csv' ,index_col=0)

    # nation_sample_3 = pd.read_csv('./tpch_1_chain_5/n3.csv' ,index_col=0)
    # supplier_sample_3 = pd.read_csv('./tpch_1_chain_5/s3.csv' ,index_col=0)
    # customer_sample_3 = pd.read_csv('./tpch_1_chain_5/c3.csv' ,index_col=0)
    # orders_sample_3 = pd.read_csv('./tpch_1_chain_5/o3.csv' ,index_col=0)
    # lineitem_sample_3 = pd.read_csv('./tpch_1_chain_5/l3.csv' ,index_col=0)

    # nation_sample_4 = pd.read_csv('./tpch_1_chain_5/n4.csv' ,index_col=0)
    # supplier_sample_4 = pd.read_csv('./tpch_1_chain_5/s4.csv' ,index_col=0)
    # customer_sample_4 = pd.read_csv('./tpch_1_chain_5/c4.csv' ,index_col=0)
    # orders_sample_4 = pd.read_csv('./tpch_1_chain_5/o4.csv' ,index_col=0)
    # lineitem_sample_4 = pd.read_csv('./tpch_1_chain_5/l4.csv' ,index_col=0)

    # nation_sample_5 = pd.read_csv('./tpch_1_chain_5/n5.csv' ,index_col=0)
    # supplier_sample_5 = pd.read_csv('./tpch_1_chain_5/s5.csv' ,index_col=0)
    # customer_sample_5 = pd.read_csv('./tpch_1_chain_5/c5.csv' ,index_col=0)
    # orders_sample_5 = pd.read_csv('./tpch_1_chain_5/o5.csv' ,index_col=0)
    # lineitem_sample_5 = pd.read_csv('./tpch_1_chain_5/l5.csv' ,index_col=0)

    tables_1 = [nation_sample_1, supplier_sample_1, customer_sample_1, orders_sample_1,lineitem_sample_1]
    tables_2 = [nation_sample_2, supplier_sample_2, customer_sample_2, orders_sample_2,lineitem_sample_2]
    tables_3 = [nation_sample_3, supplier_sample_3, customer_sample_3, orders_sample_3,lineitem_sample_3]
    tables_4 = [nation_sample_4, supplier_sample_4, customer_sample_4, orders_sample_4,lineitem_sample_4]
    tables_5 = [nation_sample_5, supplier_sample_5, customer_sample_5, orders_sample_5,lineitem_sample_5]
    keys = ['NationKey', 'NationKey', 'CustKey', 'OrderKey']

    print("step 1 over")

    join_1 = chain_join(tables_1, keys)
    join_2 = chain_join(tables_2, keys)
    join_3 = chain_join(tables_3, keys)
    join_4 = chain_join(tables_4, keys)
    join_5 = chain_join(tables_5, keys)

    print("step 2 over")

    # hs_1 = hash_j(join_1)
    # print("Hash success")

    # f = open("./tpch_1_chain_5/q1_hs.pkl","wb")
    # pickle.dump(hs_1,f)
    # f.close()

    # hs_2 = hash_j(join_2)
    # print("Hash success")

    # f = open("./tpch_1_chain_5/q2_hs.pkl","wb")
    # pickle.dump(hs_2,f)
    # f.close()

    # hs_3 = hash_j(join_3)
    # print("Hash success")

    # f = open("./tpch_1_chain_5/q3_hs.pkl","wb")
    # pickle.dump(hs_3,f)
    # f.close()

    # hs_4 = hash_j(join_4)
    # print("Hash success")

    # f = open("./tpch_1_chain_5/q4_hs.pkl","wb")
    # pickle.dump(hs_4,f)
    # f.close()

    # hs_5 = hash_j(join_5)
    # print("Hash success")

    # f = open("./tpch_1_chain_5/q5_hs.pkl","wb")
    # pickle.dump(hs_5,f)
    # f.close()
    
    # hs_1 = pickle.load(open("./tpch_1_chain_5/q1_hs.pkl", "rb"))
    # hs_2 = pickle.load(open("./tpch_1_chain_5/q2_hs.pkl", "rb"))
    # hs_3 = pickle.load(open("./tpch_1_chain_5/q3_hs.pkl", "rb"))
    # hs_4 = pickle.load(open("./tpch_1_chain_5/q4_hs.pkl", "rb"))
    # hs_5 = pickle.load(open("./tpch_1_chain_5/q5_hs.pkl", "rb"))
    # print("hash successfully loaded")

    # ----------------------------------- Estimation ----------------------------------- 

    # j_size = [e_size(join_1), e_size(join_2), e_size(join_3), e_size(join_4), e_size(join_5)]
    # print("Estimated join sizes:")
    # print(j_size)

    # ans, Os = gen_os([join_1, join_2, join_3, join_4, join_5])
    # print("Estimated overlap sizes: ")
    # print(ans)
    # print(Os)
    
    # direct_start = time.perf_counter()
    # # As = calc_As([join_1, join_2, join_3, join_4, join_5], Os, ans)
    # e_union = calc_U([join_1, join_2, join_3, join_4, join_5])
    # # print("Estimated union size: ")
    # # print(e_union)
    
    # direct_end = time.perf_counter()
    # print(f"direct in {direct_end - direct_start:0.4f} seconds")
    
    # print("ratio: ")
    # print(j_size / e_union)
    
    base_start = time.perf_counter()
    join_1_f = join_1.f_join()
    join_2_f = join_2.f_join()
    join_3_f = join_3.f_join()
    join_4_f = join_4.f_join()
    join_5_f = join_5.f_join()
    
    # exact_j = [join_1_f.shape[0], join_2_f.shape[0], join_3_f.shape[0], join_4_f.shape[0], join_5_f.shape[0]]
    # print("Exact join sizes:")
    # print(exact_j)
    
    # exact_o = exact_olp([join_1_f, join_2_f, join_3_f, ])
    # print("Exact overlap sizes:")
    # print(exact_o)

    exact_union = pd.concat([join_1_f, join_2_f, join_3_f, join_4_f, join_5_f]).drop_duplicates() 
    sample_from_join(exact_union, n)
    
    # print("Exact union size:")
    # print(exact_union.shape[0])
    
    base_end = time.perf_counter()
    print(f"baseline in {base_end - base_start:0.4f} seconds")
    
    # print("ratio: ")
    # print(exact_j / exact_union.shape[0])
    
    # ratio_exact = []
    # for j in exact_j:
    #     ratio_exact.append(j / exact_union.shape[0])
    # print(ratio_exact)


if __name__ == '__main__':
    main()