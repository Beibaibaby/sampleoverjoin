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

def process_tpch(fixed, scale):

    # Qx: nation, supplier, customer, orders, lineitem
    nation = pd.read_table('./tpch_3/nation.tbl', index_col=False, names=['NationKey','NationName','RegionKey','Comment'], delimiter = '|').iloc[:, :-1]
    supplier = pd.read_table('./tpch_3/supplier.tbl', index_col=False, names=['SuppKey','SuppName','Address','NationKey','Phone','Acctbl','Comment'], delimiter = '|').iloc[:, :-1]
    customer = pd.read_table('./tpch_3/customer.tbl', index_col=False, names=['CustKey','CustName','Address','NationKey','Phone','Acctbal','MktSegment','Comment'], delimiter = '|').iloc[:, :-1]
    orders = pd.read_table('./tpch_3/orders.tbl', index_col=False, names=['OrderKey','CustKey','OrderStatus','TotalPrice','OrderDate','OrderPriority','Clerk','ShipPriority','Comment'], delimiter = '|').iloc[:, :-1]
    lineitem = pd.read_table('./tpch_3/lineitem.tbl', index_col=False, names=['OrderKey','PartKey','SuppKey','LineNumber','Quantity','ExtendedPrice','Discount','Tax','ReturnFlag',
    'LineStatus','ShipDate','CommitDate','ReceiptDate','ShipinStruct','ShipMode','Comment'], delimiter = '|').iloc[:, :-1]

    # Change rows to random order
    nation.sample(frac=1)
    supplier.sample(frac=1)
    customer.sample(frac=1)
    orders.sample(frac=1)
    lineitem.sample(frac=1)

    # should adjust percentage and scale according to table size
    nation_sample = nation
    supplier_sample = fixed_sample(supplier, fixed, scale)
    customer_sample = fixed_sample(customer, fixed, scale)
    orders_sample = fixed_sample(orders, fixed, scale)
    lineitem_sample = fixed_sample(lineitem, fixed, scale)

    return nation_sample, supplier_sample, customer_sample,orders_sample,lineitem_sample
    # return lineitem_sample,orders_sample,customer_sample,supplier_sample,nation_sample

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
    scale = 0.8
    overlap = 0.05

    n = 10000
    k = 0

    # nation_sample_1,supplier_sample_1,customer_sample_1,orders_sample_1,lineitem_sample_1 = process_tpch(overlap, scale)
    # nation_sample_1.to_csv(r'./acyclic_3/n1.csv')
    # supplier_sample_1.to_csv(r'./acyclic_3/s1.csv')
    # customer_sample_1.to_csv(r'./acyclic_3/c1.csv')
    # orders_sample_1.to_csv(r'./acyclic_3/o1.csv')
    # lineitem_sample_1.to_csv(r'./acyclic_3/l1.csv')

    # nation_sample_2,supplier_sample_2,customer_sample_2,orders_sample_2,lineitem_sample_2 = process_tpch(overlap, scale)
    # nation_sample_2.to_csv(r'./acyclic_3/n2.csv')
    # supplier_sample_2.to_csv(r'./acyclic_3/s2.csv')
    # customer_sample_2.to_csv(r'./acyclic_3/c2.csv')
    # orders_sample_2.to_csv(r'./acyclic_3/o2.csv')
    # lineitem_sample_2.to_csv(r'./acyclic_3/l2.csv')

    # nation_sample_3,supplier_sample_3,customer_sample_3,orders_sample_3,lineitem_sample_3 = process_tpch(overlap, scale)
    # nation_sample_3.to_csv(r'./acyclic_3/n3.csv')
    # supplier_sample_3.to_csv(r'./acyclic_3/s3.csv')
    # customer_sample_3.to_csv(r'./acyclic_3/c3.csv')
    # orders_sample_3.to_csv(r'./acyclic_3/o3.csv')
    # lineitem_sample_3.to_csv(r'./acyclic_3/l3.csv')

    nation_sample_1 = pd.read_csv('./acyclic_3/n1.csv' ,index_col=0)
    supplier_sample_1 = pd.read_csv('./acyclic_3/s1.csv' ,index_col=0)
    customer_sample_1 = pd.read_csv('./acyclic_3/c1.csv' ,index_col=0)
    orders_sample_1 = pd.read_csv('./acyclic_3/o1.csv' ,index_col=0)
    lineitem_sample_1 = pd.read_csv('./acyclic_3/l1.csv' ,index_col=0)

    nation_sample_2 = pd.read_csv('./acyclic_3/n2.csv' ,index_col=0)
    supplier_sample_2 = pd.read_csv('./acyclic_3/s2.csv' ,index_col=0)
    customer_sample_2 = pd.read_csv('./acyclic_3/c2.csv' ,index_col=0)
    orders_sample_2 = pd.read_csv('./acyclic_3/o2.csv' ,index_col=0)
    lineitem_sample_2 = pd.read_csv('./acyclic_3/l2.csv' ,index_col=0)

    nation_sample_3 = pd.read_csv('./acyclic_3/n3.csv' ,index_col=0)
    supplier_sample_3 = pd.read_csv('./acyclic_3/s3.csv' ,index_col=0)
    customer_sample_3 = pd.read_csv('./acyclic_3/c3.csv' ,index_col=0)
    orders_sample_3 = pd.read_csv('./acyclic_3/o3.csv' ,index_col=0)
    lineitem_sample_3 = pd.read_csv('./acyclic_3/l3.csv' ,index_col=0)

    tables_1 = [nation_sample_1, supplier_sample_1, customer_sample_1, orders_sample_1,lineitem_sample_1]
    tables_2 = [nation_sample_2, supplier_sample_2, customer_sample_2, orders_sample_2,lineitem_sample_2]
    tables_3 = [nation_sample_3, supplier_sample_3, customer_sample_3, orders_sample_3,lineitem_sample_3]
    keys = ['NationKey', 'NationKey', 'CustKey', 'OrderKey']

    tables_1 = sort_by_index(tables_1)
    tables_2 = sort_by_index(tables_2)
    tables_3 = sort_by_index(tables_3)

    print("step 1 over")

    join_1 = chain_join(tables_1, keys)
    join_2 = chain_join(tables_2, keys)
    join_3 = chain_join(tables_3, keys)

    print("step 2 over")

    # hs_1 = hash_j(join_1)
    # print("Hash success")

    # f = open("./acyclic_3/q1_hs.pkl","wb")
    # pickle.dump(hs_1,f)
    # f.close()

    # hs_2 = hash_j(join_2)
    # print("Hash success")

    # f = open("./acyclic_3/q2_hs.pkl","wb")
    # pickle.dump(hs_2,f)
    # f.close()

    # hs_3 = hash_j(join_3)
    # print("Hash success")

    # f = open("./acyclic_3/q3_hs.pkl","wb")
    # pickle.dump(hs_3,f)
    # f.close()
    
    hs_1 = pickle.load(open("./acyclic_3/q1_hs.pkl", "rb"))
    hs_2 = pickle.load(open("./acyclic_3/q2_hs.pkl", "rb"))
    hs_3 = pickle.load(open("./acyclic_3/q3_hs.pkl", "rb"))
    # print(hs_1[2]) 


    # ----------------------------------- Estimation ----------------------------------- 

    # j_size = [e_size(join_1), e_size(join_2), e_size(join_3)]
    # print("Estimated join sizes:")
    # print(j_size)

    # ans, Os = gen_os([join_1, join_2, join_3])
    # print("Estimated overlap sizes: ")
    # print(ans)
    # print(Os)

    # As = calc_As([join_1, join_2, join_3], Os, ans, j_size)
    # e_union = calc_U(As)
    # print("Estimated union size: ")
    # print(e_union)

    
    # join_1_f = join_1.f_join()
    # join_2_f = join_2.f_join()
    # join_3_f = join_3.f_join()
    
    # exact_j = [join_1_f.shape[0], join_2_f.shape[0], join_3_f.shape[0]]
    # print("Exact join sizes:")
    # print(exact_j)
    
    # exact_o = exact_olp([join_1_f, join_2_f, join_3_f])
    # print("Exact overlap sizes:")
    # print(exact_o)

    # exact_union = pd.concat([join_1_f, join_2_f, join_3_f]).drop_duplicates() 
    # print("Exact union size:")
    # print(exact_union.shape[0])



    # ----------------------------------- sample from disjoint ----------------------------------- 
    # exact_S, exact_time = exact_sample_from_disjoint([join_1, join_2, join_3], n, [hs_1, hs_2, hs_3])
    # print(exact_time)
    # exact_S.to_csv(r'./acyclic_3/exact_sample_10000.csv')
    # with open('./acyclic_3/exact_time_10000.txt', 'w') as f:
    #     f.write(json.dumps(exact_time))


    # olken_S, olken_time = olken_sample_from_disjoint([join_1, join_2, join_3], n, [hs_1, hs_2, hs_3])
    # print(olken_time)
    # olken_S.to_csv(r'./acyclic_3/olken_sample_10000.csv')
    # with open('./acyclic_3/olken_time_10000.txt', 'w') as f:
    #     f.write(json.dumps(olken_time))


    # olken_u_b_S, olken_u_b_time = olken_olken_sample_union_bernoulli([join_1, join_2, join_3], n, [hs_1, hs_2, hs_3])
    # print(olken_u_b_time)
    # olken_u_b_S.to_csv(r'./acyclic_3/olken_u_b_sample_10000.csv')
    # with open('./acyclic_3/olken_u_b_time_10000.txt', 'w') as f:
    #     f.write(json.dumps(olken_u_b_time))

    exact_u_b_S, exact_u_b_time = exact_olken_sample_union_bernoulli([join_1, join_2, join_3], n, [hs_1, hs_2, hs_3])
    print(exact_u_b_time)
    exact_u_b_S.to_csv(r'./acyclic_3/exact_u_b_sample_10000.csv')
    with open('./acyclic_3/exact_u_b_time_10000.txt', 'w') as f:
        f.write(json.dumps(exact_u_b_time))


    # baseline
    # base_start = time.perf_counter()
    # join_1_f = join_1.f_join()
    # join_2_f = join_2.f_join()
    # join_3_f = join_3.f_join()
    # cur_time = time.perf_counter()
    # print(f"join in {cur_time - base_start:0.4f} seconds")
    # full_frames = [join_1_f, join_2_f, join_3_f]
    # join_f = pd.concat(full_frames).drop_duplicates()
    # base_sample_result = sample_from_join(join_f, n)
    # base_end = time.perf_counter()
    # print(f"baseline in {base_end - base_start:0.4f} seconds")


if __name__ == '__main__':
    main()