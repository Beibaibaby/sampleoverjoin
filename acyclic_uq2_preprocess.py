import json
import pandas as pd
from equi_chain_overlap import *
from acyclic_join import *
from build_hash import *
import pickle
from sample_from_disjoint import *
from sample_union_bernoulli import *
from acyclic_overlap import *

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


def fake_join_size(t1, t2, key):
    uniq_1 = t1[key].unique()
    uniq_2 = t2[key].unique()
    uniq = uniq_1[np.in1d(uniq_1,uniq_2)]
    size = 0
    for v in uniq:
        size += (t1[t1[key] == v].shape[0] *
            t2[t2[key] == v].shape[0])
    return size
        

def to_q1(tables, keys):
    '''
    return chain query: nation, supplier, customer, orders, lineitem 
    '''
    nation = table_node(tables[0], None, 'NationKey')
    
    supplier = table_node(tables[1], nation, 'NationKey')
    nation.childs.append(supplier)
    
    customer = table_node(tables[2], supplier, 'CustKey')
    supplier.childs.append(customer)
    
    orders = table_node(tables[3], customer, 'OrderKey')
    customer.childs.append(orders)
    
    lineitem = table_node(tables[4], orders, None)
    orders.childs.append(lineitem)

    # hash
    nation.hash_acyc_j()
    
    join_q1 = acyclic_join(nation)
    
    # norm
    
    Ms = []
    for i in range(len(tables)):
        if i == 0:
            Ms.append(0)
        else:
            Ms.append(max_d(tables[i], keys[i-1]))
    
    return join_q1, Ms
    
    
def to_q2(tables, keys):
    '''
    return chain query: nation, (supplier, customer), orders, lineitem 
    '''
    supplier_customer_table = pd.merge(tables[1], tables[2], on = 'NationKey', how = 'inner')
    
    nation = table_node(tables[0], None, 'NationKey')
    
    supplier_customer = table_node(supplier_customer_table, nation, 'CustKey')
    nation.childs.append(supplier_customer)
    
    orders = table_node(tables[3], supplier_customer, 'OrderKey')
    supplier_customer.childs.append(orders)
    
    lineitem = table_node(tables[4], orders, None)
    orders.childs.append(lineitem)
    
    # hash
    nation.hash_acyc_j()
    
    join_q2 = acyclic_join(nation)
    
    # norm
    
    Ms = [0, max_d(supplier_customer_table, keys[0]), 1, max_d(tables[3], keys[2]), max_d(tables[4], keys[3])]
    
    return join_q2, Ms


def to_q3(tables, keys):
    '''
    return acyclic query: nation, supplier, customer, orders1, lineitem1, (orders2, lineitem2)
    '''
    orders_1_table = tables[3].iloc[:int(tables[3].shape[0] / 2),:]
    orders_2_table = tables[3].iloc[int(tables[3].shape[0] / 2):,:]

    lineitem_1_table = tables[4].iloc[:int(tables[4].shape[0] / 2),:]
    lineitem_2_table = tables[4].iloc[int(tables[4].shape[0] / 2):,:]

    nation = table_node(tables[0], None, 'NationKey')

    supplier = table_node(tables[1], nation, 'NationKey')
    nation.childs.append(supplier)
    
    customer = table_node(tables[2], supplier, 'CustKey')
    supplier.childs.append(customer)

    orders_1 = table_node(orders_1_table, customer, 'OrderKey')
    customer.childs.append(orders_1)

    orders_2 = table_node(orders_2_table, customer, 'OrderKey')
    customer.childs.append(orders_2)

    lineitem_1 = table_node(lineitem_1_table, orders_1, None)
    orders_1.childs.append(lineitem_1)

    lineitem_2 = table_node(lineitem_2_table, orders_2, None)
    orders_2.childs.append(lineitem_2)
    
    # hash
    nation.hash_acyc_j()
    
    join_q3 = acyclic_join(nation)
    
    # norm
    # remember to add write up
    m_order = max_d(orders_1_table, 'CustKey') * max_d(lineitem_1_table, 'OrderKey') + max_d(orders_2_table, 'CustKey') * max_d(lineitem_1_table, 'OrderKey')
    
    fake_join_size(orders_1_table, orders_2_table, 'OrderKey')
    
    Ms = [0, max_d(tables[1], keys[0]), max_d(tables[2], keys[1]), m_order, 1]
    return join_q3, Ms


def to_q4(tables, keys):
    '''
    return acyclic query: nation, supplier1, supplier2, customer, orders1, (orders2, lineitem)
    '''
    supplier_1_table = tables[1].iloc[:int(tables[4].shape[0] / 2),:]
    supplier_2_table = tables[1].iloc[int(tables[4].shape[0] / 2):,:]

    orders_1_table = tables[3].iloc[:int(tables[3].shape[0] / 2),:]
    orders_2_table = tables[3].iloc[int(tables[3].shape[0] / 2):,:]
    
    supplier_2 = table_node(supplier_2_table, None, 'NationKey')
    
    nation = table_node(tables[0],  supplier_2, 'NationKey')
    supplier_2.childs.append(nation)

    supplier_1 = table_node(supplier_1_table, nation, 'NationKey')
    nation.childs.append(supplier_1)

    customer = table_node(tables[2], supplier_1, 'CustKey')
    supplier_1.childs.append(customer)

    orders_1 = table_node(orders_1_table, customer, 'OrderKey')
    customer.childs.append(orders_1)

    orders_2 = table_node(orders_2_table, customer, 'OrderKey')
    customer.childs.append(orders_2)

    lineitem_1 = table_node(tables[4], orders_1, None)
    orders_1.childs.append(lineitem_1)

    lineitem_2 = table_node(tables[4], orders_2, None)
    orders_2.childs.append(lineitem_2)
    
    supplier_2.hash_acyc_j()
    
    m_order = max_d(orders_1_table, 'CustKey') + max_d(orders_2_table, 'CustKey') * max_d(tables[4], 'OrderKey')
    
    Ms = [0, max_d(supplier_1_table, keys[0])+max_d(supplier_2_table, keys[0]), max_d(tables[2], keys[1]), m_order, 1]

    join_q4 = acyclic_join(supplier_2)
    return join_q4, Ms

def f_1_join(tables):
    keys = ['NationKey', 'NationKey', 'CustKey', 'OrderKey']
    result = tables[0]
    for i in range(1,len(tables)):
        result = pd.merge(result, tables[i], on = keys[i-1], how = 'inner')
    return result

def f_2_join(tables):
    supplier_customer_table = pd.merge(tables[1], tables[2], on = 'NationKey', how = 'inner')
    keys = ['NationKey', 'CustKey', 'OrderKey']
    new_tables = [tables[0], supplier_customer_table, tables[3], tables[4]]
    result = new_tables[0]
    for i in range(1,len(new_tables)):
        result = pd.merge(result, new_tables[i], on = keys[i-1], how = 'inner')
    return result

def f_3_join(tables):
    '''
    return acyclic query: nation, supplier, customer, orders1, lineitem1, (orders2, lineitem2)
    '''
    orders_1_table = tables[3].iloc[:int(tables[3].shape[0] / 2),:]
    orders_2_table = tables[3].iloc[int(tables[3].shape[0] / 2):,:]

    lineitem_1_table = tables[4].iloc[:int(tables[4].shape[0] / 2),:]
    lineitem_2_table = tables[4].iloc[int(tables[4].shape[0] / 2):,:]
    
    ol2 = pd.merge(orders_2_table, lineitem_2_table, on = 'OrderKey', how = 'inner')
    ol1 = pd.merge(orders_1_table, lineitem_1_table, on = 'OrderKey', how = 'inner')
    
    result = tables[0]
    result = pd.merge(result, tables[1], on = 'NationKey', how = 'inner')
    result = pd.merge(result, tables[2], on = 'NationKey', how = 'inner')
    result = pd.merge(result, ol1, on = 'CustKey', how = 'inner')
    result = pd.merge(result, ol2, on = 'CustKey', how = 'inner')

    return result

def f_4_join(tables):
    '''
    return acyclic query: nation, supplier1, supplier2, customer, orders1, (orders2, lineitem)
    '''
    supplier_1_table = tables[1].iloc[:int(tables[4].shape[0] / 2),:]
    supplier_2_table = tables[1].iloc[int(tables[4].shape[0] / 2):,:]

    orders_1_table = tables[3].iloc[:int(tables[3].shape[0] / 2),:]
    orders_2_table = tables[3].iloc[int(tables[3].shape[0] / 2):,:]
    
    o2l = pd.merge(orders_2_table, tables[4], on = 'OrderKey', how = 'inner')
    
    supp = pd.merge(supplier_2_table, tables[2], on = 'NationKey', how = 'inner')
    supp = pd.merge(supp, orders_1_table, on = 'CustKey', how = 'inner')
    supp = pd.merge(supp, o2l, on = 'CustKey', how = 'inner')
    
    result = tables[0]
    result = pd.merge(result, supplier_1_table, on = 'NationKey', how = 'inner')
    
    result =  pd.merge(result, supp, on = 'NationKey', how = 'inner')
    
    return result

def main():
    scale = 0.005
    overlap = 0.001

    n = 10000
    k = 0

    # Qx: nation, supplier, customer, orders, lineitem 
    # nation = pd.read_table('./tpch_1/nation.tbl', index_col=False, names=['NationKey','NationName','RegionKey','Comment'], delimiter = '|').iloc[:, :-1]
    # supplier = pd.read_table('./tpch_1/supplier.tbl', index_col=False, names=['SuppKey','SuppName','Address','NationKey','Phone','Acctbl','Comment'], delimiter = '|').iloc[:, :-1]
    # customer = pd.read_table('./tpch_1/customer.tbl', index_col=False, names=['CustKey','CustName','Address','NationKey','Phone','Acctbal','MktSegment','Comment'], delimiter = '|').iloc[:, :-1]
    # orders = pd.read_table('./tpch_1/orders.tbl', index_col=False, names=['OrderKey','CustKey','OrderStatus','TotalPrice','OrderDate','OrderPriority','Clerk','ShipPriority','Comment'], delimiter = '|').iloc[:, :-1]
    # lineitem = pd.read_table('./tpch_1/lineitem.tbl', index_col=False, names=['OrderKey','PartKey','SuppKey','LineNumber','Quantity','ExtendedPrice','Discount','Tax','ReturnFlag',
    # 'LineStatus','ShipDate','CommitDate','ReceiptDate','ShipinStruct','ShipMode','Comment'], delimiter = '|').iloc[:, :-1]

    # print("read successfully")
    
    # # supplier.sample(frac=1)
    # # customer.sample(frac=1)
    # # orders.sample(frac=1)
    # # lineitem.sample(frac=1)

    # nation_sample_1,supplier_sample_1,customer_sample_1,orders_sample_1,lineitem_sample_1 = process_tpch(overlap, scale, nation, supplier, customer, orders, lineitem)
    # nation_sample_1.to_csv(r'./uq2_3_4/n1.csv')
    # supplier_sample_1.to_csv(r'./uq2_3_4/s1.csv')
    # customer_sample_1.to_csv(r'./uq2_3_4/c1.csv')
    # orders_sample_1.to_csv(r'./uq2_3_4/o1.csv')
    # lineitem_sample_1.to_csv(r'./uq2_3_4/l1.csv')

    # print("1 done")

    # nation_sample_2,supplier_sample_2,customer_sample_2,orders_sample_2,lineitem_sample_2 = process_tpch(overlap, scale, nation, supplier, customer, orders, lineitem)
    # nation_sample_2.to_csv(r'./uq2_3_4/n2.csv')
    # supplier_sample_2.to_csv(r'./uq2_3_4/s2.csv')
    # customer_sample_2.to_csv(r'./uq2_3_4/c2.csv')
    # orders_sample_2.to_csv(r'./uq2_3_4/o2.csv')
    # lineitem_sample_2.to_csv(r'./uq2_3_4/l2.csv')

    # print("2 done")

    # nation_sample_3,supplier_sample_3,customer_sample_3,orders_sample_3,lineitem_sample_3 = process_tpch(overlap, scale, nation, supplier, customer, orders, lineitem)
    # nation_sample_3.to_csv(r'./uq2_3_4/n3.csv')
    # supplier_sample_3.to_csv(r'./uq2_3_4/s3.csv')
    # customer_sample_3.to_csv(r'./uq2_3_4/c3.csv')
    # orders_sample_3.to_csv(r'./uq2_3_4/o3.csv')
    # lineitem_sample_3.to_csv(r'./uq2_3_4/l3.csv')

    # print("3 done")
    
    # nation_sample_4,supplier_sample_4,customer_sample_4,orders_sample_4,lineitem_sample_4 = process_tpch(overlap, scale, nation, supplier, customer, orders, lineitem)
    # nation_sample_4.to_csv(r'./uq2_3_4/n4.csv')
    # supplier_sample_4.to_csv(r'./uq2_3_4/s4.csv')
    # customer_sample_4.to_csv(r'./uq2_3_4/c4.csv')
    # orders_sample_4.to_csv(r'./uq2_3_4/o4.csv')
    # lineitem_sample_4.to_csv(r'./uq2_3_4/l4.csv')

    # print("4 done")

    nation_sample_1 = pd.read_csv('./uq2_3_4/n1.csv' ,index_col=0)
    supplier_sample_1 = pd.read_csv('./uq2_3_4/s1.csv' ,index_col=0)
    customer_sample_1 = pd.read_csv('./uq2_3_4/c1.csv' ,index_col=0)
    orders_sample_1 = pd.read_csv('./uq2_3_4/o1.csv' ,index_col=0)
    lineitem_sample_1 = pd.read_csv('./uq2_3_4/l1.csv' ,index_col=0)

    nation_sample_2 = pd.read_csv('./uq2_3_4/n2.csv' ,index_col=0)
    supplier_sample_2 = pd.read_csv('./uq2_3_4/s2.csv' ,index_col=0)
    customer_sample_2 = pd.read_csv('./uq2_3_4/c2.csv' ,index_col=0)
    orders_sample_2 = pd.read_csv('./uq2_3_4/o2.csv' ,index_col=0)
    lineitem_sample_2 = pd.read_csv('./uq2_3_4/l2.csv' ,index_col=0)

    nation_sample_3 = pd.read_csv('./uq2_3_4/n3.csv' ,index_col=0)
    supplier_sample_3 = pd.read_csv('./uq2_3_4/s3.csv' ,index_col=0)
    customer_sample_3 = pd.read_csv('./uq2_3_4/c3.csv' ,index_col=0)
    orders_sample_3 = pd.read_csv('./uq2_3_4/o3.csv' ,index_col=0)
    lineitem_sample_3 = pd.read_csv('./uq2_3_4/l3.csv' ,index_col=0)
    
    nation_sample_4 = pd.read_csv('./uq2_3_4/n4.csv' ,index_col=0)
    supplier_sample_4 = pd.read_csv('./uq2_3_4/s4.csv' ,index_col=0)
    customer_sample_4 = pd.read_csv('./uq2_3_4/c4.csv' ,index_col=0)
    orders_sample_4 = pd.read_csv('./uq2_3_4/o4.csv' ,index_col=0)
    lineitem_sample_4 = pd.read_csv('./uq2_3_4/l4.csv' ,index_col=0)


    tables_1 = [nation_sample_1, supplier_sample_1, customer_sample_1, orders_sample_1,lineitem_sample_1]
    tables_2 = [nation_sample_2, supplier_sample_2, customer_sample_2, orders_sample_2,lineitem_sample_2]
    tables_3 = [nation_sample_3, supplier_sample_3, customer_sample_3, orders_sample_3,lineitem_sample_3]
    tables_4 = [nation_sample_4, supplier_sample_4, customer_sample_4, orders_sample_4,lineitem_sample_4]
    keys = ['NationKey', 'NationKey', 'CustKey', 'OrderKey']
    
    tables = [tables_1, tables_2, tables_3, tables_4]
    
    print("step 1 over")

    join_1, M1 = to_q1(tables_1, keys)
    print("1 done")
    
    join_2, M2 = to_q2(tables_2, keys)
    print("2 done")
    
    join_3, M3 = to_q3(tables_3, keys)
    print("3 done")
    
    join_4, M4 = to_q4(tables_4, keys)
    print("4 done")
    
    print("step 2 over")
    
    js = [join_1, join_2, join_3, join_4]
    Ms = [M1, M2, M3, M4]

    # ----------------------------------- Estimation ----------------------------------- 
    
    direct_start = time.perf_counter()
    j_size = [acyc_e_size(join_1), acyc_e_size(join_2), acyc_e_size(join_3), acyc_e_size(join_4)]
    print("Estimated join sizes:")
    print(j_size)

    # ans, Os = gen_os([join_1, join_2, join_3])
    # print("Estimated overlap sizes: ")
    # print(ans)
    # print(Os)

    # As = calc_As([join_1, join_2, join_3], Os, ans, j_size)
    e_union = calc_U(js, tables, Ms, keys)
    print("Estimated union size: ")
    print(e_union)
    
    print("ratio: ")
    print(j_size / e_union)
    
    direct_end = time.perf_counter()
    print(f"direct in {direct_end - direct_start:0.4f} seconds")
    
    
    base_start = time.perf_counter()
    
    join_1_f = f_1_join(tables_1)
    join_2_f = f_2_join(tables_2)
    join_3_f = f_3_join(tables_3)
    join_4_f = f_4_join(tables_3)
    
    exact_j = [join_1_f.shape[0], join_2_f.shape[0], join_3_f.shape[0]]
    print("Exact join sizes:")
    print(exact_j)
    
    # exact_o = exact_olp([join_1_f, join_2_f, join_3_f])
    # print("Exact overlap sizes:")
    # print(exact_o)
    
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

     # ----------------------------------- sample from disjoint ----------------------------------- 
    # exact_S, exact_time = exact_sample_from_disjoint([join_1, join_2, join_3], n)
    # print(exact_time)
    # exact_S.to_csv(r'./uq2_3_4/exact_sample_1000000.csv')
    # with open('./uq2_3_4/exact_time_1000000.txt', 'w') as f:
    #     f.write(json.dumps(exact_time))


    # olken_S, olken_time = olken_sample_from_disjoint([join_1, join_2, join_3], n)
    # print(olken_time)
    # olken_S.to_csv(r'./uq2_3_4/olken_sample_1000000.csv')
    # with open('./uq2_3_4/olken_time_1000000.txt', 'w') as f:
    #     f.write(json.dumps(olken_time))

    # ----------------------------------- sample from union bernoulli  ----------------------------------- 

    # olken_u_b_S, olken_u_b_time = olken_olken_sample_union_bernoulli([join_1, join_2, join_3], n, [hs_1, hs_2, hs_3])
    # print(olken_u_b_time)
    # olken_u_b_S.to_csv(r'./uq2_3_4/olken_u_b_sample_1000000.csv')
    # with open('./uq2_3_4/olken_u_b_time_1000000.txt', 'w') as f:
    #     f.write(json.dumps(olken_u_b_time))

    # exact_u_b_S, exact_u_b_time = exact_olken_sample_union_bernoulli([join_1, join_2, join_3], n, [hs_1, hs_2, hs_3])
    # print(exact_u_b_time)
    # exact_u_b_S.to_csv(r'./uq2_3_4/exact_u_b_sample_1000000.csv')
    # with open('./uq2_3_4/exact_u_b_time_1000000.txt', 'w') as f:
    #     f.write(json.dumps(exact_u_b_time))


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