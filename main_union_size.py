import pandas as pd
from equi_chain_overlap import *
from acyclic_join import *

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
    nation_sample = nation
    supplier_sample = fixed_sample(supplier, fixed, scale)
    customer_sample = fixed_sample(customer, fixed, scale)
    orders_sample = fixed_sample(orders, fixed, scale)
    lineitem_sample = fixed_sample(lineitem, fixed, scale)

    return supplier_sample,nation_sample,customer_sample,orders_sample,lineitem_sample
    # return lineitem_sample,orders_sample,customer_sample,supplier_sample,nation_sample

def main():
    scale = 0.01
    overlap = 0.005

    n = 0
    k = 0

    supplier_sample_1,nation_sample_1,customer_sample_1,orders_sample_1,lineitem_sample_1 = process_tpch(overlap, scale)
    supplier_sample_2,nation_sample_2,customer_sample_2,orders_sample_2,lineitem_sample_2 = process_tpch(overlap, scale)
    supplier_sample_3,nation_sample_3,customer_sample_3,orders_sample_3,lineitem_sample_3 = process_tpch(overlap, scale)
    supplier_sample_4,nation_sample_4,customer_sample_4,orders_sample_4,lineitem_sample_4 = process_tpch(overlap, scale)

    # lineitem_sample_1,orders_sample_1,customer_sample_1,supplier_sample_1,nation_sample_1 = process_tpch(overlap, 0.01)
    # lineitem_sample_2,orders_sample_2,customer_sample_2,supplier_sample_2,nation_sample_2 = process_tpch(overlap, 0.02)
    # lineitem_sample_3,orders_sample_3,customer_sample_3,supplier_sample_3,nation_sample_3 = process_tpch(overlap, 0.03)
    # lineitem_sample_4,orders_sample_4,customer_sample_4,supplier_sample_4,nation_sample_4 = process_tpch(overlap, 0.04)

    tables_1 = [supplier_sample_1, nation_sample_1, customer_sample_1, orders_sample_1,lineitem_sample_1]
    tables_2 = [supplier_sample_2, nation_sample_2, customer_sample_2, orders_sample_2,lineitem_sample_2]
    tables_3 = [supplier_sample_3, nation_sample_3, customer_sample_3, orders_sample_3,lineitem_sample_3]
    tables_4 = [supplier_sample_4, nation_sample_4, customer_sample_4, orders_sample_4,lineitem_sample_4]
    keys = ['NationKey', 'NationKey', 'CustKey', 'OrderKey']

    # tables_1 = [lineitem_sample_1,orders_sample_1,customer_sample_1,supplier_sample_1,nation_sample_1]
    # tables_2 = [lineitem_sample_2,orders_sample_2,customer_sample_2,supplier_sample_2,nation_sample_2]
    # tables_3 = [lineitem_sample_3,orders_sample_3,customer_sample_3,supplier_sample_3,nation_sample_3]
    # tables_4 = [lineitem_sample_4,orders_sample_4,customer_sample_4,supplier_sample_4,nation_sample_4]
    # keys = ['OrderKey', 'CustKey', 'NationKey', 'NationKey']

    print("step 1 over")

    join_1 = chain_join(tables_1, keys)
    join_2 = chain_join(tables_2, keys)
    join_3 = chain_join(tables_3, keys)
    join_4 = chain_join(tables_4, keys)

    print("step 2 over")

    j_size = [e_size(join_1), e_size(join_2), e_size(join_3), e_size(join_4)]
    print("Estimated join sizes:")
    print(j_size)

    ans, Os = gen_os([join_1, join_2, join_3, join_4])
    print("Estimated overlap sizes: ")
    print(ans)
    print(Os)

    As = calc_As([join_1, join_2, join_3, join_4], Os, ans, j_size)
    print("A:")
    print(As)
    e_union = calc_U(As)
    print("Estimated union size: ")
    print(e_union)

    print("Estimated sum: ")
    print(sum(j_size))

    
    join_1_f = join_1.f_join()
    join_2_f = join_2.f_join()
    join_3_f = join_3.f_join()
    join_4_f = join_4.f_join()
    
    exact_j = [join_1_f.shape[0], join_2_f.shape[0], join_3_f.shape[0], join_4_f.shape[0]]
    print("Exact join sizes:")
    print(exact_j)
    
    exact_o = exact_olp([join_1_f, join_2_f, join_3_f, join_4_f])
    print("Exact overlap sizes:")
    print(exact_o)

    exact_union = pd.concat([join_1_f, join_2_f, join_3_f, join_4_f]).drop_duplicates() 
    print("Exact union size:")
    print(exact_union.shape[0])

    print("Exact sum: ")
    print(sum(exact_j))

    print("Estimated ratio: ", e_union / sum(j_size))
    print("Exact ratio: ", exact_union.shape[0] / sum(exact_j))

if __name__ == '__main__':
    main()