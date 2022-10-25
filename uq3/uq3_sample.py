import pickle
import pandas as pd
import numpy as np
from build_hash import *
from acyclic_join import *
from uq3_direct_overlap import *
from olken_single import *
from exact_single import *


def olken_sample_from_disjoint(js, n, hss):
    time_store = []

    S = pd.DataFrame()
    N = len(js)

    # store join size
    J = np.zeros(N)
    for i in range(0, N):
        J[i] = e_size(js[i])
    
    C = np.sum(J)

    # probability of choosing J_i
    P = np.zeros(N)

    weight_start = time.perf_counter()
    ws = []
    for i in range(N):
        P[i] = J[i] / C
        ws.append(olkens_store_ws(js[i], hss[i]))
    weight_end = time.perf_counter()

    print("weights updated in ", weight_end - weight_start, " s")

    f = open("./uq3_3_3/olkens_weights.pkl","wb")
    pickle.dump(ws,f)
    f.close()

    print("successfully stored")
    
    # ws = pickle.load(open("./uq3_3_3/olkens_weights.pkl", "rb"))
    # print("weights successfully loaded")

    sample_start = time.perf_counter()
    
    it = 0
    its = []
    while S.shape[0] < n:
        it+=1
        j = np.random.choice(np.arange(0, N), p = P)
        # print("join path: ", j)
        # print(js[i].tables[2])
        ts = olken_sample_from_s_join(js[j], hss[j], ws[j], J[j])
        if len(ts) == len(js[j].tables):
            result = ts[0]
            for i in range(1,len(js[j].tables)):
                result = pd.merge(result, ts[i], on = js[j].keys[i-1], how = 'inner')
            S = pd.concat([S, result])
            its.append(it)
            if(len(S) % 100 == 0):
                cur_time = time.perf_counter()
                time_store.append(cur_time - sample_start)
            # print(S.shape[0])
        else:
            continue
        
    print("iterations: ", it)
    
    return S, time_store


def exact_sample_from_disjoint(js, n, hss):
    time_store = []

    S = pd.DataFrame()
    N = len(js)

    # store join size
    J = np.zeros(N)

    for i in range(0, N):
        J[i] = e_size(js[i])
    
    C = np.sum(J)

    # probability of choosing J_i
    P = np.zeros(N)

    weight_start = time.perf_counter()
    ws = []
    for i in range(N):
        P[i] = J[i] / C
        ws.append(exact_store_ws(js[i], hss[i]))
    
    weight_end = time.perf_counter()
    print("weights updated in ", weight_end - weight_start, " s")

    f = open("./uq3_3_3/exact_weights.pkl","wb")
    pickle.dump(ws,f)
    f.close()

    print("successfully stored")
    
    # ws = pickle.load(open("./uq3_3_3/exact_weights.pkl", "rb"))
    # print("weights successfully loaded")

    sample_start = time.perf_counter()

    while S.shape[0] < n:
        j = np.random.choice(np.arange(0, N), p = P)
        # print("join path: ", j)
        # print(js[i].tables[2])
        ts = exact_sample_from_s_join(js[j], hss[j], ws[j])
        # print(ts)
        result = ts[0]
        for i in range(1,len(ts)):
            result = pd.merge(result, ts[i], on = js[j].keys[i-1], how = 'inner')
        # print(result)
        S = pd.concat([S, result])
        if(len(S) % 100 == 0):
            cur_time = time.perf_counter()
            time_store.append(cur_time - sample_start)
        # print(S.shape[0])
    
    return S, time_store

def olken_olken_sample_union_bernoulli(js, n, hss, norm_js):
    time_store = []

    S = pd.DataFrame()
    N = len(js)

    # store join size
    J = np.zeros(N)
    for i in range(0, N):
        J[i] = e_size(js[i])
    
    U = uq3_calc_U(js, norm_js)
    print("U calculated: ", U)

    # probability of choosing J_i
    P = np.zeros(N)

    # weight_start = time.perf_counter()
    # ws = []
    for i in range(N):
        P[i] = J[i] / U
    # # print(P)
    #     ws.append(olkens_store_ws(js[i], hss[i]))
    # weight_end = time.perf_counter()

    # print("weights updated in ", weight_end - weight_start, " s")

    # f = open("./uq3_3_3/olkens_weights.pkl","wb")
    # pickle.dump(ws,f)
    # f.close()

    # print("successfully stored")
    
    ws = pickle.load(open("./uq3_3_3/olkens_weights.pkl", "rb"))
    print("weights successfully loaded")

    sample_start = time.perf_counter()

    first_seen = []
    keep = True
    first = True

    while S.shape[0] < n:
        round_record = []
        for j in range(len(js)):
            result = pd.DataFrame()
            p_j = P[j]
            r_j = random.random()
            if r_j > p_j:
                # print("next")
                continue
            else:
                check = True
                fail = False
                while (check):
                    ts = olken_sample_from_s_join(js[j], hss[j], ws[j], J[j])
                    check = False
                    if len(ts) != len(js[j].tables):
                        fail = True
                        break
                    result = ts[0]
                    for i in range(1,len(ts)):
                        result = pd.merge(result, ts[i], on = js[j].keys[i-1], how = 'inner')
                    for t in round_record:
                        if result.equals(t):
                            check = True 
                # print(result)
                if not fail:
                    for index, row in S.iterrows():  
                        if result.equals(row):
                            if first_seen[index] == j :
                                first = False
                                round_record.append(result)
                            else: 
                                keep = False
                            break
                
                    if (keep and first):
                        first_seen.append(j)
                        round_record.append(result)

                    # print("keep ", keep)
                    # print("first ", first)
                
        if (S.shape[0] + len(round_record) > n):
            # print("exceed")
            round_record.clear()
            
        for tuple in round_record:
            S = pd.concat([S, tuple])
            if(len(S) % 100 == 0):
                cur_time = time.perf_counter()
                time_store.append(cur_time - sample_start)

        # print(S.shape[0])
        round_record.clear()    
    
    return S, time_store


def exact_olken_sample_union_bernoulli(js, n, hss, norm_js):
    time_store = []

    S = pd.DataFrame()
    N = len(js)

    # store join size
    J = np.zeros(N)
    for i in range(0, N):
        J[i] = e_size(js[i])
    
    U = uq3_calc_U(js, norm_js)

    # probability of choosing J_i
    P = np.zeros(N)

    # weight_start = time.perf_counter()
    # ws = []
    for i in range(N):
        P[i] = J[i] / U
    # # print(P)
    #     ws.append(exact_store_ws(js[i], hss[i]))
    # weight_end = time.perf_counter()

    # print("weights updated in ", weight_end - weight_start, " s")

    # f = open("./tpch_3_chain_5/exact_weights.pkl","wb")
    # pickle.dump(ws,f)
    # f.close()

    # print("successfully stored")
    
    ws = pickle.load(open("./uq3_3_3/exact_weights.pkl", "rb"))
    print("weights successfully loaded")

    sample_start = time.perf_counter()

    first_seen = []
    keep = True
    first = True

    while S.shape[0] < n:
        round_record = []
        for j in range(len(js)):
            result = pd.DataFrame()
            p_j = P[j]
            r_j = random.random()
            if r_j > p_j:
                # print("next")
                continue
            else:
                check = True
                while (check):
                    ts = exact_sample_from_s_join(js[j], hss[j], ws[j])
                    check = False
                    result = ts[0]
                    for i in range(1,len(js[j].tables)):
                        result = pd.merge(result, ts[i], on = js[j].keys[i-1], how = 'inner')
                    for t in round_record:
                        if result.equals(t):
                            check = True 
                    else:
                        continue
                # print(result)
                for index, row in S.iterrows():  
                    if result.equals(row):
                        if first_seen[index] == j :
                            first = False
                            round_record.append(result)
                        else: 
                            keep = False
                        break
                
                if (keep and first):
                    first_seen.append(j)
                    round_record.append(result)

                # print("keep ", keep)
                # print("first ", first)
                
        if (S.shape[0] + len(round_record) > n):
            # print("exceed")
            round_record.clear()
            
        for tuple in round_record:
            S = pd.concat([S, tuple])
            if(len(S) % 100 == 0):
                cur_time = time.perf_counter()
                time_store.append(cur_time - sample_start)

        # print(S.shape[0])
        round_record.clear()    
    
    return S, time_store






