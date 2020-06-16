import numpy as np
from scipy import stats

def rank_cal(rank_list, target_index):
    rank = 0.
    target_score = rank_list[target_index]
    for score in rank_list:
        if score >= target_score:
            rank += 1.
    return rank

def reciprocal_rank(rank):
    return 1./rank

def accuracy_at_k(rank, k):
    if rank <= k:
        return 1.
    else:
        return 0.

def rank_eval_example(pred, labels):
    mrr = []
    macc1 = []
    macc5 = []
    macc10 = []
    macc50 = []
    cur_pos = []
    for i in range(len(pred)):
        rank = rank_cal(cas_pred[i], cas_labels[i])
        mrr.append(reciprocal_rank(rank))
        macc1.append(accuracy_at_k(rank,1))
        macc5.append(accuracy_at_k(rank,5))
        macc10.append(accuracy_at_k(rank,10))
        macc50.append(accuracy_at_k(rank,50))
    return mrr, macc1, macc5, macc10, macc50

def _flatten_y(y_ori, num_elem):
    y_flat = []
    for i in range(num_elem):
        if i==y_ori:
            y_flat.append(1)
        else:
            y_flat.append(0)
    assert sum(y_flat)==1
    y_flat = np.array(y_flat)
    return y_flat

def _apk(actual, pred,k):
    predicted = np.argsort(pred)[-k:][::-1]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k)

def _hitsk(actual, pred,k=1):
    predicted = np.argsort(pred)[-k:][::-1]
    aucc = 0
    for i in predicted:
        if i in actual:
            aucc+=1
    return aucc/ min(len(actual), k)

def rank_eval(pred, labels, sl, test_batch_perf=False):
    scores={}
    if test_batch_perf:
#         print("---------TEST BATCH-------")
#         print(pred.shape, labels.shape, sl.shape)
        num_test, num_elem = pred.shape
        tau = 0
        row = 0
        k_list = [1,5,10,20,50,100]
        for each_case in range(num_test):
            y_pred = pred[each_case].flatten()
            y_flat = _flatten_y(labels[each_case], num_elem)
            row += stats.spearmanr(y_pred, y_flat)[0]
            tau += stats.kendalltau(y_pred, y_flat)[0]
    #     print("------TAU ROW BATCH-------")
    #     print(tau/num_test, row/num_test)
        scores["tau"] = tau/num_test
        scores["row"] = row/num_test
        for k in k_list:
            hit=0
            for i in range(num_test):
                y_pred = pred[each_case].flatten()
                y_flat = _flatten_y(labels[each_case], num_elem)
                h = _hitsk(y_flat, y_pred,k)
                hit+=h
    #         print("HITS@{} ={}".format(k, hit/num_test))
            scores['hits@' + str(k)] = hit/num_test

        for k in k_list:
            map_=0
            for i in range(num_test):
                y_pred = pred[each_case].flatten()
                y_flat = _flatten_y(labels[each_case], num_elem)
                m = _apk(y_flat, y_pred,k)
                map_+= m
    #         print("MAP@{} ={}".format(k, map_/num_test))
            scores['map@' + str(k)] = map_/num_test  
#         print(scores)        
    mrr = 0
    macc1 = 0
    macc5 = 0
    macc10 = 0
    macc50 = 0
    macc100 = 0
    cur_pos = 0
    for i in range(len(sl)):
        length = sl[i]
        cas_pred = pred[cur_pos : cur_pos+length]
        cas_labels = labels[cur_pos : cur_pos+length]
        cur_pos += length
        rr = 0
        acc1 = 0
        acc5 = 0
        acc10 = 0
        acc50 = 0
        acc100 = 0
        for j in range(len(cas_pred)):
            rank = rank_cal(cas_pred[j], cas_labels[j])
            rr += reciprocal_rank(rank)
            acc1 += accuracy_at_k(rank,1)
            acc5 += accuracy_at_k(rank,5)
            acc10 += accuracy_at_k(rank,10)
            acc50 += accuracy_at_k(rank,50)
            acc100 += accuracy_at_k(rank,100)
        mrr += rr/float(length)
        macc1 += acc1/float(length)
        macc5 += acc5/float(length)
        macc10 += acc10/float(length)
        macc50 += acc50/float(length)
        macc100 += acc100/float(length) 
    return mrr, macc1, macc5, macc10, macc50, macc100, scores