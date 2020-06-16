#!/usr/bin/python
# -*- coding: utf-8 -*-

from model import *
from util import *
from tqdm import tqdm
import sys
import argparse
import json

class Config(object):
    """Configuration of model"""
    batch_size = 16
    test_batch_size = 16
    embedding_size = 16
    hidden_size = 32
    num_epochs = 10
    max_length = 30

    n_time_interval = 40
    max_time = 120
    time_unit = 1  # 3600 for Meme；1 for Weibo and Twitter

    l2_weight = 5e-5
    dropout = 0.8
    patience = 5
    freq = 5
    gpu_no = '0'
    model_name = 'hidan'
    data_name = 'data/twitter'
    learning_rate = 0.001
    optimizer = 'adam'
    random_seed = 1402


class Input(object):
    def __init__(self, config, data, train=True):
        self.batch_size = config.batch_size
        self.test_batch_size = config.test_batch_size
        self.num_nodes = config.num_nodes
        self.max_length = config.max_length
        self.n_time_interval = config.n_time_interval
        self.max_time = config.max_time
        self.time_unit = config.time_unit
        if train==True:
            self.inputs, self.targets, self.time_interval_index, self.seq_lenghth = batch_generator_withtime(data, self.batch_size, self.max_length, self.n_time_interval, self.max_time, self.time_unit)
        else:
            self.inputs, self.targets, self.time_interval_index, self.seq_lenghth = batch_generator_withtime(data, self.test_batch_size, self.max_length, self.n_time_interval, self.max_time, self.time_unit)
        self.batch_num = len(self.inputs)
        self.cur_batch = 0
        print("____INPUT_____")
        print(self.batch_num, self.cur_batch)
    def next_batch(self):
        x = self.inputs[self.cur_batch]
        y = self.targets[self.cur_batch]
        sl = self.seq_lenghth[self.cur_batch]
        tii = self.time_interval_index[self.cur_batch]
        self.cur_batch = (self.cur_batch +1) % self.batch_num
#         print("NEXT___BATCH")
#         print(self.cur_batch, len(x),len(y),len(tii),len(sl))
        return x, y, tii, sl

def args_setting(config):
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lr", type=float, help="learning rate")
    parser.add_argument("-x", "--edim", type=int, help="embedding dimension")
    parser.add_argument("-e", "--hdim", type=int, help="hidden dimension")
    parser.add_argument("-d", "--data", help="data name")
    parser.add_argument("-g", "--gpu", help="gpu id")
    parser.add_argument("-b", "--bs", type=int, help="batch size")
    parser.add_argument("-t", "--tu", type=float, help="time unit")
    args = parser.parse_args()
    if args.lr:
        config.learning_rate = args.lr
    if args.edim:
        config.embedding_size = args.edim
    if args.hdim:
        config.hidden_size = args.hdim
    if args.bs:
        config.batch_size = args.bs
    if args.data:
        config.data_name = args.data
    if args.gpu:
        config.gpu_no = args.gpu
    if args.tu:
        config.time_unit = args.tu
    return config

def train():
    config = Config()
    config = args_setting(config)
    data_name = config.data_name
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_no
    train_data, valid_data, test_data, nodes, node_to_id = \
        read_raw_data_withtime(data_name + '-cascades')

    config.num_nodes = len(nodes)
    train_size = train_data[-2]
    valid_size = valid_data[-2]
    test_size = test_data[-2]
    print ("---SIZE---",train_size, valid_size, test_size)
#     print("------TIME---", train_data[-1],valid_data[-1],test_data[-1])

    num_epochs = config.num_epochs

    input_train = Input(config, train_data, True)
    input_valid = Input(config, valid_data, False)
    input_test = Input(config, test_data, False)

    model = Model(config)
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    model.build_model()
    tf.set_random_seed(config.random_seed)
    sess.run(tf.global_variables_initializer())

    print('Start training...')

    max_logits = float('inf')
    stop_count = 0
    best_mrr = 0
    best_macc1 = 0
    best_macc5 = 0
    best_macc10 = 0
    best_macc50 = 0
    best_macc100 = 0
    test_scores_final=[]

    for epoch in range(num_epochs):
        epoch_cll = 0

        valid_cll = 0
        valid_ill = 0
        valid_mrr = 0
        valid_macc1 = 0
        valid_macc5 = 0
        valid_macc10 = 0
        valid_macc50 = 0
        valid_macc100 = 0

        test_cll = 0
        test_ill = 0
        test_mrr = 0
        test_macc1 = 0
        test_macc5 = 0
        test_macc10 = 0
        test_macc50 = 0
        test_macc100 = 0
        test_scores = []

        train_info = "Data: {0:>3}, Model: {1:>3}, Learning Rate: {2:>3.5f}, Embedding Size: {3:>3.0f}, Hidden Size: {4:>3.0f}"
        print(train_info.format(config.data_name, config.model_name, config.learning_rate, config.embedding_size, config.hidden_size))

        for i in tqdm(range(input_train.batch_num)):
            epoch_cll += model.train_batch(sess, input_train.next_batch())
            # epoch_nll += model.train_exam(sess, input_train.next_batch())

        msg = "Epoch: {0:>1}, Train CLL: {1:>6.5f}"
        print(msg.format(epoch + 1, epoch_cll/float(train_size)))

        if (epoch+1)%config.freq==0:

            for j in tqdm(range(input_valid.batch_num)):
                batch_cll, batch_ill, mrr, macc1, macc5, macc10, macc50, macc100, _ = model.test_batch(sess, input_valid.next_batch())
                valid_cll += batch_cll
                valid_ill += batch_ill
                valid_mrr += mrr
                valid_macc1 += macc1
                valid_macc5 += macc5
                valid_macc10 += macc10
                valid_macc50 += macc50
                valid_macc100 += macc100

            for k in tqdm(range(input_test.batch_num)):
                batch_cll, batch_ill, mrr, macc1, macc5, macc10, macc50, macc100, scores = model.test_batch(sess, input_test.next_batch(), test_batch_perf=True)
                test_cll += batch_cll
                test_ill += batch_ill
                test_mrr += mrr
                test_macc1 += macc1
                test_macc5 += macc5
                test_macc10 += macc10
                test_macc50 += macc50
                test_macc100 += macc100
                test_scores.append(scores)
            
            test_scores_final.append(test_scores)    

            msg = "Epoch: {0:>1}, Valid CLL: {1:>6.5f}, Valid ILL: {2:>6.5f}, Test CLL: {3:>6.5f}, Test ILL: {4:>6.5f}"
            print(msg.format(epoch + 1, valid_cll/float(valid_size), valid_ill/float(valid_size), test_cll/float(test_size), test_ill/float(test_size)))

            msg = "Valid Results MRR: {0:>6.5f}, ACC1: {1:>6.5f}, ACC5: {2:>6.5f}, ACC10: {3:>6.5f}, ACC50: {4:>6.5f}, , ACC100: {5:>6.5f}"
            print(msg.format( valid_mrr/float(valid_size), valid_macc1/float(valid_size), valid_macc5/float(valid_size), valid_macc10/float(valid_size), valid_macc50/float(valid_size), valid_macc100/float(valid_size)  ))

            msg = "Test Results MRR: {0:>6.5f}, ACC1: {1:>6.5f}, ACC5: {2:>6.5f}, ACC10: {3:>6.5f}, ACC50: {4:>6.5f}, ACC100: {5:>6.5f}"
            print(msg.format( test_mrr/float(test_size), test_macc1/float(test_size), test_macc5/float(test_size), test_macc10/float(test_size), test_macc50/float(test_size), test_macc100/float(test_size)  ))

            if valid_cll < max_logits:
                max_logits = valid_cll
                best_mrr = test_mrr
                best_macc1 = test_macc1
                best_macc5 = test_macc5
                best_macc10 = test_macc10
                best_macc50 = test_macc50
                best_macc100 = test_macc100
                stop_count = 0
            else:
                stop_count += 1

            if stop_count>=config.patience:
                msg = "Best Results MRR: {0:>6.5f}, ACC1: {1:>6.5f}, ACC5: {2:>6.5f}, ACC10: {3:>6.5f}, ACC50: {4:>6.5f}, ACC100: {5:>6.5f}"
                print(msg.format(best_mrr/float(test_size), best_macc1/float(test_size), best_macc5/float(test_size), best_macc10/float(test_size), best_macc50/float(test_size), best_macc100/float(test_size) ))
                break

    with open(data_name + '_res.txt', 'a') as f:
        f.write(config.model_name + ':\n')
        f.write('MRR: '+ str(best_mrr/float(test_size)) + '\n')
        f.write('ACC1: '+ str(best_macc1/float(test_size)) + '\n')
        f.write('ACC5: '+ str(best_macc5/float(test_size)) + '\n')
        f.write('ACC10: '+ str(best_macc10/float(test_size)) + '\n')
        f.write('ACC50: '+ str(best_macc50/float(test_size)) + '\n')
        f.write('ACC100: '+ str(best_macc100/float(test_size)) + '\n')
    print('Finish training...')
#     print(test_scores_final)
#     print(len(test_scores_final), len(test_scores_final[0]))
    scores_best = {}
    scores_best["tau"]=0.0
    scores_best["row"]=0.0
    scores_best["hits@1"] = 0.0
    scores_best["hits@5"] = 0.0
    scores_best["hits@10"] = 0.0
    scores_best["hits@20"] = 0.0
    scores_best["hits@50"] = 0.0
    scores_best["hits@100"] = 0.0
    scores_best["map@1"] = 0.0
    scores_best["map@5"] = 0.0
    scores_best["map@10"] = 0.0
    scores_best["map@20"] = 0.0
    scores_best["map@50"] = 0.0
    scores_best["map@100"] = 0.0
    count = 0
    for each_batch in test_scores_final[-1]:
        scores_best["tau"]+=each_batch["tau"]
        scores_best["row"]+=each_batch["row"]
        scores_best["hits@1"]+=each_batch["map@1"]
        scores_best["hits@5"]+=each_batch["map@5"]
        scores_best["hits@10"]+=each_batch["map@10"]
        scores_best["hits@20"]+=each_batch["map@20"]
        scores_best["hits@50"]+=each_batch["map@50"]
        scores_best["hits@100"]+=each_batch["map@100"]
        scores_best["map@1"]+=each_batch["map@1"]
        scores_best["map@5"]+=each_batch["map@5"]
        scores_best["map@10"]+=each_batch["map@10"]
        scores_best["map@20"]+=each_batch["map@20"]
        scores_best["map@50"]+=each_batch["map@50"]
        scores_best["map@100"]+=each_batch["map@100"]
        count+=1
        
    for each_eval in scores_best:
        scores_best[each_eval] = scores_best[each_eval]/count
    scores_best["MRR"] = best_mrr/float(test_size)
    scores_best["ACC1"] = best_macc1/float(test_size)
    scores_best["ACC5"] = best_macc5/float(test_size)
    scores_best["ACC10"] = best_macc10/float(test_size)
    scores_best["ACC50"] = best_macc50/float(test_size)
    scores_best["ACC100"] = best_macc100/float(test_size)
    
    scores_best["batch_size"] = config.batch_size
    scores_best["test_batch_size"] = config.test_batch_size
    scores_best["embedding_size"] = config.embedding_size
    scores_best["hidden_size"] = config.hidden_size
    scores_best["num_epochs"] = config.num_epochs
    scores_best["freq"] = config.freq
    
    print(scores_best)
    with open("final_results.json","w") as f:
         json.dump(scores_best,f, indent=True)
    sess.close()

if __name__ == '__main__':
    train()
