import random, os, threading, math
import numpy as np
import hnswlib
from opts import parse_args_knn

class HNSW_KNN():
    # Adopted from ACM Repository
    def __init__(self, opt):
        self.index = hnswlib.Index(space=opt.search_metric, dim=opt.feat_size)
        self.lock = threading.Lock()
        self.idx2label = {}
        self.cur_idx = 0
        self.dset_size = 1024
        self.index.init_index(max_elements=self.dset_size, ef_construction=opt.HNSW_ef, M=opt.HNSW_M)
        self.num_neighbours = opt.num_neighbours 

    def learn_step(self, X, y):
        assert(X.shape[0]==y.shape[0])

        num_added = X.shape[0]
        start_idx = self.cur_idx
        self.cur_idx += num_added
        
        if self.cur_idx >= self.dset_size - 2:
            with self.lock:
                self.dset_size = pow(2, math.ceil(math.log2(self.cur_idx)))
                self.dset_size *= 4
                self.index.resize_index(self.dset_size)
        
        idx = []
        for label in range(y.shape[0]):
            idx.append(start_idx)
            self.idx2label[start_idx] = y[label]
            start_idx += 1
        
        self.index.add_items(data=X, ids=np.asarray(idx))

    def predict_step(self, X, num_neighbours):
        idx_pred_list, distances = self.index.knn_query(data=X, k=num_neighbours)
        labels = []
        
        for idx_pred in idx_pred_list:
            possible_labels = np.array([self.idx2label[idx] for idx in idx_pred]).astype(int)
            counts = np.bincount(possible_labels)
            label = np.argmax(counts)
            labels.append(label if counts[label] > 1 else possible_labels[0])
        return np.array(labels) 
    

if __name__ == '__main__':
    # Parse arguments and init a simple print logger
    opt = parse_args_knn()
    random.seed(opt.seed)
    os.environ['PYTHONHASHSEED'] = str(opt.seed)
    np.random.seed(opt.seed)
    print('==> Params for this experiment:'+str(opt))

    feats, labels = np.load(opt.log_dir+'/'+opt.exp_name+'/full_features.npy'), np.load(opt.log_dir+'/'+opt.exp_name+'/labels.npy')
    os.makedirs(os.path.join(f'{opt.log_dir}/{opt.exp_name}/', 'online'), exist_ok=True)
    opt.feat_size = feats.shape[1]

    knn = HNSW_KNN(opt=opt)
    predarr, labelarr, acc = np.zeros(labels.shape[0], dtype='u2'), np.zeros(labels.shape[0], dtype='u2'), np.zeros(labels.shape[0], dtype='bool')

    for i in range(feats.shape[0]- opt.delay):
        # Current and delayed feat
        feat_learn = np.expand_dims(feats[i], axis=0)
        feat_pred = np.expand_dims(feats[i+opt.delay], axis=0)

        if i >  opt.delay+1: 
            pred = knn.predict_step(X=feat_pred, num_neighbours=opt.num_neighbours)
            predarr[i+opt.delay] = pred
            labelarr[i+opt.delay] = labels[i+opt.delay]
            is_correct = (int(pred)==int(labels[i+opt.delay]))
            acc[i+opt.delay] = is_correct*1.0
            
        if i%opt.print_freq == 0:
            cum_acc = np.array(acc[:i+opt.delay]).mean()
            print('Step:\t'+str(i)+'\tCumul Acc:\t'+str(cum_acc))

        knn.learn_step(X=feat_learn, y=np.array([labels[i]]))

    np.save(opt.log_dir+'/'+opt.exp_name+'/online/knn_pred_'+str(opt.online_exp_name)+'.npy', predarr)
    np.save(opt.log_dir+'/'+opt.exp_name+'/online/knn_label_'+str(opt.online_exp_name)+'.npy', labelarr)
