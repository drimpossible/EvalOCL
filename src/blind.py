import multiprocessing
import numpy as np
from scipy import stats
from os.path import exists


def load_filelist(filepath):
    imglist, y = [], []
    with open(filepath,'r') as f:
        for line in f:
            label, _, _ = line.strip().split('\t')
            y.append(int(label)) 
    return imglist, y


def update_mode(blind_clf, y_new, y_old, y_mode):     # TODO: Simple implementation here, optimize the mode calculation by a heap for extremely large k at the expense of readability?
    blind_clf[y_new] += 1
    blind_clf[y_old] -= 1
    
    if y_new == y_mode or blind_clf[y_new] >= blind_clf[y_mode]: 
        mode = y_new
    elif y_old == y_mode:
        mode = max(blind_clf, key=blind_clf.get)
    else:
        mode = y_mode    
    return blind_clf, mode


def get_preds(labels, shift, k):
    blind_clf = {}
    num_cls = np.unique(labels).shape[0]
    pred = np.ones_like(labels)[shift:]

    for i in range(num_cls):
        blind_clf[i] = 0
    blind_clf[-1] = 0

    mode = labels[0]

    for i in range(shift, shift+k):
        blind_clf, mode = update_mode(blind_clf=blind_clf, y_new=labels[i-1], y_old=-1, y_mode=mode)
        pred[i] = mode

    for i in range(shift+k, len(labels)):
        blind_clf, mode = update_mode(blind_clf=blind_clf, y_new=labels[i-1], y_old=labels[i-1-k], y_mode=mode)
        pred[i] = mode
        
    np.save(LOGDIR+'/shift_'+dataset+'_blind_preds_'+str(k)+'_'+str(shift)+'.npy', pred)
    return pred


if __name__ == '__main__':
    dataset='cglm'
    #dataset='cloc'

    ORDERFILEDIR='../data/'+dataset+'/' 
    LOGDIR='../data/blind/'

    _, labels = load_filelist(filepath=ORDERFILEDIR+'/train.txt')
    labels = np.array(labels, dtype='u2')
    pred = np.ones_like(labels)[1:]

    # Get dataset mode
    modelabel = stats.mode(labels)[0][0]
    pred = pred*modelabel
    np.save(LOGDIR+'/'+dataset+'_mode.npy', pred)

    del pred, modelabel

    p = multiprocessing.Pool(16)
    result = []
    for k in [1, 2, 5, 10, 20, 30, 40, 60, 75, 100, 250, 500, 1000, 5000, 25000, 75000]:
        for shift in [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384, 32768, 65536, 131072]:
            print(k, shift)
            
            if not exists(LOGDIR+'/shift_'+dataset+'_blind_preds_'+str(k)+'_'+str(shift)+'.npy'): 
                result.append(p.apply_async(get_preds, [labels,shift,k]))

    for r in result:
        r.wait()

    print('Extracted all blind classifier results!')
