'''
Created on Sep 22, 2015

@author: tim
'''
import cPickle as pickle
import numpy as np


def get_errors(path):

    X = pickle.load(open(path,'r'))

    train_top5 = []
    test_top5 = []
    
    print X['model_state']['train_outputs'][0]
    
    for x in X['model_state']['train_outputs']:
        train_top5.append(x[0]['logprob'][2]/float(x[1]))
        
        
    for x in X['model_state']['test_outputs']:
        test_top5.append(x[0]['logprob'][2]/float(x[1]))
    
    return [train_top5, test_top5]




bit8 = get_errors('/home/tim/data/results/imagenet_8/ConvNet__2015-09-06_14.50.19/90.384')
bit32 = get_errors('/home/tim/data/results/imagenet_32/ConvNet__2015-08-23_08.40.01/90.384')

print np.min(bit8[1])
print np.min(bit32[1])

pickle.dump(bit8, open('/home/tim/data/results/imagenet_8/results.p','wb'))
pickle.dump(bit32, open('/home/tim/data/results/imagenet_32/results.p','wb'))