'''
Created on May 2, 2015

@author: tim
'''
import numpy as np
from util3 import Util
import matplotlib.pyplot as plt
from scipy import stats

u = Util()

y = np.random.randint(0,15000,(70000,))

#y = u.create_t_matrix(y)
y = np.float32(np.array(y)).reshape(70000,1)


u.save_hdf5_matrix('/home/tim/data/mnist/y_15000.hdf5', y)

def get_data_sets(path):
    train = []
    cv = []
    files = u.get_files_in_path(path)
    n = len(files)    
    for f in files:
        if 'cv' not in f:            
            train.append(u.load_hdf5_matrix(f))
        else:
            cv.append(u.load_hdf5_matrix(f))
            
    train =  np.vstack(train)
    cv =  np.vstack(cv)  
    return [cv]

def get_interval_data_sets(path):
    train = []
    cv = []
    files = u.get_files_in_path(path)
    n = len(files)
    for f in files:
        if 'cv' not in f:
            train.append(u.load_hdf5_matrix(f))
        else:
            cv.append(u.load_hdf5_matrix(f))
            
    train =  np.mean(np.vstack(train).T*100,1)
    cv =  np.mean(np.vstack(cv).T*100,1)
    return [train,cv]



def get_95_interval(data):
    x = np.mean(np.min(np.vstack(np.array(data[0])),1))
    se = 1.96*np.std(np.min(np.vstack(np.array(data[0])),1))/np.sqrt(5)
    print [x-se,x+se]
    
def get_99_interval(data):
    x = np.mean(np.min(np.vstack(np.array(data)),1))    
    se = 2.57*np.std(np.min(np.vstack(np.array(data[0])),1))/np.sqrt(5)
    print [x-se,x+se]
    
def t_test(A, B):
    A = np.min(np.vstack(np.array(A)),1)
    B = np.min(np.vstack(np.array(B)),1)
    print 'var A: {0}'.format(np.var(A))
    print 'var B: {0}'.format(np.var(B))
    print stats.mstats.normaltest(A)
    print stats.mstats.normaltest(B)
    print stats.ttest_ind(A,B)

get_95_interval(get_data_sets('/home/tim/data/mnist/results/8bit_standard/'))
get_95_interval(get_data_sets('/home/tim/data/mnist/results/8bit_optimal/'))
get_95_interval(get_data_sets('/home/tim/data/mnist/results/8bit/'))
get_95_interval(get_data_sets('/home/tim/data/mnist/results/32bit/'))
get_95_interval(get_data_sets('/home/tim/data/mnist/results/32bit_decay/'))

print '-----------------------------------------'

get_95_interval(get_data_sets('/home/tim/data/mnist/results/8bit_standard_model/'))
get_95_interval(get_data_sets('/home/tim/data/mnist/results/8bit_optimal_model/'))
get_95_interval(get_data_sets('/home/tim/data/mnist/results/8bit_model/'))
get_95_interval(get_data_sets('/home/tim/data/mnist/results/32bit_model/'))
get_95_interval(get_data_sets('/home/tim/data/mnist/results/32bit_decay_model/'))
get_95_interval(get_data_sets('/home/tim/data/mnist/results/8bit_model_100/'))

#t_test(get_data_sets('/home/tim/data/mnist/results/8bit_model/'), get_data_sets('/home/tim/data/mnist/results/8bit_model_50/'))
get_95_interval(get_data_sets('/home/tim/data/mnist/results/8bit_model/'))
get_99_interval(get_data_sets('/home/tim/data/mnist/results/8bit_model/'))


A = get_data_sets('/home/tim/data/mnist/results/32bit/')+get_data_sets('/home/tim/data/mnist/results/32bit_model/')
B = get_data_sets('/home/tim/data/mnist/results/32bit_decay/')+get_data_sets('/home/tim/data/mnist/results/32bit_decay_model/')

print len(A)
print len(B)

#t_test(A,B)

get_99_interval(A)
get_99_interval(B)

data = get_data_sets('/home/tim/data/mnist/results/8bit_standard/')

data = data + get_data_sets('/home/tim/data/mnist/results/8bit_optimal/')
data = data + get_data_sets('/home/tim/data/mnist/results/8bit/')
data = data + get_data_sets('/home/tim/data/mnist/results/32bit/')
data = data + get_data_sets('/home/tim/data/mnist/results/32bit_decay/')

print np.array(data).shape
u.plot_error_bar_diagram(data, '/home/tim/data/mnist/results/', skip=0,title="Average data parallel error MNIST")


data = get_data_sets('/home/tim/data/mnist/results/8bit_standard/')

data = data + get_data_sets('/home/tim/data/mnist/results/8bit_optimal_model/')
data = data + get_data_sets('/home/tim/data/mnist/results/8bit_model/')
data = data + get_data_sets('/home/tim/data/mnist/results/32bit_model/')
data = data + get_data_sets('/home/tim/data/mnist/results/32bit_decay_model/')

u.plot_error_bar_diagram(data, '/home/tim/data/mnist/results/', skip=0,title="")


'''
data = get_data_sets('/home/tim/data/mnist/results/32bit_decay/')
print np.min(data[0])
print np.min(np.vstack(np.array((data[0]))),1)
print data[0].shape
data = data + get_data_sets('/home/tim/data/mnist/results/32bit_without_decay/')
print np.min(get_data_sets('/home/tim/data/mnist/results/32bit_without_decay/')[1])
print np.min(np.vstack(np.array((get_data_sets('/home/tim/data/mnist/results/32bit_without_decay/')[1]))),1)
u.plot_error_bar_diagram(data, '/home/tim/data/mnist/results/', skip=0,title="Dropout decay on MNIST")
'''

#0.0106804 0.0113726
#data = get_data_sets('/home/tim/data/mnist/results/8bit_standard/')
#data = data + get_data_sets('/home/tim/data/mnist/results/8bit/')
#u.plot_error_bar_diagram(data, '/home/tim/data/mnist/results/', skip=0,title="MNIST 32-bit vs 8-bit classifcation error")


'''
#1024x1024 data
bits32 = np.array([5.8,2.5,1.51,0.79,0.44,0.28,0.215,0.16])
bits8 = np.array([2.5,1.3,0.705,0.43,0.28,0.21,0.19,0.16])
baseline = np.array([2.75,1.50,0.960,0.720,0.575,0.53,0.490,0.467])
'''
'''
#2048x2048 data
bits32 = np.array([16.7,8.5,3.7,2.19,1.15,0.74,0.53,0.4])
bits8 = np.array([6.2,3.3,1.83,1.06,0.68,0.47,0.435,0.36])
baseline = np.array([5.04,2.84,2.02,1.6,1.33,1.17,1.12,1.09])
bits32_2 = np.array([5.5,3.0,1.82,1.17,0.84,0.65,0.635,0.67])
bits8_2 = np.array([3.52,2.1,1.39,1.0,0.8,0.71,0.705,0.62])
'''

'''
#2048x2048x2048x2048x2048x2048x2048 data
bits32 = np.array([75,38,19,9.8,5.6,3.2,2.2,1.65])
bits8 = np.array([28,14.6,7.9,4.6,2.9,2.0,1.75,1.55])
baseline = np.array([9.0,7.7,5.5,5.3,5.0,4.9,5.15,5.1])
bits32_2 = np.array([24.2,12.9,7.4,4.7,3.5,3.05,2.97,2.85])
bits8_2 = np.array([13,8.4,5.74,4.18,3.3,2.91,2.91,2.83])
'''

'''
#2048x2048 data python
bits32 = np.array([14.88,7.5,3.85,2.01,1.13,0.67,0.48])
bits8 = np.array([6.09,3.12,1.67,0.96,0.60,0.41,0.34])
baseline = np.array([1.86,1.56,1.26,1.065,0.99,0.99,1.02])
'''
'''
#2048x2048x2048x2048x2048x2048x2048 data python
bits32 = np.array([66,32.8,17.1,9.13,5.05,2.90,2.02])
bits8 = np.array([26.77,13.65,7.4,4.3,2.72,1.88,1.53])
baseline = np.array([9,6.86,5.45,4.6,4.3,4.23,4.45])


x = np.array([32,64,128,256,512,1024,2048])*4
plt.figure()
plt.ylim([0,3])
plt.plot(x,bits8,color='orange',label='4 GPU 8-bit')
plt.plot(x,bits32,color='black',label='4 GPU 32-bit')     
plt.plot(x,baseline,color='red',label='1 GPU')     
#plt.title("Simplest errorbars, 0.2 in x, 0.4 in y")

plt.legend()




fig = plt.gcf()
fig.suptitle('Mini-batch vs. speedup', fontsize=20)
plt.ylabel('Time per epoch', fontsize=16)
plt.xlabel('Mini-batch size', fontsize=18)

plt.savefig('/home/tim/data/mnist/results/2048_7_batch_size_vs_speed.png')

plt.figure()
plt.ylim([0,5])
plt.plot(x,baseline/bits8,color='orange',label='4 GPU 8-bit')
plt.plot(x,baseline/bits32,color='black',label='4 GPU 32-bit')     

plt.legend()

fig = plt.gcf()
fig.suptitle('Mini-batch vs. speedup', fontsize=20)
plt.ylabel('Speedup', fontsize=16)
plt.xlabel('Mini-batch size', fontsize=18)

plt.savefig('/home/tim/data/mnist/results/2048_7_speedup_batchsize_python.png')
'''
'''
fig = plt.figure()
ax = fig.add_subplot(111)

bits32 = np.array([11.1,15.5,25.3,45.9,104.8,184,362,432])
bits8 = np.array([11.3,13.6,19.8,35,80.6,123,245,314])
speedup = bits32/bits8

bits32_4 = np.array([22.3,29.2,45.7,74.7,147,240,393,474])
bits8_4 = np.array([22.5,24.7,30.8,46.4,84,140,205,256])
speedup_4 = bits32_4/bits8_4

x = np.array([128,256,512,1024,2048,3072,4096,5120])
plt.ylim([0,700])
lns1 = ax.plot(x,bits8,color='green',label='2 GPU 8-bit time')
lns2 = ax.plot(x,bits32,color='blue',label='2 GPU 32-bit time')  
lns3 = ax.plot(x,bits8_4,'r--',color='green',label='4 GPU 8-bit time')
lns4 = ax.plot(x,bits32_4,'r--',color='blue',label='4 GPU 32-bit time')  
ax2 = ax.twinx()
lns5 = ax2.plot(x,speedup,color='red',label='2 GPU Speedup')  
lns6 = ax2.plot(x,speedup_4,'r--',color='red',label='4 GPU Speedup')       
#plt.title("Simplest errorbars, 0.2 in x, 0.4 in y")

ax.legend(loc=0)

ax2.set_ylim(0, 3)

ax.set_ylabel('Time per dot product in ms', fontsize=16)
ax.set_xlabel('Dimension of N', fontsize=18)
ax2.set_ylabel('Speedup', fontsize=16)

lns = lns1+lns2+lns3+lns4+lns5+lns6
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

fig = plt.gcf()
fig.suptitle('Matrix dot product: dot(256xN,NxN) 32-bit vs 8-bit', fontsize=20)


plt.savefig('/home/tim/Dropbox/research_data/model_parallelism_2GPUs.png')
'''


bits32 = np.array([0.79,0.785,0.82,2.11,5.5,19.8,75.4,289,670,1189,1856])[0:7]
bits8 = np.array([0.74,0.768,0.8,1.44,3.1,9.7,35,134,310,550,858])[0:7]
#no_avg_bits1 = np.array([1.52,1.5,1.53,2,3.5,8.9,31,113,261,462,730])[0:9]

bits1 = np.array([1.52+0.223,1.5+0.1,1.53+0.061,2+0.057,3.5+0.055,8.9+0.045,31+0.062,113+0.12,261,462,730])[0:7]
x = np.int32(np.round((np.array([8,16,24,128,256,512,1024,2048,3072,4096,5120])[0:7]**2)/1000.))
print x
fig = plt.figure()
ax = fig.add_subplot(111)
plt.ylim([0,1800])
#lns1 = ax.plot(x,bits8,color='green',label='8-bit time')
#lns2 = ax.plot(x,bits32,color='blue',label='32-bit time')  
#lns3 = ax.plot(x,bits1,color='black',label='1-bit time')  

speedup = bits8/bits1
speedup8 = bits32/bits8
speedup32 = bits32/bits1
#lns3 = ax.plot(x,bits8_4,'r--',color='green',label='4 GPU 8-bit time')
#lns4 = ax.plot(x,bits32_4,'r--',color='blue',label='4 GPU 32-bit time')  
#ax2 = ax.twinx()
#lns5 = ax.plot(x,speedup,color='red',label='Speedup 8-bit/1-bit')  
lns6 = ax.plot(x,speedup8,color='blue',label='Speedup 8-bit')   
lns7 = ax.plot(x,speedup32,color='green',label='Speedup 1-bit')       
#plt.title("Simplest errorbars, 0.2 in x, 0.4 in y")

ax.legend(loc=0)

#ax2.set_ylim(0, 4)
#ax.set_xlim(0, 5200)
ax.set_ylim(0.75, 2.5)

ax.set_xlabel('Parameters per layer' , fontsize=18)
#ax.set_ylabel('Time per transfer in ms', fontsize=16)
#ax2.set_ylabel('Speedup', fontsize=16)
ax.set_ylabel('Speedup', fontsize=16)

#lns = lns1+lns2+lns3+lns5+lns6+lns7
lns = lns6+lns7
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='lower right')

ax.set_xticklabels([str(i) + 'k' for i in range(0,1400,200)])



ax.locator_params(nbins=10)

fig = plt.gcf()
#fig.suptitle('Speedup compared to 32-bit transfers', fontsize=20)


plt.savefig('/home/tim/Dropbox/research/8-bit compression/data/bandwidth_raw_32vs8vs1_speedups.png')

'''
t1, cv1 = data = get_interval_data_sets('/home/tim/data/mnist/results/interval1/')
t5, cv5 = data = get_interval_data_sets('/home/tim/data/mnist/results/interval5/')
t25, cv25 = data = get_interval_data_sets('/home/tim/data/mnist/results/interval25/')
t100, cv100 = data = get_interval_data_sets('/home/tim/data/mnist/results/interval100/')
bit32, cvbit32 = data = get_interval_data_sets('/home/tim/data/mnist/results/32bit/')

x = np.arange(t1.shape[0])

print t1.shape
print t5.shape

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ylim([0,1800])
lns1 = ax.plot(x,cv1,color='green',label='Interval: 1')
lns2 = ax.plot(x,cv5,color='orange',label='Interval: 5')
lns3 = ax.plot(x,cv25,color='red',label='Interval: 25')
lns4 = ax.plot(x,cv100,color='blue',label='Interval: 100')
lns5 = ax.plot(x,cvbit32,color='black',label='32-bit')
#lns4 = ax.plot(x,bits32_4,'r--',color='blue',label='4 GPU 32-bit time')       
#plt.title("Simplest errorbars, 0.2 in x, 0.4 in y")



ax.legend(loc=0)


ax.set_xlim(0, 75)
ax.set_ylim(0, 3)

ax.set_ylabel('Time per transfer in ms', fontsize=16)
ax.set_xlabel('Dimension of N', fontsize=18)


lns = lns1+lns2+lns3+lns4+lns5
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

fig = plt.gcf()
fig.suptitle('Transfer times for a NxN matrix', fontsize=20)


plt.savefig('/home/tim/Dropbox/research/8-bit compression/data/32bit_vs_8bit_mnist.png')
'''

