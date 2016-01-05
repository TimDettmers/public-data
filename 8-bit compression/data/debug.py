'''
Created on Jun 20, 2015

@author: tim
'''
import numpy as np
from util3 import Util
np.set_printoptions(suppress=True, precision =5)
u = Util()

x = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

expo = np.array([10,1,0.1,0.01,0.001,0.0001,0.00001,0.000001])

def get_branches(number, lower, upper):
    return [(number-((number-lower)/2.0), lower, number),(number+((upper-number)/2.0), number, upper)]

def get_all_branches(number, lower, upper, depth, numbers):    
    depth += 1
    for branch in get_branches(number, lower, upper):
        number, lower, upper =  branch
        numbers.append(number)
        if depth < 3: 
            get_all_branches(number, lower, upper, depth, numbers)
    return numbers
    
 
numbers = get_all_branches(0.45,0.1,0.99,0,[])
numbers.append(0.45)
numbers.append(0.1)
numbers.sort()
numbers = np.array(numbers)

data = []
for num in expo:
    data += (num*numbers).tolist()
    
print np.array(data).shape
data.sort()
print (np.array(data)*10).tolist()
      
''' 
branches = []
branches += get_branches(0.45, 0.1, 0.99)
print branches
for branch in branches:
    print get_branches(branch[0], branch[1], branch[2])
    
'''