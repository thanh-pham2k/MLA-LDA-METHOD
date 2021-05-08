import pandas as pd
import numpy as np


#Load dataset
import matplotlib.pyplot as plt
raw_dataset = [(1, 1, 1), (1, 2, 1), (1, 3, 1), (2, 1, 1), (2, 2, 1), (2, 3, 1), (2, 3.5, 1), (2.5, 2, 1), (3.5, 1, 1), (3.5, 2, 1), (3.5, 3, 2), (3.5, 4, 2), (4.5, 1, 2), (4.5, 2, 2), (4.5, 3, 2), (5, 4, 2), (5, 5, 2), (6, 3, 2), (6, 4, 2), (6, 5, 2)]

df = pd.DataFrame(raw_dataset,columns=['x','y','class'])

df.set_index(['class'],inplace = True)

#Calculate S_W^-1
def sum(data_set, mean):
    sum_matrix = np.zeros(2)
    for value in data_set:
        print(np.outer(value-mean,value-mean))
        sum_matrix = sum_matrix + np.outer(value-mean, value-mean)
    return sum_matrix

class_1_dataset = df.loc['1']
class_1_dataset_mean = class_1_dataset.mean().values
class_1_dataset_values = class_1_dataset.values

class_2_dataset = df.loc['2']
class_2_dataset_mean = class_2_dataset.mean().values
class_2_dataset_values = class_2_dataset.values#đưa về array


class_1_sum = sum(class_1_dataset_values,class_1_dataset_mean)
class_2_sum = sum(class_2_dataset_values,class_2_dataset_mean)

#project data on 1 line
def newCoordinates(class_1_sum,class_2_sum,class_1_mean,class_2_mean,class_1_dataset,class_2_dataset):
    
    S_W = class_1_sum + class_2_sum
    
    S_1_Inv = np.linalg.inv(S_W)
    
    weight =  np.matmul(S_1_Inv,(class_1_dataset_mean-class_2_dataset_mean).transpose())#Phải sử dụng để fit chiều.
    setPoint_C1 = []

    len(class_1_dataset)
    for i in range(0,len(class_1_dataset)):
        setPoint_C1.append(np.matmul(weight,(class_1_dataset_values[i]).transpose()))
    
    setPoint_C2=[]
    print("---------")
    print("Class 2: ",end='\n')
    for i in range(0,len(class_2_dataset)):
        setPoint_C2.append(np.matmul(weight,(class_2_dataset_values[i]).transpose()))
    return setPoint_C1, setPoint_C2

setPoint_C1, setPoint_C2 = newCoordinates(class_1_sum,class_2_sum,class_1_dataset_mean,class_2_dataset_mean,class_1_dataset,class_2_dataset)
print("---------")
print("Class 1: ",end='\n')
print(setPoint_C1,end='\n')
min_C1_value = setPoint_C1[0]
max_C1_value = setPoint_C1[-1]


print("---------")
print("Class 2: ",end='\n')
print(setPoint_C2,end='\n')
min_C2_value = setPoint_C2[0]
max_C2_value = setPoint_C2[-1]


C1_C2_point_separation = max_C1_value + (min_C2_value-max_C1_value)/2 #Line separate


data1 = (setPoint_C1, setPoint_C2)
colors1 = ("red", "green")
groups1 = ("C1", "C2") 


#Plot 2 region separable
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

for data, color, group in zip(data1, colors1, groups1):
    x = data
    y = np.zeros(len(x))
    ax1.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
    ax1.axvline(x=C1_C2_point_separation, color='k', linestyle='--')
plt.legend()
plt.show()

