import numpy as np 
import matplotlib.pyplot as py

class svm(): 
  def __init__(self): 
    self.w=[] 
    self.b=[]
    starting_w=1 
    starting_b=10 
    self.w=[starting_w,starting_w,starting_b] #some implementations have rolled bias together to simplify calculations. 
    self.C = 1.0
  def fit (self,data,labels,epochs=100): 
    temp_data= np.array(data) 
    bias = np.array(np.ones(len(labels)))
    bias *=-1
    bias=bias.reshape(len(bias),1)
    final = np.c_[temp_data,bias]
    data=final
    for epoch in range(1,epochs):
      for i in range (len(labels)): 
        #iterate 
        if (1-(labels[i]*(np.dot(self.w,data[i]) ))) >0: 
          w_grad = -1*((np.transpose(self.w)*(1/epoch)) - (labels[i]*self.C)*np.transpose(data[i]) )  
        else: 
          w_grad=-1*((1/epoch)*np.transpose(self.w))
        self.w=self.w+w_grad 
  def predict(self,data): 
    data= np.c_[[data],-1]
    return (np.sign(np.dot(self.w,data[0])))
