# -*- coding: utf-8 -*-
"""Exhaustive SVM.ipynb

Automatically generated by Colaboratory.

"""

import numpy as np 
import matplotlib.pyplot as py

class svm(): 
  def __init__(self): 
    self.w =[] 
    self.b=[]
    self.optimum={}
  
  def predict(self,data):
    return np.sign(np.dot(self.w,data)+self.b)
  
  def fit(self,data,labels): 
    data_temp= np.array(data) 
    data_temp=np.ndarray.flatten(data_temp) 
    max_data = max(data_temp) 
    min_data= min(data_temp) 
    recent_optimum = max_data
    
    transforms = [[1,1],[1,-1],[-1,-1],[-1,1]]
    found_one=False
    for b in np.arange (-1*max_data,1*max_data,0.1): #move through b
      self.b=b
      for transformations in transforms:  #move through transformations
        self.w=np.array([recent_optimum,recent_optimum])
        for vector_vals in np.arange(-1,1,0.01): #move through vector values
          w_t=self.w * transformations
          correct=0
          for i in range(len(labels)):
            if (labels[i]*np.dot(w_t,data[i]) + self.b)>=1 : 
              correct+=1
          if correct == len(data): 
            self.optimum[np.linalg.norm(w_t)]=[w_t,b]
            correct=0
            found_one=True
          else: 
            correct=0
          self.w=self.w-0.01 #arbitrary
    
    if found_one: 
      #once combo is found: 
      norms = sorted([n for n in self.optimum]) #sorted lowest to highest 
      #||w|| : [w,b]
      opt_choice = self.optimum[norms[0]]
      self.w = opt_choice[0] #gather w_t
      self.b = opt_choice[1] #gather b
      print (f'done with norm of : ', norms[0])
    else: 
      print ('not linear separable ') 

  def predict(self,data): 
    return (np.sign(np.dot(self.w,data)+self.b))

