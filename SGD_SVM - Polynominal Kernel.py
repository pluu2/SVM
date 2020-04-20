
def kernel (input,c=1,p=2): #polymonial kernel
  return ((np.transpose(input)*input) + c)**p

class svm(): 
  def __init__(self): 
    self.w=[] 
    self.b=[] #not used
    starting_w=1 
    starting_b=10
    self.w=[kernel(starting_w),kernel(starting_w), starting_b] 
    self.C = 1.0
  def fit (self,data,labels,epochs=100): 
    temp_data= np.array(data) #rolled bias into w
    bias = np.array(np.ones(len(labels)))
    bias *=-1
    bias=bias.reshape(len(bias),1)
    final = np.c_[temp_data,bias]
    data=final
    for epoch in range(1,epochs):
      for i in range (len(labels)): 
        kernalized_data = [kernel(data[i,0]),kernel(data[i,1]),-1]
        if (1-(labels[i]*(np.dot(self.w,kernalized_data) ))) >0: 
          w_grad = -1*((np.transpose(self.w)*(1/epoch)) - (labels[i]*self.C)*np.transpose(kernalized_data) )  
        else: 
          w_grad=-1*((1/epoch)*np.transpose(self.w))
        self.w=self.w+w_grad 
  def predict(self,data): 
    kernalized_data = [kernel(data[0]),kernel(data[1]),-1]
    return (np.sign(np.dot(self.w,kernalized_data)))
