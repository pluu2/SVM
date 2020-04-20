# Support Vector Machine Experiments 
The following repo contains practice with implementing SVMs from scratch. I will attempt to improve my understanding of SVMs by beginning with a basic exhaustive search SVM, then improving on the code as I learn better methods of optimization. 

I will also implement Kernels to allow for non-linear separability of data. 

Exhaustive SVM is simply iterating through all combinations of w, and b, and finding the value that generates the smallest normalization value. 




SVM_SGD_BiasFeature.py , uses SGD to solve 'w', by modifying decision boundary with respect to w, and b in order to minimize the hinge loss:

L(0,1-(yi(wxi +b)). 

![SVM](/images/svm_graph.png)


Figure: Red is points classified as -1, Green are points classified as +1, blue points were points used to train SVM located in their respective classification areas. 


SGD_SVM - Polynominal Kernel.py contains implementation using python for a SGD optimized SVM with polynomial kernel. 

![SVM_kernel](/images/SVM_Polynomial_Kernel.png)


## To Do: 
[x] exhaustive SVM 

[X] SGD SVM 

[ ] Kernel - Polynonmial 

[ ] Kernel - Central Basis Function (Gaussian) 

