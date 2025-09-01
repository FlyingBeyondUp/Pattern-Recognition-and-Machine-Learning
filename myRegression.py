# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 18:41:59 2025

@author: LZ166
"""

import torch
from abc import ABC, abstractmethod
from typing import Union,Tuple
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge, ARDRegression as SklearnARD
import numpy as np

def generate_data(n,noise):
    X=torch.rand(n)
    y=torch.sin(4*torch.pi*X)+noise*torch.randn(n)
    return (X,y)

def polynomial_features(x, degree):
    # x: (n_samples,)
    # returns: (n_samples, degree+1)
    powers = torch.arange(degree + 1, dtype=x.dtype, device=x.device)
    return x.unsqueeze(1) ** powers    # broadcasting

class KernelFeatures:
    def __init__(self,gamma=None):
        self.gamma=gamma
        self.x_train=None
    
    def fit(self,x):
        x_diff = x.unsqueeze(1) - x.unsqueeze(0)  # shape (n_samples, n_samples)
        sq_diff = x_diff ** 2
        
        # Use the median of the squared differences (vectorized)
        # Optionally, exclude zeros (diagonal) if you want
        mask = ~torch.eye(x.shape[0], dtype=bool, device=x.device)
        median_sq_diff = torch.median(sq_diff[mask])
        self.gamma = 1.0 / (median_sq_diff + 1e-8)
        self.x_train=x.clone()
        return  torch.exp(-self.gamma * sq_diff)
    
    def transform(self,x):
        if self.gamma==None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        x_dff=x.unsqueeze(1)-self.x_train.unsqueeze(0)
        return torch.exp(-self.gamma*x_dff**2)
        
        
# def kernel_features(x0,x1, gamma=None):
#     # x0,x1: shape (n_samples,)
#     # K_{nm}=k(x1_n,x0_m)
#     x_diff = x1.unsqueeze(1) - x0.unsqueeze(0)  # shape (n_samples, n_samples)
#     sq_diff = x_diff ** 2
#     if gamma is None:
#         # Use the median of the squared differences (vectorized)
#         # Optionally, exclude zeros (diagonal) if you want
#         mask = ~torch.eye(x0.shape[0], dtype=bool, device=x.device)
#         median_sq_diff = torch.median(sq_diff[mask])
#         gamma = 1.0 / (median_sq_diff + 1e-8)
#     K = torch.exp(-gamma * sq_diff)
#     return K,gamma

class MyRegression(ABC):
    @abstractmethod
    def fit(self,X:torch.Tensor,y:torch.Tensor)->None:
        pass
    
    @abstractmethod
    def predict(self,X:torch.Tensor)->Union[torch.Tensor,Tuple[torch.Tensor,torch.Tensor]]:
        pass
    
    
class LinearRegression(MyRegression):
    def __init__(self,lam:float=0.0):
        self.lam=lam
        self.w=None
    
    def get_MPinverse(self,X):
        U,Sigma,Vt=torch.linalg.svd(X,full_matrices=False)
        Sigma_inv=torch.where(Sigma>1e-6,1/Sigma,torch.zeros_like(Sigma))
        return Vt.T@torch.diag(Sigma_inv)@U.T
        
    def fit(self,X,y):
        if self.lam!=0:
            self.w=torch.linalg.solve(X.T@X+self.lam*torch.eye(X.shape[1]),X.T@y)
        else:
            self.w=self.get_MPinverse(X)@y
        self.beta=X.shape[0]/torch.linalg.norm(y-X@self.w,2)**2
    
    def predict(self,X):
        if self.w is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return X@self.w
    

class BayesianLinearRegression(MyRegression):
    def __init__(self,alpha_init=1e-3,beta_init=1e-1,tol=1e-3,max_iter=1000,n_interval=10):
        self.alpha=alpha_init
        self.beta=beta_init
        self.tol=tol
        self.max_iter=max_iter
        self.n_interval=n_interval
        self.m=None
        self.S=None
    
    def fit(self,X,y,record_log_evidence=[],requires_record=False):
        XtX=X.T@X
        Xty=X.T@y
        eigen_XtX=torch.linalg.eigvalsh(XtX)
        I=torch.eye(X.shape[1])
        N=X.shape[0]
        pre_log_evidence=-float('inf')
        
        for i in range(self.max_iter):
            S_inv=self.alpha*I+self.beta*XtX
            try:
                L=torch.linalg.cholesky(S_inv)
                z=torch.linalg.solve_triangular(L,(self.beta*Xty).unsqueeze(-1),upper=False)
                self.m=torch.linalg.solve_triangular(L.T,z,upper=True).squeeze(-1)
            except torch.linalg.LinAlgError:
                self.m=self.beta*torch.linalg.solve(S_inv,Xty)
            
            lams=self.beta*eigen_XtX
            gamma=torch.sum(lams/(self.alpha+lams))
            self.alpha=gamma/self.m.dot(self.m)
            self.beta=(N-gamma)/torch.linalg.norm(y-X@self.m)**2
            
            if i%self.n_interval==0:
                log_evidence=0.5*X.shape[1]*torch.log(self.alpha)+0.5*X.shape[0]*torch.log(self.beta)-self.E(X,y)-0.5*torch.sum(torch.log(self.alpha+lams))-0.5*N*torch.log(2*torch.tensor(torch.pi))
                if torch.abs((log_evidence-pre_log_evidence)/pre_log_evidence)<self.tol:
                    break
                pre_log_evidence=log_evidence
                if requires_record:
                    record_log_evidence.append(log_evidence.item())
        self.S=torch.linalg.inv(self.alpha*I+self.beta*XtX)
        
    def E(self,X,y):
        return 0.5*self.beta*torch.linalg.norm(y-X@self.m,2)**2+0.5*self.alpha*torch.linalg.norm(self.m,2)**2
        
    def predict(self,X):
        if self.m==None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        noise_variances=torch.einsum('ij,jk,ik->i',X,self.S,X)
        return X@self.m,1/self.beta+noise_variances

# Automatic Relevance Determination
# The current implementation is not stable
# the performance is sensitive to alpha_init
# even for the same parameters and training dataset, the trained model can behave in distinct ways
# while the Bayesian Linear Regression model is more robust
class ARDRegression(MyRegression):
    def __init__(self,alpha_init=1e-5,beta_init=1e-1,tol=1e-3,max_iter=1000,n_interval=10,alpha_bounds=[1e-9,1e9],alpha_damp=0.5):
        self.A=None
        self.alpha_init=alpha_init
        self.beta=beta_init
        self.tol=tol
        self.max_iter=max_iter
        self.n_interval=n_interval
        self.alpha_bounds=alpha_bounds
        self.alpha_damp=alpha_damp
        self.m=None
        self.S=None
        self.alphas=None
    
    def fit(self,X,y,record_log_evidence=[],requires_record=False):
        Xty=X.T@y
        XtX=X.T@X
        eigen_XtX=torch.linalg.eigvalsh(XtX)
        pre_log_evidence=-float('inf')
        N,M=X.shape
        self.alphas=self.alpha_init*torch.ones(X.shape[1])
        
        for i in range(self.max_iter):
            try:
                # Use Cholesky for better stability
                L = torch.linalg.cholesky(torch.diag(self.alphas)+self.beta*XtX)
                z = torch.linalg.solve_triangular(L, (self.beta * Xty).unsqueeze(-1), upper=False)
                self.m = torch.linalg.solve_triangular(L.T, z, upper=True).squeeze(-1)
                
                # Compute S efficiently using Cholesky factor
                S_chol = torch.linalg.solve_triangular(L, torch.eye(M), upper=False)
                self.S = S_chol.T @ S_chol
            except torch.linalg.LinAlgError:
                self.S=torch.linalg.inv(torch.diag(self.alphas)+self.beta*XtX)
                self.m=self.beta*self.S@Xty
            
            lams=self.beta*eigen_XtX
            gamma=torch.sum(lams/(self.alphas+lams))
            new_alphas=1/(self.m*self.m+self.S.diagonal()+1e-9)
            new_alphas=torch.clamp(new_alphas,self.alpha_bounds[0],self.alpha_bounds[1])
            self.alphas=new_alphas*self.alpha_damp+(1-self.alpha_damp)*self.alphas
            self.beta=N/(torch.linalg.norm(y-X@self.m)**2+gamma/self.beta)
            
            if i%self.n_interval==0:
                log_evidence=0.5*torch.sum(torch.log(self.alphas))+0.5*X.shape[0]*torch.log(self.beta)-self.E(X,y)-0.5*torch.sum(torch.log(self.alphas+lams))-0.5*N*torch.log(2*torch.tensor(torch.pi))
                if torch.abs((log_evidence-pre_log_evidence)/pre_log_evidence)<self.tol:
                    break
                pre_log_evidence=log_evidence
                if requires_record:
                    record_log_evidence.append(log_evidence.item())
                    
    def E(self,X,y):
        return 0.5*self.beta*torch.linalg.norm(y-X@self.m,2)**2+0.5*torch.sum(self.m*self.alphas*self.m)
        
    def predict(self,X):
        if self.m==None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        noise_variances=torch.einsum('ij,jk,ik->i',X,self.S,X)
        return X@self.m,1/self.beta+noise_variances
    
        


if __name__=='__main__':
    n_total=200
    n_train=int(n_total*0.7)
    n_degree=20
    sigma=0.2
    
    x,y=generate_data(n=n_total, noise=sigma)
    mean_x,std_x=x.mean(),x.std()
    x=(x-mean_x)/(std_x+1e-8)
    
    x_train,x_val=x[:n_train],x[n_train:]
    y_train,y_val=y[:n_train],y[n_train:]
    kernel_features=KernelFeatures()
    X_train=kernel_features.fit(x_train)
    X_val=kernel_features.transform(x_val)
    
    
    
    # preprocess polynomial features
    # X=polynomial_features(x, n_degree)
    # X_train,y_train,X_val,y_val=X[:n_train],y[:n_train],X[n_train:],y[n_train:]
    # x_val=x[n_train:]
    # mean = X_train[:,1:].mean(dim=0, keepdim=True)
    # std  = X_train[:,1:].std(dim=0, keepdim=True, unbiased=False)
    # X_train[:,1:] = (X_train[:,1:] - mean) / (std + 1e-8)
    # X_val[:,1:]   = (X_val[:,1:] - mean) / (std + 1e-8)
    
    
    
    # test the implemented regression algorithms
    ols=LinearRegression()
    ols.fit(X_train,y_train)
    y_ols=ols.predict(X_val)
    
    ridge=LinearRegression(lam=1e-4)
    ridge.fit(X_train,y_train)
    y_ridge=ridge.predict(X_val)
    
    BLR=BayesianLinearRegression(alpha_init=1e-4,beta_init=1e-1)
    list_log_p=[]
    BLR.fit(X_train,y_train,list_log_p,requires_record=True)
    plt.plot(range(len(list_log_p)),list_log_p)
    y_BLR,_=BLR.predict(X_val)
    
    ARD=ARDRegression(alpha_init=1e-4,beta_init=1e-1)
    list_log_p=[]
    ARD.fit(X_train,y_train,list_log_p,requires_record=True)
    plt.plot(range(len(list_log_p)),list_log_p)
    y_ARD,_=ARD.predict(X_val)
    
    criterion=torch.nn.MSELoss()
    print('ols val loss:',criterion(y_ols,y_val))
    print('ridge val loss:',criterion(y_ridge,y_val))
    print('Bayesian Linear Regression val loss:',criterion(y_BLR,y_val))
    print('ARDRegression val loss:',criterion(y_ARD,y_val))
    
    print('----------------------------------------------')
    print('model capacities:')
    print(f'ols: {torch.norm(ols.w,2)}')
    print(f'ridge: {torch.norm(ridge.w,2)}')
    print(f'Bayesian Ridge: {torch.norm(BLR.m,2)}')
    print(f'ARD: {torch.norm(ARD.m,2)}')
    print('----------------------------------------------')
    print('estimated beta:')
    print(f'ground truth: {1/sigma**2}')
    print(f'ols: {ols.beta}')
    print(f'ridge: {ridge.beta}')
    print(f'Bayesian Ridge: {BLR.beta}')
    print(f'ARD: {ARD.beta}')
    print('----------------------------------------------')
    
    
    
    x_all=torch.linspace(0,1,1000)
    x_all=(x_all-mean_x)/(std_x+1e-8)
    
    X_all=kernel_features.transform(x_all)
    
    # X_all=polynomial_features(x_all, n_degree)
    # X_all[:,1:]=(X_all[:,1:]-mean)/(std+1e-8)
    
    plt.figure()
    plt.title('ols')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x_val,y_val,label='data')
    plt.plot(x_all,ols.predict(X_all))
    plt.show()
    
    plt.figure()
    plt.title('ridge')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x_val,y_val,label='data')
    plt.plot(x_all,ridge.predict(X_all))
    plt.show()
    
    y_BLR_all,noise_BLR=BLR.predict(X_all)
    plt.figure()
    plt.title('Bayesian Linear Regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x_val,y_val,label='data')
    plt.plot(x_all,y_BLR_all,label='predicted mean')
    plt.plot(x_all,y_BLR_all+torch.sqrt(noise_BLR),linestyle='-.',color='grey',label='predicted upper bound')
    plt.plot(x_all,y_BLR_all-torch.sqrt(noise_BLR),linestyle='-.',color='grey',label='predicted lower bound')
    plt.show()
    
    y_ARD_all,noise_ARD=ARD.predict(X_all)
    plt.figure()
    plt.title('ARD')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x_val,y_val,label='data')
    plt.plot(x_all,y_ARD_all,label='predicted mean')
    plt.plot(x_all,y_ARD_all+torch.sqrt(noise_ARD),linestyle='-.',color='grey',label='predicted upper bound')
    plt.plot(x_all,y_ARD_all-torch.sqrt(noise_ARD),linestyle='-.',color='grey',label='predicted lower bound')
    plt.show()
    
    
    
    # comparison with standard implementation
    X_np = X_train.numpy()
    y_np = y_train.numpy()

    # Sklearn Bayesian Ridge
    sklearn_br = BayesianRidge(max_iter=1000, tol=1e-3,alpha_init=1e-1,lambda_init=1e-4)
    sklearn_br.fit(X_np, y_np)
    y_sklearn_br = sklearn_br.predict(X_val.numpy())

    # Sklearn ARD
    sklearn_ard = SklearnARD(max_iter=1000, tol=1e-3,alpha_1=1e-7)
    sklearn_ard.fit(X_np, y_np)
    y_sklearn_ard = sklearn_ard.predict(X_val.numpy())
    
    print("Sklearn BR val loss:", criterion(torch.tensor(y_sklearn_br), y_val))
    print("Sklearn ARD val loss:", criterion(torch.tensor(y_sklearn_ard), y_val))
    
    plt.figure()
    plt.title('sklearn')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x_val,y_val,label='data')
    plt.plot(x_all,sklearn_ard.predict(X_all.numpy()),label='sklearn ARD')
    plt.plot(x_all,sklearn_br.predict(X_all.numpy()),label='sklearn BR')
    plt.legend()
    plt.show()
    
    
    
    # check sparsity
    w_ARD=ARD.m.clone()
    w_ARD=w_ARD*~torch.tensor(torch.abs(w_ARD)/torch.max(torch.abs(w_ARD))<2e-1)
    
    plt.figure()
    plt.title('sparsity')
    plt.plot(x_all,X_all@w_ARD)
    plt.scatter(x_train,y_train)
    plt.scatter(x_train[w_ARD!=0],y_train[w_ARD!=0],color='red')
    plt.show()
    
    
    
            