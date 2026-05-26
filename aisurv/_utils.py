# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 21:45:54 2026

@author: biost
"""

# coxph - Cox proportional hazards model
# deep  - deep learning



import numpy as np
import math


# %% Kaplan-Meier estimator

# class km0:
    
#     ## initial function: attributes 
#     def __init__(self,
#                  y,delta,
#                  type='right'
#                  ):

#         # for: data of onservations
#         self.y=y
#         self.delta=delta
#         self.n = len(y)
        
#         # for: utils attributes
#         self.type = type
        
#         # for: initialize fitted results
#         self.result = {}
        
#     def survival(self,t,se=False):
        
#         # preparation
#         t_jump = np.sort(np.unique(self.y[self.delta==1]))
#         d = np.array([np.sum(self.y==x) for x in t_jump])
#         R = np.array([np.sum(self.y>=x) for x in t_jump])
#         prods = 1-d/R
            
#         # calculate the values of KM at pre-specified time points tm
#         if self.type == 'right':
#             St = np.array([np.prod(prods[t_jump<=x]) for x in t])
#         else:
#             St = np.array([np.prod(prods[t_jump<x]) for x in t])
        
#         # store estimate

#         self.result['survival'] = {'time':t,'estimate':St}
        
#         # standard errors
#         if se is True:
#             RRd = np.array(R*(R-d),dtype='float')
#             RRd[RRd==0] = float('inf')
#             St_se_pre = np.array(
#                 [np.sum(d[t_jump<=x]/RRd[t_jump<=x]) for x in t])
#             self.result['survival']['se'] = St*np.sqrt(St_se_pre)

#     def km_influence(self,t):
        
#         Hy = np.array([np.sum(y<=x) for x in self.y])/self.n
#         Hy[Hy==1] = float('inf')
#         St = km(t=t,y=self.y,delta=self.delta,se=False)['estimate']
#         temp_1 = np.stack([delta*(y<=x)/(1-Hy) for x in t],axis=1)
#         temp_2 = np.stack([[
#             np.sum(delta*(y<=np.min([x,yi]))/((1-Hy)**2))/n
#             for yi in y] for x in t],axis=1)
#         influence = (temp_2-temp_1)*St
        
#         return influence

# def km(time,y,delta,type="right"):
    
#     # preparation
#     t_jump = np.sort(np.unique(y[delta==1]))
#     d = np.array([np.sum(y==x) for x in t_jump])
#     R = np.array([np.sum(y>=x) for x in t_jump])
#     prods = 1-d/R
        
#     # calculate the values of KM at pre-specified time points tm
#     if type == 'right':
#         St = np.array([np.prod(prods[t_jump<=x]) for x in time])
#     else:
#         St = np.array([np.prod(prods[t_jump<x]) for x in time])
        
#     # returned values
#     return St

# def km_influence(t,y,delta):
    
#     n = len(y)
#     Hy = np.array([np.sum(y<=x) for x in y])/n
#     Hy[Hy==1] = float('inf')
#     St = km(t=t,y=y,delta=delta,se=False)['estimate']
#     temp_1 = np.stack([delta*(y<=x)/(1-Hy) for x in t],axis=1)
#     temp_2 = np.stack([[
#         np.sum(delta*(y<=np.min([x,yi]))/((1-Hy)**2))/n
#         for yi in y] for x in t],axis=1)
#     influence = (temp_2-temp_1)*St
    
#     return influence

# %% Copula-Graphic estimator with a given copula


# %%

# The Soft-Thresholding function
#   - the solution to LASSO penalty under the simplest situation: 
#     2^{-1}(z-x)^2+penalty
#   - this is the equation (2.6) in Fan and Li (2001)
def Soft_Threshold(x,tunpara):
    return np.sign(x)*np.maximum(np.abs(x)-tunpara,0)
    
def basis_Bernstein(x,k,d,il,ir):

    # x: evaluated points (can be a vector)
    # k: the kth basis function
    # d: the degree
    # il: the left end point of the interval
    # ir: the right end point of the interval

    x_scale = (x-il)/(ir-il)
    return math.comb(d,k)*(x_scale**k)*((1-x_scale)**(d-k))









