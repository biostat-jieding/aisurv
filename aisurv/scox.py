
import numpy as np
from . import km
from . import cox

###############################################
# %% cox model without any axiliary information
class _(km._):
    
    ## initial function: attributes 
    def __init__(self,y,delta,X,D):
    
        # for: data of onservations
        self.y=y
        self.delta=delta
        self.X=X
        self.D=D
        
        # for: utils attributes
        self.eps = 1e-6
        self.maxit = 1000
        self.n,self.p = X.shape
        self.G = len(np.unique(D))
        self.ng = [np.sum(self.D==g) for g in range(self.G)]
        
        # for: initialize fitted results
        self.result = {}
        
        # for: stored intermediate quantities (share across different methods)
        self._intermediate = {}

    ## estimation and inference on (regression) coefficient (log of hazard ratio)
    def coefficient(self,index=None): 
        
        # prepare indices of interested coefficients
        if index is None:
            index = list(range(self.p))
        
        # calculate the maximum partial likelihood estimator
        if 'beta' not in self._intermediate:
            self._intermediate['beta'] = self._MPLE()
        beta = self._intermediate['beta']
        alpha = beta[index]
        
        # variance
        alpha_influence = self._coefficient_influence(beta=beta,index=index)
        alpha_avar = (alpha_influence**2).sum(axis=0)/self.n
        alpha_se = np.sqrt(alpha_avar/self.n)

        # store fitted values
        self.result['coefficient'] = {'estimate':alpha}
        self.result['coefficient']['se'] = alpha_se

    ## influence of estimated coefficients
    def _coefficient_influence(self,beta,index=None,omega=None):
        
        # preparations
        if index is None:
            index = list(range(self.p))
    
        # prepare the omega quantity
        if omega is None:
            infom = - self._partial_likelihood_derivative(beta=beta,d1=False,d2=True)['d2']
            infom_inv = np.linalg.inv(infom)
            hd_alpha = np.zeros(shape=(self.p,len(index)))
            hd_alpha[index,:] = np.eye(len(index))
            omega = infom_inv.dot(hd_alpha)
            
        # calculate the influence function
        influence_score = self._partial_likelihood_derivative_influence(beta=beta)
        influence = influence_score.dot(omega)

        # returned values
        return influence

    ## partial likelihood function
    def _partial_likelihood(self,beta):

        # calculate mathematically
        Xb   = np.dot(self.X,beta)
        eXb  = np.exp(Xb)
        SS0 = np.zeros((self.n,1))
        for g in range(self.G):
            SS0[self.D==g,:] = np.vstack(
                [eXb[self.D==g][self.y[self.D==g]>=yi].sum() 
                 for yi in self.y[self.D==g]])/self.ng[g]
        plik = np.mean((Xb-np.log(SS0))*self.delta)
    
        # output the value of the likelihood function
        return plik
    
    ## derivatives of partial likelihood function (first and second)
    def _partial_likelihood_derivative(self,beta,d1=True,d2=True):
        
        # preparations
        Xb   = np.dot(self.X,beta)
        eXb  = np.exp(Xb)
        XeXb = self.X*eXb.reshape((-1,1))
        SS0 = np.zeros((self.n,1))
        SS1 = np.zeros((self.n,self.p))
        for g in range(self.G):
            SS0[self.D==g,:] = np.vstack(
                [eXb[self.D==g][self.y[self.D==g]>=yi].sum() 
                 for yi in self.y[self.D==g]])/self.ng[g]
            SS1[self.D==g,:] = np.vstack(
                [XeXb[self.D==g,:][self.y[self.D==g] >= yi].sum(axis=0) 
                 for yi in self.y[self.D==g]])/self.ng[g]

        # the first derivative of log partial likelihood (score) 
        out = {}
        if d1 == True:
            out['d1'] = ((self.X-SS1/SS0)*self.delta[:,None]).mean(axis=0)
           
        # the second derivative of log partial likelihood (negative information matrix)
        if d2 == True:
            SS2 = np.zeros((self.n,self.p,self.p))
            for g in range(self.G):
                SS2[self.D==g,:,:] = np.stack(
                    [np.dot(self.X[self.D==g,:][self.y[self.D==g]>=yi,:].T,XeXb[self.D==g,:][self.y[self.D==g]>=yi,:]) 
                     for yi in self.y[self.D==g]],axis=0)/self.ng[g]
            I1 = (SS2*self.delta[:,None,None]/SS0[:,None]).mean(axis=0)
            I2 = np.dot(SS1.T,SS1*self.delta[:,None]/(SS0**2))/self.n
            out['d2'] = I2-I1
        
        # returned values
        return out
    
    ## influence of the first derivative of the partial likelihood
    def _partial_likelihood_derivative_influence(self,beta):
        
        # preparations
        Xb   = np.dot(self.X,beta)
        eXb  = np.exp(Xb)
        XeXb = self.X*eXb.reshape((-1,1))
        SS0 = np.zeros((self.n,1))
        SS1 = np.zeros((self.n,self.p))
        for g in range(self.G):
            SS0[self.D==g,:] = np.vstack(
                [eXb[self.D==g][self.y[self.D==g]>=yi].sum() 
                 for yi in self.y[self.D==g]])/self.ng[g]
            SS1[self.D==g,:] = np.vstack(
                [XeXb[self.D==g,:][self.y[self.D==g] >= yi].sum(axis=0) 
                 for yi in self.y[self.D==g]])/self.ng[g]

        # the first derivative of log partial likelihood (Score) 
        influence_left = ((self.X-SS1/SS0)*self.delta[:,None]) 
        influence_right = np.zeros((self.n,self.p))
        for g in range(self.G):
            influence_right[self.D==g,:] = eXb[self.D==g][:,None] * np.stack([(
                (self.X[self.D==g,:][i,:] - SS1[self.D==g,:]/SS0[self.D==g,:])*(
                    (self.delta[self.D==g]==1) 
                    & (self.y[self.D==g]<=self.y[self.D==g][i]))[:,None]/SS0[self.D==g,:]
                ).sum(axis=0) for i in range(self.ng[g])
            ],axis=0)/self.ng[g]
        influence = influence_left - influence_right
        
        # returned values
        return influence

    ## calculate the maximum partial likelihood estimator
    def _MPLE(self):

        # calculate the estimator
        beta_old = np.zeros(self.p)
        for numit in range(int(self.maxit)):
            
            # - update the estimator
            plik_d = self._partial_likelihood_derivative(
                beta=beta_old,d1=True,d2=True)
            dev  = np.linalg.solve(plik_d['d2'],plik_d['d1'])
            beta = beta_old - dev
            
            # - judge the convergence
            if (np.abs(dev).max()>self.eps):
                beta_old = beta
            else:
                break

        # returned values
        return beta

   ## calculate the Breslow estimator
    def _Breslow(self,time,beta,g):

        # estimator
        eXb  = np.exp(np.dot(self.X,beta))
        SS0  = np.vstack([eXb[self.D==g][self.y[self.D==g]>=yi].sum() 
                          for yi in self.y[self.D==g]])/self.ng[g]
        Lamt = np.array([
            np.sum((self.y[self.D==g]<=x)[self.delta[self.D==g]==1]/SS0[
                self.D==g,:][self.delta[self.D==g]==1,:].reshape(-1))
            for x in time])/self.ng[g]

        # returned values
        return Lamt



###########################################################################
# %% synthesis of survival probabilities from Kaplan-Meier estimator (kmsp)
class kmsp(cox.kmsp,_):

    ## initial function: attributes 
    def __init__(self,
                 y,delta,X,D,ai,
                 transition=['km','cox'][0],
                 degree=3):
        
        # inherit attributes from its father
        _.__init__(self,y=y,delta=delta,X=X,D=D)
        cox.kmsp.__init__(self,y=y,delta=delta,X=X,degree=degree,ai=ai)

        # for: parameters used in processing auxiliary information
        self.transition = transition

        # for: original model
        self.model_ori = _(y=self.y,delta=self.delta,X=self.X,D=self.D)


    ## conversion of auxiliary information (into transitional elements)
    def _ai_conversion_transition(self):

        # transitional estimator based on different approaches
        if self.transition in ['km']: 
            
            # - transitional estimator based on Kaplan-Meier estimator
            trans_list,trans_influence_list = km.kmsp._ai_conversion_transition(self)

        else:

            # - preparations
            if 'beta' not in self.model_ori._intermediate:
                self.model_ori._intermediate['beta'] = self.model_ori._MPLE()
            beta = self.model_ori._intermediate['beta']
            Xb   = np.dot(self.X,beta)
            eXb  = np.exp(Xb)
            XeXb = self.X*eXb.reshape((-1,1))
            SS0 = np.zeros((self.n,1))
            SS1 = np.zeros((self.n,self.p))
            SS2 = np.zeros((self.n,self.p,self.p))
            for g in range(self.G):
                SS0[self.D==g,:] = np.vstack(
                    [eXb[self.D==g][self.y[self.D==g]>=yi].sum() 
                     for yi in self.y[self.D==g]])/self.ng[g]
                SS1[self.D==g,:] = np.vstack(
                    [XeXb[self.D==g,:][self.y[self.D==g] >= yi].sum(axis=0) 
                     for yi in self.y[self.D==g]])/self.ng[g]
                SS2[self.D==g,:,:] = np.stack(
                    [np.dot(self.X[self.D==g,:][self.y[self.D==g]>=yi,:].T,XeXb[self.D==g,:][self.y[self.D==g]>=yi,:]) 
                     for yi in self.y[self.D==g]],axis=0)/self.ng[g]
            infom_1 = (SS2*self.delta[:,None,None]/SS0[:,None]).mean(axis=0)
            infom_2 = np.dot(SS1.T,SS1*self.delta[:,None]/(SS0**2))/self.n
            infom = infom_1-infom_2
            score_influence_1 = ((self.X-SS1/SS0)*self.delta[:,None]) 
            score_influence_2 = np.zeros((self.n,self.p))
            for g in range(self.G):
                score_influence_2[self.D==g,:] = eXb[self.D==g][:,None] * np.stack([(
                    (self.X[self.D==g,:][i,:] - SS1[self.D==g,:]/SS0[self.D==g,:])*(
                        (self.delta[self.D==g]==1) 
                        & (self.y[self.D==g]<=self.y[self.D==g][i]))[:,None]/SS0[self.D==g,:]
                    ).sum(axis=0) for i in range(self.ng[g])
                ],axis=0)/self.ng[g]
            score_influence = score_influence_1 - score_influence_2

            # - list of transitional elements
            trans_list = [[] for _ in range(self.group_num)]
            trans_influence_list = [[] for _ in range(self.group_num)]
            for ig in range(self.group_num):
                
                # -- setups of the current subgroup
                group_ = self.ai[self.pair[ig][0]][self.pair[ig][1]]
                id_ = group_['id']
                id_prop = len(id_)/self.n
                t_ = group_['t']

                # -- specific estiamtes for transitional estimators
                id_TF = np.array([i in id_ for i in range(self.n)])
                Lamt = [np.array([
                    np.sum((self.y[self.D==g]<=x)[self.delta[self.D==g]==1]/SS0[self.D==g,:][self.delta[self.D==g]==1,:].reshape(-1))
                    for x in t_])/self.ng[g] for g in range(self.G)]
                StX_id_TF = (np.stack([(self.ng[g]/self.n)*np.exp(-Lamt[g])[:,None]**eXb 
                                       for g in range(self.G)],axis=0).sum(axis=0) )*id_TF
                trans_ = np.mean(StX_id_TF,axis=1)/id_prop

                # -- influence functions # 要改！
                # Lamt_influence_1 = np.stack([
                #     (self.y<=x)*self.delta/SS0.reshape(-1) for x in t_],axis=1)
                # Lamt_influence_2 = eXb[:,None]*np.stack([  
                #     [np.sum((self.y<=np.min([x,yi]))[self.delta==1]/(SS0[self.delta==1,:].reshape(-1)**2))
                #     for yi in self.y] for x in t_],axis=1)/self.n
                # Lamt_influence = Lamt_influence_1 - Lamt_influence_2
                # Lamt_d1 = np.stack([np.sum(
                #     (SS1*(self.y<=x)[:,None])[self.delta==1,:]/(SS0[self.delta==1,:]**2),
                #     axis=0) for x in t_],axis=1)/self.n
                # psi_d1 = np.stack([np.mean(
                #     (Lamt_d1[:,[i]]*eXb - XeXb.T*Lamt[i])*StX_id_TF[i,],axis=1)
                #     for i in range(len(t_))],axis=1)
                
                
                Lamt_influence = np.zeros((self.n,len(t_)))
                for g in range(self.G):
                    Lamt_influence_1 = np.stack([
                        (self.y[self.D==g]<=x)*self.delta[self.D==g]/SS0[self.D==g,:].reshape(-1) for x in t_],axis=1)
                    Lamt_influence_2 = eXb[self.D==g][:,None]*np.stack([  
                        [np.sum((self.y[self.D==g]<=np.min([x,yi]))[self.delta[self.D==g]==1]/(SS0[self.D==g,:][self.delta[self.D==g]==1,:].reshape(-1)**2))
                        for yi in self.y[self.D==g]] for x in t_],axis=1)/self.ng[g]
                    Lamt_influence[self.D==g,:] = Lamt_influence_1 - Lamt_influence_2
                    
                psi_d1 = []
                for g in range(self.G):
                    Lamt_d1 = np.stack([np.sum(
                        (SS1[self.D==g,:]*(self.y[self.D==g]<=x)[:,None])[self.delta[self.D==g]==1,:]/(SS0[self.D==g,:][self.delta[self.D==g]==1,:]**2),
                        axis=0) for x in t_],axis=1)/self.ng[g]
                    psi_d1_ = np.stack([np.mean(
                        (Lamt_d1[:,[i]]*eXb[self.D==g] - XeXb[self.D==g,:].T*Lamt[g][i])*StX_id_TF[:,self.D==g][i,],axis=1)
                        for i in range(len(t_))],axis=1)
                    psi_d1.append(psi_d1_)
                psi_d1 = np.stack([(self.ng[g]/self.n)*psi_d1[g] for g in range(self.G)],axis=0).sum(axis=0)

                h = np.linalg.solve(infom,psi_d1)
                psi = StX_id_TF - trans_[:,None]*id_TF
                StX_id_TF.mean(axis=1)/id_prop
                psi_du1 = - np.mean(StX_id_TF*eXb,axis=1)
                trans_influence_ = (psi.T + Lamt_influence*psi_du1 + 
                                    score_influence.dot(h))/id_prop

                # -- store estimated quantities
                trans_list[ig] = trans_
                trans_influence_list[ig] = trans_influence_
              
        # returned values  
        return  trans_list,trans_influence_list


