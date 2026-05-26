
import numpy as np
from . import km

###############################################
# %% cox model without any axiliary information
class _(km._):
    
    ## initial function: attributes 
    def __init__(self,y,delta,X):
    
        # inherit attributes from its father
        km._.__init__(self,y=y,delta=delta)
        
        # for: data of onservations
        self.X=X
        
        # for: utils attributes
        self.eps = 1e-6
        self.maxit = 1000
        self.p = X.shape[1]

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

    ## estimation and inference on hazard ratio (exponent of coefficient)
    def hazard_ratio(self,index=None):

        # prepare indices of interested coefficients
        if index is None:
            index = list(range(self.p))
 
        # calculate the maximum partial likelihood estimator
        if 'beta' not in self._intermediate:
            self._intermediate['beta'] = self._MPLE()
        beta = self._intermediate['beta']
        hr = np.exp(beta[index])
        
        # variance
        hr_influence = self._hazard_ratio_influence(beta=beta,index=index)
        hr_avar = (hr_influence**2).sum(axis=0)/self.n
        hr_se = np.sqrt(hr_avar/self.n)

        # store fitted values
        self.result['hazard_ratio'] = {'estimate':hr}
        self.result['hazard_ratio']['se'] = hr_se

    ## estimation and inference on risk score (exponent of linear predictor)
    def risk_score(self,x):

        # calculate the maximum partial likelihood estimator
        if 'beta' not in self._intermediate:
            self._intermediate['beta'] = self._MPLE()
        beta = self._intermediate['beta']
        rs = np.exp(x.dot(beta))

        # variance
        rs_influence = self._risk_score_influence(beta=beta,x=x)
        rs_avar = (rs_influence**2).sum(axis=0)/self.n
        rs_se = np.sqrt(rs_avar/self.n)

        # store fitted values
        self.result['risk_score'] = {'estimate':rs}
        self.result['risk_score']['se'] = rs_se

    ## estimation and inference on baseline cumulative hazard function at given time
    def baseline_cumulative_hazard(self,time):
        
        # preparations
        if 'beta' not in self._intermediate:
            self._intermediate['beta'] = self._MPLE()
        beta = self._intermediate['beta']
        bch = self._Breslow(time=time,beta=beta)

        # influence function and the associated variance 
        bch_influence = self._baseline_cumulative_hazard_influence(
            time=time,beta=beta)
        bch_avar = (bch_influence**2).sum(axis=0)/self.n
        bch_se = np.sqrt(bch_avar/self.n)

        # store fitted values
        self.result['baseline_cumulative_hazard'] = {'estimate':bch}
        self.result['baseline_cumulative_hazard']['se'] = bch_se

    ## estimation and inference on conditional survival probabilities
    def survival_probability(self,time,x,cross=True):
        
        # the estimaor
        if 'beta' not in self._intermediate:
            self._intermediate['beta'] = self._MPLE()
        beta = self._intermediate['beta']
        Lamt = self._Breslow(time=time,beta=beta)
        exb = np.exp(x.dot(beta))
        if cross is False:
            sp = np.exp(-Lamt*exb)
        else:
            sp = np.exp(-Lamt[:,None]*exb)
        
        # asymptotic variance
        sp_influence = self._survival_probability_influence(
            beta=beta,time=time,x=x,cross=cross)
        sp_avar = (sp_influence**2).sum(axis=0)/self.n
        sp_se = np.sqrt(sp_avar/self.n)

        # store fitted values
        self.result['survival_probability'] = {'estimate':sp}
        self.result['survival_probability']['se'] = sp_se
        
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

    ## influence of hazard ratio 
    def _hazard_ratio_influence(self,beta,index=None):

        # preparations
        if index is None:
            index = list(range(self.p))

        # # prepare the omega quantity
        # if omega is None:

        # influence function
        hr = np.exp(beta[index])
        alpha_influence = self._coefficient_influence(beta=beta,index=index)
        influence = alpha_influence*hr

        # returned values
        return influence

    ## influence of risk score
    def _risk_score_influence(self,x,beta):

        # preparations
        rs = np.exp(x.dot(beta))

        # influence function
        beta_influence = self._coefficient_influence(beta=beta)
        hd_rs = x.T.dot(np.diag(rs))
        influence = beta_influence.dot(hd_rs)

        # returned values
        return influence
    
    ## influence of baseline cumulative hazard function
    def _baseline_cumulative_hazard_influence(self,time,beta):

        # preparations
        eXb  = np.exp(np.dot(self.X,beta))
        XeXb = self.X*eXb.reshape((-1,1))
        SS0  = np.vstack([eXb[self.y>=yi].sum() for yi in self.y])/self.n
        SS1  = np.vstack([XeXb[self.y >= yi].sum(axis=0) 
                          for yi in self.y])/self.n

        # influence function (with beta fixed)
        Lamt_influence_11 = np.stack([
            (self.y<=x)*self.delta/SS0.reshape(-1) for x in time],axis=1)
        Lamt_influence_12 = eXb[:,None]*np.stack([  
            [np.sum((self.y<=np.min([x,yi]))[self.delta==1]/(SS0[self.delta==1,:].reshape(-1)**2))
                for yi in self.y] for x in time],axis=1)/self.n
        Lamt_influence_1 = Lamt_influence_11 - Lamt_influence_12

        # influence function (the remaining part with respect to coefficient)        
        beta_influence = self._coefficient_influence(beta=beta)
        hd_bch = - np.stack([np.sum(
            (SS1*(self.y<=x)[:,None])[self.delta==1,:]/(SS0[self.delta==1,:]**2),
            axis=0) for x in time],axis=1)/self.n
        Lamt_influence_2 = beta_influence.dot(hd_bch)
        
        # obtain hte final influence function
        influence = Lamt_influence_1 + Lamt_influence_2  

        # returned values
        return influence

    ## influence of survival probability
    def _survival_probability_influence(self,time,x,beta,cross=True):

        # preparations
        Lamt = self._Breslow(time=time,beta=beta)
        exb = np.exp(x.dot(beta))
        if cross is False:
            sp = np.exp(-Lamt*exb)
        else:
            sp = np.exp(-Lamt[:,None]*exb)

        # influence function
        eXb  = np.exp(np.dot(self.X,beta))
        XeXb = self.X*eXb.reshape((-1,1))
        SS0  = np.vstack([eXb[self.y>=yi].sum() for yi in self.y])/self.n
        SS1  = np.vstack([XeXb[self.y >= yi].sum(axis=0) 
                        for yi in self.y])/self.n
        Lamt_influence_11 = np.stack([
            (self.y<=x)*self.delta/SS0.reshape(-1) for x in time],axis=1)
        Lamt_influence_12 = eXb[:,None]*np.stack([  
            [np.sum((self.y<=np.min([x,yi]))[self.delta==1]/(SS0[self.delta==1,:].reshape(-1)**2))
                for yi in self.y] for x in time],axis=1)/self.n
        Lamt_influence_1 = Lamt_influence_11 - Lamt_influence_12
        hd_bch = - np.stack([np.sum(
            (SS1*(self.y<=a)[:,None])[self.delta==1,:]/(SS0[self.delta==1,:]**2),
            axis=0) for a in time],axis=1)/self.n
        beta_influence = self._coefficient_influence(beta=beta)
        if cross is False:
            hd_sp = - (hd_bch+x.T*Lamt)*sp*exb  
            influence = -Lamt_influence_1*sp*exb + beta_influence.dot(hd_sp)
        else:
            influence = np.zeros(shape=(self.n,len(time),x.shape[0]))
            for itime in range(len(time)):
                hd_sp = - (hd_bch[:,[itime]]+x.T*Lamt[itime])*sp[itime,:]*exb  
                influence[:,itime,:] = - Lamt_influence_1[:,[itime]]*sp[itime,:]*exb + beta_influence.dot(hd_sp)
                
        # returned values
        return influence

    ## partial likelihood function
    def _partial_likelihood(self,beta):

        # calculate mathematically
        Xb   = np.dot(self.X,beta)
        SS0  = np.array([np.exp(Xb)[self.y>=yi].sum() 
                         for yi in self.y])/self.n
        plik = np.mean((Xb-np.log(SS0))*self.delta)
    
        # output the value of the likelihood function
        return plik
    
    ## derivatives of partial likelihood function (first and second)
    def _partial_likelihood_derivative(self,beta,d1=True,d2=True):
        
        # preparations
        Xb   = np.dot(self.X,beta)
        eXb  = np.exp(Xb)
        XeXb = self.X*eXb.reshape((-1,1))
        SS0  = np.vstack([eXb[self.y>=yi].sum() for yi in self.y])/self.n
        SS1  = np.vstack([XeXb[self.y >= yi].sum(axis=0) 
                          for yi in self.y])/self.n

        # the first derivative of log partial likelihood (score) 
        out = {}
        if d1 == True:
            out['d1'] = ((self.X-SS1/SS0)*self.delta[:,None]).mean(axis=0)
           
        # the second derivative of log partial likelihood (negative information matrix)
        if d2 == True:
            SS2 = np.stack([np.dot(self.X[self.y>=yi,:].T,XeXb[self.y>=yi,:]) 
                            for yi in self.y],
                           axis=0)/self.n
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
        SS0  = np.vstack([eXb[self.y>=yi].sum() for yi in self.y])/self.n
        SS1  = np.vstack([XeXb[self.y >= yi].sum(axis=0) for yi in self.y])/self.n

        # the first derivative of log partial likelihood (Score) 
        influence_left = ((self.X-SS1/SS0)*self.delta[:,None]) 
        influence_right = eXb[:,None] * np.stack([(
            (self.X[i,:] - SS1/SS0)*((self.delta==1) & (self.y<=self.y[i]))[:,None]/SS0
            ).sum(axis=0) for i in range(self.n)
        ],axis=0)/self.n
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
    def _Breslow(self,time,beta):

        # estimator
        eXb  = np.exp(np.dot(self.X,beta))
        SS0  = np.vstack([eXb[self.y>=yi].sum() for yi in self.y])/self.n
        Lamt = np.array([
            np.sum((self.y<=x)[self.delta==1]/SS0[self.delta==1,:].reshape(-1))
            for x in time])/self.n

        # returned values
        return Lamt

###########################################################################
# %% synthesis of survival probabilities from Kaplan-Meier estimator (kmsp)
class kmsp(_,km.kmsp):

    ## initial function: attributes 
    def __init__(self,
                 y,delta,X,ai,
                 transition=['km','cox'][0],
                 degree=3):
        
        # inherit attributes from its father
        _.__init__(self,y=y,delta=delta,X=X)
        km.kmsp.__init__(self,y=y,delta=delta,degree=degree,ai=ai)

        # for: parameters used in processing auxiliary information
        self.transition = transition

        # for: original model
        self.model_ori = _(y=self.y,delta=self.delta,X=self.X)

    ## estimation and inference on coefficient
    def coefficient(self,index=None): 
        
        # preparations
        if index is None:
            index = list(range(self.p))
        
        # original (classical) estimator
        self.model_ori.coefficient(index=index)
        alpha_ori = self.model_ori.result['coefficient']['estimate']
        alpha_ori_influence = self.model_ori._coefficient_influence(
            beta=self.model_ori._intermediate['beta'],index=index)
        
        # estimator with auxiliary information
        alpha,alpha_se = self._ais(
            tp=alpha_ori,tp_influence=alpha_ori_influence)
        self.result['coefficient'] = {'estimate':alpha}
        self.result['coefficient']['se'] = alpha_se  

    ## estimation and inference on hazard ratio
    def hazard_ratio(self,index=None):

        # preparations
        if index is None:
            index = list(range(self.p))

        # original (classical) estimator and its influence function
        self.model_ori.hazard_ratio(index=index)
        hr_ori = self.model_ori.result['hazard_ratio']['estimate']
        hr_ori_influence = self.model_ori._hazard_ratio_influence(
            beta=self.model_ori._intermediate['beta'],index=index)
        
        # estimator with auxiliary information
        hr,hr_se = self._ais(
            tp=hr_ori,tp_influence=hr_ori_influence)
        self.result['hazard_ratio'] = {'estimate':hr}
        self.result['hazard_ratio']['se'] = hr_se  

    ## estimation and inference on risk score
    def risk_score(self,x):
        
        # original (classical) estimator and its influence function
        self.model_ori.risk_score(x=x)
        rs_ori = self.model_ori.result['risk_score']['estimate']
        rs_ori_influence = self.model_ori._risk_score_influence(
            beta=self.model_ori._intermediate['beta'],x=x)
             
        # estimator with auxiliary information
        rs,rs_se = self._ais(
            tp=rs_ori,tp_influence=rs_ori_influence)
        self.result['risk_score'] = {'estimate':rs}
        self.result['risk_score']['se'] = rs_se 

    ## estimation and inference on baseline cumulative hazard function
    def baseline_cumulative_hazard(self,time):

        # original (classical) estimator and its influence function
        self.model_ori.baseline_cumulative_hazard(time=time)
        bch_ori = self.model_ori.result['baseline_cumulative_hazard']['estimate']
        bch_ori_influence = self.model_ori._baseline_cumulative_hazard_influence(
            beta=self.model_ori._intermediate['beta'],time=time)
        
        # estimator with auxiliary information
        bch,bch_se = self._ais(
            tp=bch_ori,tp_influence=bch_ori_influence)
        self.result['baseline_cumulative_hazard'] = {'estimate':bch}
        self.result['baseline_cumulative_hazard']['se'] = bch_se 

    ## estimation and inference on survival probability
    def survival_probability(self,time,x,cross=True):

        # original (classical) estimator and its influence function
        self.model_ori.survival_probability(time=time,x=x,cross=cross)
        sp_ori = self.model_ori.result['survival_probability']['estimate']
        sp_ori_influence = self.model_ori._survival_probability_influence(
            beta=self.model_ori._intermediate['beta'],time=time,x=x,cross=cross)
        
        # estimator with auxiliary information
        if cross is False:
            sp,sp_se = self._ais(
                tp=sp_ori,tp_influence=sp_ori_influence)
        else:
            num_time = len(time)
            num_x = x.shape[0]
            sp = np.zeros(shape=(num_time,num_x))
            sp_se = np.zeros(shape=(num_time,num_x))
            for itime in range(num_time):
                sp[itime,:],sp_se[itime,:] = self._ais(
                    tp=sp_ori[itime,:],tp_influence=sp_ori_influence[:,itime,:])            
        self.result['survival_probability'] = {'estimate':sp}
        self.result['survival_probability']['se'] = sp_se 

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
            SS0  = np.vstack([eXb[self.y>=yi].sum() for yi in self.y])/self.n
            SS1  = np.vstack([XeXb[self.y >= yi].sum(axis=0) 
                                for yi in self.y])/self.n
            SS2 = np.stack([np.dot(self.X[self.y>=yi,:].T,XeXb[self.y>=yi,:]) 
                            for yi in self.y],
                            axis=0)/self.n
            infom_1 = (SS2*self.delta[:,None,None]/SS0[:,None]).mean(axis=0)
            infom_2 = np.dot(SS1.T,SS1*self.delta[:,None]/(SS0**2))/self.n
            infom = infom_1-infom_2
            score_influence_1 = ((self.X-SS1/SS0)*self.delta[:,None]) 
            score_influence_2 = eXb[:,None] * np.stack([ 
                ((self.X[i,:] - SS1/SS0)*(
                    (self.delta==1) & (self.y<=self.y[i]))[:,None]/SS0).sum(axis=0) 
                for i in range(self.n)
            ],axis=0)/self.n
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
                Lamt = np.array([
                    np.sum((self.y<=x)[self.delta==1]/SS0[self.delta==1,:].reshape(-1))
                    for x in t_])/self.n
                StX_id_TF = (np.exp(-Lamt)[:,None]**eXb)*id_TF
                trans_ = np.mean(StX_id_TF,axis=1)/id_prop

                # -- influence functions
                Lamt_influence_1 = np.stack([
                    (self.y<=x)*self.delta/SS0.reshape(-1) for x in t_],axis=1)
                Lamt_influence_2 = eXb[:,None]*np.stack([  
                    [np.sum((self.y<=np.min([x,yi]))[self.delta==1]/(SS0[self.delta==1,:].reshape(-1)**2))
                    for yi in self.y] for x in t_],axis=1)/self.n
                Lamt_influence = Lamt_influence_1 - Lamt_influence_2
                Lamt_d1 = np.stack([np.sum(
                    (SS1*(self.y<=x)[:,None])[self.delta==1,:]/(SS0[self.delta==1,:]**2),
                    axis=0) for x in t_],axis=1)/self.n
                psi_d1 = np.stack([np.mean(
                    (Lamt_d1[:,[i]]*eXb - XeXb.T*Lamt[i])*StX_id_TF[i,],axis=1)
                    for i in range(len(t_))],axis=1)
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


# # %% mpl coefficients from misspecified Cox proportional hazards model (pmcox)
# class coxcoef(kmsp): # 所有都继承kmsp（除了转化ai的函数_ai_conversion）

#     def __init__(self,y,delta,X,ai):

#         # inherit attributes from its father
#         _.__init__(self,y=y,delta=delta,X=X)

#         # for: accessible auxiliary information
#         self.ai = ai
#         self.pair = []
#         for isource in range(len(self.ai)):
#             for igroup in range(len(self.ai[isource])):
#                 self.pair.append((isource,igroup))
#         self.group_num = len(self.pair)

#     def _ai_conversion(self):

#         1

#         # if robust is False:
#         #     alpha_varcov = np.linalg.inv(infom11-np.dot(w.T,infom21))
#         # else:
#         #     infom_ortho = infom11-np.dot(w.T,infom21)
#         #     v = np.zeros(shape=(self.p,p1))
#         #     v[index,:] = np.eye(p1)
#         #     v[np.delete(np.arange(self.p),index),:] = - w
#         #     influence_score_ortho = self._partial_likelihood_derivative_influence(
#         #         beta=beta).dot(v)
#         #     influence = influence_score_ortho.dot(np.linalg.inv(infom_ortho))
#         #     alpha_varcov = influence.T.dot(influence)/self.n





