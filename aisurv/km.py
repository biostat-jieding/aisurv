
import numpy as np
from . import _utils

############################################################
# %% Kaplan-Meier estimator without any axiliary information
class _:
    
    ## initial function: attributes 
    def __init__(self,y,delta):  
        
        # for: data of onservations
        self.y=y
        self.delta=delta

        # for: utils attributes
        self.n = len(y)
        
        # for: initialize fitted results
        self.result = {}
        
        # for: stored intermediate quantities (share across different methods)
        self._intermediate = {}
        
    ## estimation and inference on survival probability at given time
    def survival_probability(self,time):  # 修改: 更加精细化, 输入index of subgroup

        # calculate the Kaplan-Meier estimator
        sp = self._km(time=time)
        
        # variance based on influence function
        sp_influence = self._survival_probability_influence(time=time)
        sp_avar = (sp_influence**2).sum(axis=0)/self.n
        sp_se = np.sqrt(sp_avar/self.n)  # 修改: 添加判断是否输出方差 (默认输出)
        
        # store fitted values
        self.result['survival_probability'] = {'estimate':sp}
        self.result['survival_probability']['se'] = sp_se

    ## calculate the Kaplan-Meier estimator
    def _km(self,time,type='right'):
        
        # preparation
        t_jump = np.sort(np.unique(self.y[self.delta==1]))
        d = np.array([np.sum(self.y==x) for x in t_jump])
        R = np.array([np.sum(self.y>=x) for x in t_jump])
        prods = 1-d/R
            
        # calculate the values of KM at pre-specified time points tm
        if type == 'right':
            St = np.array([np.prod(prods[t_jump<=x]) for x in time])
        else:
            St = np.array([np.prod(prods[t_jump<x]) for x in time])
        
        # store estimate
        return St

    ## influence function of proposed estimator for survival probability
    def _survival_probability_influence(self,time):
        
        # preparation and influence function
        Hy = np.array([np.sum(self.y<=x) for x in self.y])/self.n
        Hy[Hy==1] = float('inf')
        St = self._km(time=time)
        temp_1 = np.stack([self.delta*(self.y<=x)/(1-Hy) for x in time],axis=1)
        temp_2 = np.stack([[
            np.sum(self.delta*(self.y<=np.min([x,yi]))/((1-Hy)**2))/self.n
            for yi in self.y] for x in time],axis=1)
        influence = (temp_2-temp_1)*St
        
        # returned values
        return influence

###########################################################################
# %% synthesis of survival probabilities from Kaplan-Meier estimator (kmsp)
class kmsp(_):

    ## initial function: attributes 
    def __init__(self,
                 y,delta,ai,
                 degree=3):
        
        # inherit attributes from its father
        _.__init__(self,y=y,delta=delta)

        # for: accessible auxiliary information
        self.ai = ai
        self.pair = []
        for isource in range(len(self.ai)):
            for igroup in range(len(self.ai[isource])):
                self.pair.append((isource,igroup))
        self.group_num = len(self.pair)

        # for: parameters used in processing auxiliary information
        self.degree = degree

        # for: original model
        self.model_ori = _(y=self.y,delta=self.delta)

    ## estimation and inference on survival probability at given time
    def survival_probability(self,time): 
        
        # original (classical) estimator
        self.model_ori.survival_probability(time=time)
        sp_ori = self.model_ori.result['survival_probability']['estimate']
        sp_ori_influence = self.model_ori._survival_probability_influence(time=time)
        
        # estimator with auxiliary information
        sp,sp_se = self._ais(
            tp=sp_ori,tp_influence=sp_ori_influence)
        self.result['survival_probability'] = {'estimate':sp}
        self.result['survival_probability']['se'] = sp_se  

    ## synthesis of auxiliary information (general function for different target parameters)
    def _ais(self,tp,tp_influence):

        # process the auxiliary information into needed elements
        if 'ai_conversion' not in self._intermediate:
            self._ai_conversion()
        trans_influence = self._intermediate['ai_conversion']['trans_influence']
        cv = self._intermediate['ai_conversion']['cv']
        cv_varcov_inv = self._intermediate['ai_conversion']['cv_varcov_inv']

        # define the improved estimator formally
        tp_trans_cov = trans_influence.T.dot(tp_influence)/self.n
        W = tp_trans_cov.T.dot(cv_varcov_inv)
        tp_new = tp-W.dot(cv)
        
        # estimator of variance-covariance matrix
        tp_avar = (tp_influence**2).sum(axis=0)/self.n
        avar_decrease = np.diag(tp_trans_cov.T.dot(cv_varcov_inv).dot(tp_trans_cov))
        tp_new_avar = tp_avar - avar_decrease
        tp_new_se = np.sqrt(tp_new_avar/self.n)

        # returned values
        return tp_new,tp_new_se

    ## conversion of auxiliary information (and original data) into needed quantities
    def _ai_conversion(self):

        # prepare transitional elements 
        if 'trans' not in self._intermediate:
            self._intermediate['trans'] = self._ai_conversion_transition()
        trans_list,trans_influence_list = self._intermediate['trans']

        # formally calculate necessary quantities
        sp_list = [[] for _ in range(self.group_num)]
        sp_influence_list = [[] for _ in range(self.group_num)]
        source_list = [[] for _ in range(self.group_num)]
        rho_list = [[] for _ in range(self.group_num)]
        for ig in range(self.group_num):
            
            # - setups of the current subgroup
            group_ = self.ai[self.pair[ig][0]][self.pair[ig][1]]
            id_ = group_['id']
            id_prop = len(id_)/self.n
            t_ = group_['t']
            sp_ = group_['sp']
            size_ = group_['size'] if 'size' in group_ else float('inf')
                        
            # - influence function of such auxiliary information (for uncertainty)
            if size_ <  float('inf'):
                sp_influence_ = np.zeros(shape=(self.n,len(t_)))
                sp_influence_[id_,:] = _(y=self.y[id_],delta=self.delta[id_]
                                         )._survival_probability_influence(
                                             time=t_)/id_prop

            # - transitional estimator
            trans_ = trans_list[ig]
            trans_influence_ = trans_influence_list[ig]

            # - conversion of transitional elements if curve == TRUE            
            curve_ = group_['curve'] if 'curve' in group_ else False
            if curve_ is True:
                
                # -- basis over needed points
                t_min = np.min(group_['t'])
                t_max = np.max(group_['t'])
                basis = np.stack([
                    _utils.basis_Bernstein(x=t_,k=k,d=self.degree,il=t_min,ir=t_max)
                    for k in range(self.degree+1)],axis=0)
                
                # -- conversion of previous functional quantities
                t_num = len(t_)
                sp_ = np.sum((basis*sp_)[:,:(t_num-1)]*np.diff(t_),axis=1)
                trans_ = np.sum((basis*trans_)[:,:(t_num-1)]*np.diff(t_),axis=1)
                trans_influence_ = np.stack([
                    np.sum((basis*trans_influence_[i,:])[:,:(t_num-1)]*np.diff(t_),
                    axis=1) for i in range(self.n)],axis=0)
                if size_ <  float('inf'):
                    sp_influence_ = np.stack([
                        np.sum((basis*sp_influence_[i,:])[:,:(t_num-1)]*np.diff(t_),
                        axis=1) for i in range(self.n)],axis=0)
                
            # - combine needed quantities together   
            sp_list[ig] = sp_
            sp_influence_list[ig] = sp_influence_ if size_ < float('inf') else np.zeros(shape=(self.n,len(sp_)))
            source_list[ig] = [self.pair[ig][0]]*len(sp_)
            rho_list[ig] = [len(id_)/size_]*len(sp_)
            trans_list[ig] = trans_
            trans_influence_list[ig] = trans_influence_

        # auxiliary information and other quantities for all subgroups
        sp = np.concatenate(sp_list) 
        sp_influence = np.hstack(sp_influence_list)
        trans = np.concatenate(trans_list)
        trans_influence = np.hstack(trans_influence_list)
        source = np.concatenate(source_list) 
        rho = np.concatenate(rho_list) 
        
        # matrix used for the adjustment of uncertainty
        trans_colnum = trans_influence.shape[1]
        sp_varcov = np.zeros((trans_colnum,trans_colnum))
        if np.any(rho != 0) == True:
            rho_sqrt_diag = np.diag(np.sqrt(rho))
            for isource in range(len(self.ai)):
                idx = np.where(source == isource)[0]
                idx_mask = np.ix_(idx,idx)
                sp_varcov[idx_mask] = (sp_influence[:,idx].dot(
                    rho_sqrt_diag[idx_mask])).T.dot(sp_influence[:,idx].dot(
                        rho_sqrt_diag[idx_mask]))/self.n
                        
        # return calculated and subsequently needed quantities
        cv = trans - sp
        trans_varcov = trans_influence.T.dot(trans_influence)/self.n     
        cv_varcov = trans_varcov + sp_varcov
        cv_varcov_inv = np.linalg.inv(cv_varcov)
        self._intermediate['ai_conversion'] = {
            'trans_influence':trans_influence,
            'cv':cv,
            'cv_varcov_inv':cv_varcov_inv}

    # conversion of auxiliary information (transitional elements) via model-based idea
    def _ai_conversion_transition(self):

        # list of transitional elements
        trans_list = [[] for _ in range(self.group_num)]
        trans_influence_list = [[] for _ in range(self.group_num)]
        for ig in range(self.group_num):
            
            # - setups of the current subgroup
            group_ = self.ai[self.pair[ig][0]][self.pair[ig][1]]
            id_ = group_['id']
            id_prop = len(id_)/self.n
            t_ = group_['t']
            
            # -- specific estiamtes for transitional estimators
            model_km = _(y=self.y[id_],delta=self.delta[id_])
            trans_ = model_km._km(time=t_)

            # -- influence functions
            trans_influence_ = np.zeros(shape=(self.n,len(t_)))
            trans_influence_[id_,:] = model_km._survival_probability_influence(
                time=t_)/id_prop

            # store estimated quantities
            trans_list[ig] = trans_
            trans_influence_list[ig] = trans_influence_
            
        return  trans_list,trans_influence_list





