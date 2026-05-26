
import numpy as np
from . import _utils
from . import cox
from . import km



#############################################################
# %% high dimensional cox model without auxiliary information
class _(cox._):
    
    ## initial function: attributes 
    def __init__(self,y,delta,X,split=False,repeat=30):
        
        # inherit attributes from its father
        cox._.__init__(self,y=y,delta=delta,X=X)
        
        # for: sample splitting
        self.split = split
        self.repeat = repeat
        
        # for: utils
        self.tunpara_num = 30
        self.tunpara_scale = np.exp(np.linspace(
            start=np.log(20),stop=np.log(0.05),num=self.tunpara_num))

    ## estimation and inference on regression coefficients
    def coefficient(self,index=None):

        # prepare indices of interested coefficients
        if index is None:
            index = list(range(self.p))

        # judge the conduction of sample splitting
        if self.split is False:
        
            # -- lasso-based estimators for target parameters
            if 'beta_lasso' not in self._intermediate:
                self._intermediate['beta_lasso'] = self._lasso()
            beta_lasso = self._intermediate['beta_lasso']
            alpha_lasso = beta_lasso[index]

            # -- corrected estimator
            plik_d = self._partial_likelihood_derivative(beta=beta_lasso,d1=True,d2=True) 
            score = plik_d['d1']
            infom = - plik_d['d2']
            hd_alpha = np.zeros(shape=(self.p,len(index)))
            hd_alpha[index,:] = np.eye(len(index))
            if True:
                omega_alpha = self._projection(infom=infom,hd=hd_alpha) # infom_inv.dot(hd_alpha)
            else:
                infom_inv = np.linalg.inv(infom)
                omega_alpha = infom_inv.dot(hd_alpha)
            alpha = alpha_lasso + omega_alpha.T.dot(score)

            # - variance
            alpha_influence = self._coefficient_influence(
                beta=beta_lasso,index=index,omega=omega_alpha)
            alpha_avar = (alpha_influence**2).sum(axis=0)/self.n
            alpha_se = np.sqrt(alpha_avar/self.n)

        else:

            # fit across splitted samples
            model_list = self._split_model(index=index)
            alpha_list = [[[],[]] for i in range(self.repeat)]
            alpha_avar_list = [[[],[]] for i in range(self.repeat)]
            for s in range(self.repeat):
                for sc in range(2):
                    model_ = model_list[s][sc]
                    select_X__ = np.union1d(
                        index,self._intermediate['split_info']['select_X'][s][sc])
                    index_ = np.concatenate(
                        [np.where(select_X__ == a)[0] for a in index]).tolist()
                    model_.coefficient(index = index_)
                    alpha_list[s][sc]  = model_.result['coefficient']['estimate']
                    alpha_avar_list[s][sc] = (model_.result['coefficient']['se']**2)*(self.n/2)
                    
            # combine results derived from sample splitting
            alpha_list = [np.stack(a,axis=0).mean(axis=0) for a in alpha_list]
            alpha_avar_list = [np.stack(a,axis=0).mean(axis=0) for a in alpha_avar_list] 
            alpha,alpha_se = self._split_combine(tp_list=alpha_list,tp_avar_list=alpha_avar_list)

        # store the fitted results
        self.result['coefficient'] = {'estimate':alpha}
        self.result['coefficient']['se'] = alpha_se

    ## estimation and inference on hazard ratio
    def hazard_ratio(self,index=None):

        # prepare indices of interested coefficients
        if index is None:
            index = list(range(self.p))

        # judge the conduction of sample splitting
        if self.split is False:

            # -- lasso-based estimators for target parameters
            if 'beta_lasso' not in self._intermediate:
                self._intermediate['beta_lasso'] = self._lasso()
            beta_lasso = self._intermediate['beta_lasso']
            hr_lasso = np.exp(beta_lasso[index])

            # -- corrected estimator
            plik_d = self._partial_likelihood_derivative(beta=beta_lasso,d1=True,d2=True) 
            score = plik_d['d1']
            infom = - plik_d['d2']
            infom_inv = np.linalg.inv(infom)
            hd_alpha = np.zeros(shape=(self.p,len(index)))
            hd_alpha[index,:] = np.eye(len(index))
            hd_hr = hd_alpha.dot(np.diag(hr_lasso))
            hr = hr_lasso + hd_hr.T.dot(infom_inv).dot(score)

            # - variance
            hr_influence = self._hazard_ratio_influence(beta=beta_lasso,index=index)
            hr_avar = (hr_influence**2).sum(axis=0)/self.n
            hr_se = np.sqrt(hr_avar/self.n)

        else:

            # fit across splitted samples
            model_list = self._split_model(index=index)
            hr_list = [[[],[]] for i in range(self.repeat)]
            hr_avar_list = [[[],[]] for i in range(self.repeat)]
            for s in range(self.repeat):
                for sc in range(2):
                    model_ = model_list[s][sc]
                    select_X__ = np.union1d(
                        index,self._intermediate['split_info']['select_X'][s][sc])
                    index_ = np.concatenate(
                        [np.where(select_X__ == a)[0] for a in index]).tolist()
                    model_.hazard_ratio(index =index_)
                    hr_list[s][sc]  = model_.result['hazard_ratio']['estimate']
                    hr_avar_list[s][sc] = (model_.result['hazard_ratio']['se']**2)*(self.n/2)

            # combine results derived from sample splitting
            hr_list = [np.stack(a,axis=0).mean(axis=0) for a in hr_list]
            hr_avar_list = [np.stack(a,axis=0).mean(axis=0) for a in hr_avar_list] 
            hr,hr_se = self._split_combine(tp_list=hr_list,tp_avar_list=hr_avar_list)

        # store the fitted results
        self.result['hazard_ratio'] = {'estimate':hr}
        self.result['hazard_ratio']['se'] = hr_se

    ## estimation and inference on risk score
    def risk_score(self,x):
        
        # judge the conduction of sample splitting
        if self.split is False:

            # -- lasso-based estimators for target parameters
            if 'beta_lasso' not in self._intermediate:
                self._intermediate['beta_lasso'] = self._lasso()
            beta_lasso = self._intermediate['beta_lasso']
            rs_lasso = np.exp(x.dot(beta_lasso))

            # -- corrected estimator
            plik_d = self._partial_likelihood_derivative(beta=beta_lasso,d1=True,d2=True) 
            score = plik_d['d1']
            infom = - plik_d['d2']
            infom_inv = np.linalg.inv(infom)
            hd_rs = x.T.dot(np.diag(rs_lasso))
            rs = rs_lasso + hd_rs.T.dot(infom_inv).dot(score)

            # - variance
            rs_influence = self._risk_score_influence(beta=beta_lasso,x=x)
            rs_avar = (rs_influence**2).sum(axis=0)/self.n
            rs_se = np.sqrt(rs_avar/self.n)

        else:
            
            # fit across splitted samples
            model_list = self._split_model()
            rs_list = [[[],[]] for i in range(self.repeat)]
            rs_avar_list = [[[],[]] for i in range(self.repeat)]
            for s in range(self.repeat):
                for sc in range(2):
                    model_ = model_list[s][sc]
                    x_ = x[:,self._intermediate['split_info']['select_X'][s][sc]]
                    model_.risk_score(x=x_)
                    rs_list[s][sc]  = model_.result['risk_score']['estimate']
                    rs_avar_list[s][sc] = (model_.result['risk_score']['se']**2)*(self.n/2)

            # combine results derived from sample splitting
            rs_list = [np.stack(a,axis=0).mean(axis=0) for a in rs_list]
            rs_avar_list = [np.stack(a,axis=0).mean(axis=0) for a in rs_avar_list] 
            rs,rs_se = self._split_combine(tp_list=rs_list,tp_avar_list=rs_avar_list)

        # store the fitted results
        self.result['risk_score'] = {'estimate':rs}
        self.result['risk_score']['se'] = rs_se

    ## estimation and inference on baseline cumulative hazard function
    def baseline_cumulative_hazard(self,time):

        # judge the conduction of sample splitting
        if self.split is False:
        
            # -- lasso-based estimators for target parameters
            if 'beta_lasso' not in self._intermediate:
                self._intermediate['beta_lasso'] = self._lasso()
            beta_lasso = self._intermediate['beta_lasso']
            bch_lasso = self._Breslow(time=time,beta=beta_lasso)

            # -- corrected estimator
            plik_d = self._partial_likelihood_derivative(beta=beta_lasso,d1=True,d2=True) 
            score = plik_d['d1']
            infom = - plik_d['d2']
            infom_inv = np.linalg.inv(infom)
            eXb  = np.exp(np.dot(self.X,beta_lasso))
            XeXb = self.X*eXb.reshape((-1,1))
            SS0  = np.vstack([eXb[self.y>=yi].sum() for yi in self.y])/self.n
            SS1  = np.vstack([XeXb[self.y >= yi].sum(axis=0) 
                            for yi in self.y])/self.n
            hd_bch = - np.stack([np.sum(
                (SS1*(self.y<=x)[:,None])[self.delta==1,:]/(SS0[self.delta==1,:]**2),
                axis=0) for x in time],axis=1)/self.n
            bch = bch_lasso + hd_bch.T.dot(infom_inv).dot(score)

            # - variance
            bch_influence = self._baseline_cumulative_hazard_influence(
                beta=beta_lasso,time=time)
            bch_avar = (bch_influence**2).sum(axis=0)/self.n
            bch_se = np.sqrt(bch_avar/self.n)

        else:

            # fit across splitted samples
            model_list = self._split_model()
            bch_list = [[[],[]] for i in range(self.repeat)]
            bch_avar_list = [[[],[]] for i in range(self.repeat)]
            for s in range(self.repeat):
                for sc in range(2):
                    model_ = model_list[s][sc]
                    model_.baseline_cumulative_hazard(time=time)
                    bch_list[s][sc]  = model_.result['baseline_cumulative_hazard']['estimate']
                    bch_avar_list[s][sc] = (model_.result['baseline_cumulative_hazard']['se']**2)*(self.n/2)

            # combine results derived from sample splitting
            bch_list = [np.stack(a,axis=0).mean(axis=0) for a in bch_list]
            bch_avar_list = [np.stack(a,axis=0).mean(axis=0) for a in bch_avar_list] 
            bch,bch_se = self._split_combine(tp_list=bch_list,tp_avar_list=bch_avar_list)

        # store the fitted results
        self.result['baseline_cumulative_hazard'] = {'estimate':bch}
        self.result['baseline_cumulative_hazard']['se'] = bch_se   

    ## estimation and inference on baseline cumulative hazard function
    def survival_probability(self,time,x,cross=True):

        # judge the conduction of sample splitting
        if self.split is False:

            # -- lasso-based estimators for target parameters
            if 'beta_lasso' not in self._intermediate:
                self._intermediate['beta_lasso'] = self._lasso()
            beta_lasso = self._intermediate['beta_lasso']
            Lamt = self._Breslow(time=time,beta=beta_lasso)
            exb = np.exp(x.dot(beta_lasso))
            if cross is False:
                sp_lasso = np.exp(-Lamt*exb)
            else:
                sp_lasso = np.exp(-Lamt[:,None]*exb)

            # -- corrected estimator
            plik_d = self._partial_likelihood_derivative(beta=beta_lasso,d1=True,d2=True) 
            score = plik_d['d1']
            infom = - plik_d['d2']
            infom_inv = np.linalg.inv(infom)
            eXb  = np.exp(np.dot(self.X,beta_lasso))
            XeXb = self.X*eXb.reshape((-1,1))
            SS0  = np.vstack([eXb[self.y>=yi].sum() for yi in self.y])/self.n
            SS1  = np.vstack([XeXb[self.y >= yi].sum(axis=0) 
                            for yi in self.y])/self.n
            hd_bch = - np.stack([np.sum(
                (SS1*(self.y<=a)[:,None])[self.delta==1,:]/(SS0[self.delta==1,:]**2),
                axis=0) for a in time],axis=1)/self.n
            num_time = len(time)
            if cross is False:
                sp = np.zeros(num_time)
                for i in range(len(time)):
                    hd_sp = - sp_lasso[i]*exb[i]*(hd_bch[:,i]+Lamt[i]*x[i,:])  
                    sp[i] = sp_lasso[i] + hd_sp.dot(infom_inv).dot(score)
            else:
                num_x = x.shape[0]
                sp = np.zeros(shape=(num_time,num_x))
                for itime in range(len(time)):
                    hd_sp = - np.diag(sp_lasso[itime,:]*exb).dot(hd_bch[:,itime] + Lamt[itime]*x)
                    sp[itime,:] = sp_lasso[itime,:] + hd_sp.dot(infom_inv).dot(score)

            # - variance
            sp_influence = self._survival_probability_influence(
                beta=beta_lasso,time=time,x=x,cross=cross)
            sp_avar = (sp_influence**2).sum(axis=0)/self.n
            sp_se = np.sqrt(sp_avar/self.n)

        else:

            # fit across splitted samples
            model_list = self._split_model()
            sp_list = [[[],[]] for i in range(self.repeat)]
            sp_avar_list = [[[],[]] for i in range(self.repeat)]
            for s in range(self.repeat):
                for sc in range(2):
                    model_ = model_list[s][sc]
                    x_ = x[:,self._intermediate['split_info']['select_X'][s][sc]]
                    model_.survival_probability(
                        time=time,x=x_,cross=cross)
                    sp_list[s][sc]  = model_.result['survival_probability']['estimate']
                    sp_avar_list[s][sc] = (model_.result['survival_probability']['se']**2)*(self.n/2)

            # combine results derived from sample splitting
            sp_list = [np.stack(a,axis=0).mean(axis=0) for a in sp_list]
            sp_avar_list = [np.stack(a,axis=0).mean(axis=0) for a in sp_avar_list] 
            sp,sp_se = self._split_combine(tp_list=sp_list,tp_avar_list=sp_avar_list)

        # store the fitted results
        self.result['survival_probability'] = {'estimate':sp}
        self.result['survival_probability']['se'] = sp_se   

    ## coefficients with lasso
    def _lasso(self,rowidx=None,colidx=None):

        # preparations
        if rowidx is None:
            rowidx = np.arange(self.n)
        if colidx is None:
            colidx = np.arange(self.p)
        y = self.y[rowidx]
        delta = self.delta[rowidx]
        X = self.X[rowidx,:][:,colidx]
        n,p  = X.shape
        model_temp = cox._(y=y,delta=delta,X=X)
        
        # prepare candidate set of tuning parameters
        tunpara = np.sqrt(np.log(p)/n)*self.tunpara_scale
        
        # estimators over different tuning parameters
        beta_path = np.zeros((p,self.tunpara_num))
        ICs       = np.zeros(self.tunpara_num) 
        beta_init = np.zeros(p)
        for itune in range(self.tunpara_num):
            beta_old = beta_init
            for numit in range(int(self.maxit)):
                plik_d = model_temp._partial_likelihood_derivative(
                        beta=beta_old,d1=True,d2=True) 
                XXXtYYY = - np.dot(plik_d['d2'],beta_old) + plik_d['d1']
                beta_inner_old = beta_old
                for numit_inner in range(int(self.maxit)):
                    dev = 0.1 * ( 
                        np.dot(plik_d['d2'],beta_inner_old) + XXXtYYY )
                    beta_inner = _utils.Soft_Threshold(
                        x = beta_inner_old + dev,
                        tunpara = 0.1*tunpara[itune])
                    if np.abs(beta_inner-beta_inner_old).max()>self.eps:
                        beta_inner_old = beta_inner
                    else:
                        beta_c = beta_inner
                        break
                if (np.abs(beta_c-beta_old).max()>self.eps):
                    beta_old = beta_c
                else:
                    break
            beta_path[:,itune] = beta_init = beta_c
            ICs[itune] = - model_temp._partial_likelihood( # BIC
                beta=beta_c) + 0.5*np.log(n)*np.sum(np.abs(beta_c)>=self.eps)/n # 0.5*np.log(n)
            
        # final estimator with optimal information criterion
        beta = beta_path[:,np.argmin(ICs)]

        # returned values
        return beta

    def _projection(self,infom,hd):
        
        # prepare candidate set of tuning parameters
        tunpara = np.sqrt(np.log(self.p)/self.n)*self.tunpara_scale
        
        # do lasso for obtaining omega
        omega = np.zeros(shape=(hd.shape[0],hd.shape[1]))
        for j in range(omega.shape[1]):
            omega_path = np.zeros((self.p,self.tunpara_num))
            ICs       = np.zeros(self.tunpara_num) 
            omega_init = np.zeros(self.p)
            for itune in range(self.tunpara_num):
                omega_old = omega_init
                for numit in range(int(self.maxit)):
                    grad = infom.dot(omega_old)-hd[:,j]
                    omega_ = _utils.Soft_Threshold(
                        x = omega_old - 0.1*grad,
                        tunpara = 0.1*tunpara[itune])
                    if np.abs(omega_-omega_old).max()>self.eps:
                        omega_old = omega_
                    else:
                        break
                omega_path[:,itune] = omega_init = omega_
                ICs[itune] = 0.5*omega_.T.dot(infom).dot(omega_)-hd[:,j].dot(
                    omega_) + 0.5*np.log(self.n)*np.sum(np.abs(omega_)>=self.eps)/self.n # 0.5*np.log(self.n)
            
            # final estimator with optimal information criterion
            print(ICs)
            omega[:,j] = omega_path[:,np.argmin(ICs)]
            
            # returned quantity
            return omega
        

    ## split the data and prepare seleted samples and covariates
    def _split(self): 

        # split the original multiple times
        select_id = [[] for i in range(self.repeat)] 
        select_X  = [[[],[]] for i in range(self.repeat)]
        for s in range(self.repeat):

            # - labels of individuals
            select_id_pre = np.tile(np.arange(2),int(np.ceil(self.n/2)))[:self.n]
            np.random.shuffle(select_id_pre)
            select_id_ = [np.arange(self.n)[select_id_pre==i].tolist() 
                          for i in range(2)]
            select_id[s] = select_id_
            
            # - labels of covariates
            for sc in range(2):
                beta_lasso_screen = self._lasso(rowidx=select_id_[1-sc])
                # nscreen = np.max((int(np.ceil(np.sqrt(self.n/2)/np.log(self.p))),
                #                   np.sum(np.abs(beta_lasso_screen)>self.eps)))
                nscreen = np.sum(np.abs(beta_lasso_screen)>self.eps)
                select_X_ = np.argsort(-np.abs(beta_lasso_screen))[:nscreen]
                select_X[s][sc] = select_X_

        # combine results derived from sample splitting
        self._intermediate['split_info'] = {
            'select_id':select_id,'select_X':select_X}

    ## prepare splitted models
    def _split_model(self,index=None): 

        # obtain information concerning sample splitting
        if 'split_info' not in self._intermediate:
            self._split()

        # fit across splitted samples
        model_list = [[[],[]] for i in range(self.repeat)]
        for s in range(self.repeat):
                        
            # obtain estimators
            for sc in range(2):

                # extract current information concerning sample splitting
                select_id_ = self._intermediate['split_info']['select_id'][s][sc]
                select_X_  = self._intermediate['split_info']['select_X'][s][sc]
                select_X__ = select_X_ if index is None else np.union1d(index,select_X_)
                
                # fitted results based on selected features
                model_list[s][sc] = cox._(y = self.y[select_id_],
                                          delta = self.delta[select_id_],
                                          X = self.X[select_id_,:][:,select_X__])

        # return Instantiated models
        return model_list       

    ## combine results from multiple sample splitting
    def _split_combine(self,tp_list,tp_avar_list): 
            
        # combine results derived from sample splitting 
        tp = np.median(np.stack(tp_list,axis=0),axis=0)
        tp_avar = np.median(np.stack(
            [tp_avar_list[s] + (tp_list[s]-tp)**2 for s in range(self.repeat)],
            axis=0),axis=0)
        tp_se = np.sqrt(tp_avar/self.n)

        # return values
        return tp,tp_se     
    

# %% synthesis of survival probabilities from Kaplan-Meier estimator (kmsp)
class kmsp(_,cox.kmsp): 

    ## initial function: attributes 
    def __init__(self,y,delta,X,ai,
                 transition=['km','cox'][0],
                 degree=3,
                 split=False,
                 repeat=30):
        
        # inherit attributes from its father
        _.__init__(self,y=y,delta=delta,X=X,repeat=repeat,split=split)
        cox.kmsp.__init__(self,y=y,delta=delta,X=X,ai=ai,transition=transition,degree=degree)

        # for: original model
        if split is False:
            self.model_ori = _(y=self.y,delta=self.delta,X=self.X,split=False)

    ## estimation and inference on regression coefficients
    def coefficient(self,index=None):
        
        # prepare indices of interested coefficients
        if index is None:
            index = list(range(self.p))

        # judge the conduction of sample splitting
        if self.split is False:

            # original (classical) estimator and its influence function
            self.model_ori.coefficient(index=index)
            alpha_ori = self.model_ori.result['coefficient']['estimate']
            beta_lasso = self.model_ori._intermediate['beta_lasso']
            alpha_ori_influence = self.model_ori._coefficient_influence(
                index=index,beta=beta_lasso)

            # estimator with auxiliary information
            alpha,alpha_se = self._ais(
                tp=alpha_ori,tp_influence=alpha_ori_influence,split=False)

        else:

            # fit across splitted samples to obtain original (classical) estimator
            model_list = self._split_model(index=index)
            alpha_ori_list = [[[],[]] for i in range(self.repeat)]
            alpha_ori_influence_list = [[[],[]] for i in range(self.repeat)]
            for s in range(self.repeat):
                for sc in range(2):
                    model_ = model_list[s][sc]
                    select_X__ = np.union1d(
                        index,self._intermediate['split_info']['select_X'][s][sc])
                    index_ = np.concatenate(
                        [np.where(select_X__ == a)[0] for a in index]).tolist()
                    model_.coefficient(index=index_)
                    alpha_ori_list[s][sc] = model_.result['coefficient']['estimate']
                    alpha_ori_influence_list[s][sc] = model_._coefficient_influence(
                        beta=model_._intermediate['beta_lasso'],index=index_)
            
            # synthesis of auxiliary information
            alpha,alpha_se = self._ais(
                tp=alpha_ori_list,tp_influence=alpha_ori_influence_list,
                split=True,index=index)
        
        # store the fitted result
        self.result['coefficient'] = {'estimate':alpha}
        self.result['coefficient']['se'] = alpha_se
        
    ## estimation and inference on hazard ratio
    def hazard_ratio(self,index=None):
        
        # prepare indices of interested coefficients
        if index is None:
            index = list(range(self.p))

        # judge the conduction of sample splitting
        if self.split is False:

            # original (classical) estimator and its influence function
            self.model_ori.hazard_ratio(index=index)
            hr_ori = self.model_ori.result['hazard_ratio']['estimate']
            beta_lasso = self.model_ori._intermediate['beta_lasso']
            hr_ori_influence = self.model_ori._hazard_ratio_influence(
                index=index,beta=beta_lasso)

            # estimator with auxiliary information
            hr,hr_se = self._ais(
                tp=hr_ori,tp_influence=hr_ori_influence,split=False)

        else:

            # fit across splitted samples to obtain original (classical) estimator
            model_list = self._split_model(index=index)
            hr_ori_list = [[[],[]] for i in range(self.repeat)]
            hr_ori_influence_list = [[[],[]] for i in range(self.repeat)]
            for s in range(self.repeat):
                for sc in range(2):
                    model_ = model_list[s][sc]
                    select_X__ = np.union1d(
                        index,self._intermediate['split_info']['select_X'][s][sc])
                    index_ = np.concatenate(
                        [np.where(select_X__ == a)[0] for a in index]).tolist()
                    model_.hazard_ratio(index=index_)
                    hr_ori_list[s][sc] = model_.result['hazard_ratio']['estimate']
                    hr_ori_influence_list[s][sc] = model_._hazard_ratio_influence(
                        beta=model_._intermediate['beta_lasso'],index=index_)
                    
            # synthesis of auxiliary information
            hr,hr_se = self._ais(
                tp=hr_ori_list,tp_influence=hr_ori_influence_list,
                split=True,index=index)
        
        # store the fitted result
        self.result['hazard_ratio'] = {'estimate':hr}
        self.result['hazard_ratio']['se'] = hr_se

    ## estimation and inference on risk score
    def risk_score(self,x):
        
        # judge the conduction of sample splitting
        if self.split is False:

            # original (classical) estimator and its influence function
            self.model_ori.risk_score(x=x)
            rs_ori = self.model_ori.result['risk_score']['estimate']
            beta_lasso = self.model_ori._intermediate['beta_lasso']
            rs_ori_influence = self.model_ori._risk_score_influence(
                x=x,beta=beta_lasso)

            # estimator with auxiliary information
            rs,rs_se = self._ais(
                tp=rs_ori,tp_influence=rs_ori_influence,split=False)

        else:
        
            # fit across splitted samples to obtain original (classical) estimator
            model_list = self._split_model()
            rs_ori_list = [[[],[]] for i in range(self.repeat)]
            rs_ori_influence_list = [[[],[]] for i in range(self.repeat)]
            for s in range(self.repeat):
                for sc in range(2):
                    model_ = model_list[s][sc]
                    x_ = x[:,self._intermediate['split_info']['select_X'][s][sc]]
                    model_.risk_score(x=x_)
                    rs_ori_list[s][sc] = model_.result['risk_score']['estimate']
                    rs_ori_influence_list[s][sc] = model_._risk_score_influence(
                        beta=model_._intermediate['beta_lasso'],x=x_)
                    
            # synthesis of auxiliary information
            rs,rs_se = self._ais(
                tp=rs_ori_list,tp_influence=rs_ori_influence_list,split=True)
        
        # store the fitted result
        self.result['risk_score'] = {'estimate':rs}
        self.result['risk_score']['se'] = rs_se
       
    ## estimation and inference on baseline cumulative hazard
    def baseline_cumulative_hazard(self,time):
        
        # judge the conduction of sample splitting
        if self.split is False:

            # original (classical) estimator and its influence function
            self.model_ori.baseline_cumulative_hazard(time=time)
            bch_ori = self.model_ori.result['baseline_cumulative_hazard']['estimate']
            beta_lasso = self.model_ori._intermediate['beta_lasso']
            bch_ori_influence = self.model_ori._baseline_cumulative_hazard_influence(
                time=time,beta=beta_lasso)

            # estimator with auxiliary information
            bch,bch_se = self._ais(
                tp=bch_ori,tp_influence=bch_ori_influence,split=False)

        else:
            
            # fit across splitted samples to obtain original (classical) estimator
            model_list = self._split_model()
            bch_ori_list = [[[],[]] for i in range(self.repeat)]
            bch_ori_influence_list = [[[],[]] for i in range(self.repeat)]
            for s in range(self.repeat):
                for sc in range(2):
                    model_ = model_list[s][sc]
                    model_.baseline_cumulative_hazard(time=time)
                    bch_ori_list[s][sc] = model_.result['baseline_cumulative_hazard']['estimate']
                    bch_ori_influence_list[s][sc] = model_._baseline_cumulative_hazard_influence(
                        beta=model_._intermediate['beta_lasso'],time=time)
                    
            # synthesis of auxiliary information
            bch,bch_se = self._ais(
                tp=bch_ori_list,tp_influence=bch_ori_influence_list,split=True)
        
        # store the fitted result
        self.result['baseline_cumulative_hazard'] = {'estimate':bch}
        self.result['baseline_cumulative_hazard']['se'] = bch_se
        
    ## estimation and inference on survival probability
    def survival_probability(self,time,x,cross=True):
        
        # judge the conduction of sample splitting
        if self.split is False:

            # original (classical) estimator and its influence function
            self.model_ori.survival_probability(time=time,x=x,cross=cross)
            sp_ori = self.model_ori.result['survival_probability']['estimate']
            beta_lasso = self.model_ori._intermediate['beta_lasso']
            sp_ori_influence = self.model_ori._survival_probability_influence(
                time=time,x=x,beta=beta_lasso,cross=cross)

            # estimator with auxiliary information
            if cross is False:
                sp,sp_se = self._ais(tp=sp_ori,tp_influence=sp_ori_influence,split=False)
            else:
                num_time = len(time)
                sp_list = [[] for a in range(num_time)]
                sp_se_list = [[] for a in range(num_time)]
                for itime in range(num_time):
                    sp_list[itime],sp_se_list[itime] = self._ais(
                        tp=sp_ori[itime,:],tp_influence=sp_ori_influence[:,itime,:],split=False)
                sp = np.stack(sp_list,axis=0)
                sp_se = np.stack(sp_se_list,axis=0)
                
        else:

            # fit across splitted samples to obtain original (classical) estimator
            model_list = self._split_model()
            sp_ori_list = [[[],[]] for i in range(self.repeat)]
            sp_ori_influence_list = [[[],[]] for i in range(self.repeat)]
            for s in range(self.repeat):
                for sc in range(2):
                    model_ = model_list[s][sc]
                    x_ = x[:,self._intermediate['split_info']['select_X'][s][sc]]
                    model_.survival_probability(time=time,x=x_,cross=cross)
                    sp_ori_list[s][sc] = model_.result['survival_probability']['estimate']
                    sp_ori_influence_list[s][sc] = model_._survival_probability_influence(
                        beta=model_._intermediate['beta_lasso'],time=time,x=x_,cross=cross)
                    
            # synthesis of auxiliary information
            if cross is False:
                sp,sp_se = self._ais(
                    tp=sp_ori_list,tp_influence=sp_ori_influence_list,split=True)
            else:
                num_time = len(time)
                num_x = x.shape[0]
                sp = np.zeros(shape=(num_time,num_x))
                sp_se = np.zeros(shape=(num_time,num_x))
                for itime in range(num_time):
                    sp_ori_list_ = [[sp_ori_list[i][0][itime,:],sp_ori_list[i][1][itime,:]] 
                                    for i in range(self.repeat)]
                    sp_ori_influence_list_ = [[sp_ori_influence_list[i][0][:,itime,:],
                                            sp_ori_influence_list[i][1][:,itime,:]] 
                                            for i in range(self.repeat)]
                    sp[itime,:],sp_se[itime,:] = self._ais(
                        tp=sp_ori_list_,tp_influence=sp_ori_influence_list_,split=True)   

        # store the fitted results
        self.result['survival_probability'] = {'estimate':sp}
        self.result['survival_probability']['se'] = sp_se
        
    ## split the data and prepare seleted samples and covariates
    def _split(self):

        # inherit elements from the father class
        _._split(self)

        # prepare sample splitting adapted auxiliary information
        ai = [[[],[]] for i in range(self.repeat)]
        for s in range(self.repeat):
            for sc in range(2):
                select_id_ = self._intermediate['split_info']['select_id'][s][sc]
                ai_ = [[] for i in range(len(self.ai))] 
                for ig in range(self.group_num):
                    group_ = self.ai[self.pair[ig][0]][self.pair[ig][1]].copy()
                    id_ = np.array([i in group_['id'] for i in range(self.n)])
                    group_['id'] = np.where(id_[select_id_])[0]
                    ai_[self.pair[ig][0]].append(group_)
                ai[s][sc] = ai_
        self._intermediate['split_info']['ai'] = ai

    ## conversion of auxiliary information (into transitional elements)
    def _ai_conversion_transition(self,index=None):
        
        if self.split is False:
        
            # transitional estimator based on different approaches
            if self.transition in ['km']: 
                
                # - transitional estimator based on Kaplan-Meier estimator
                trans_list,trans_influence_list = km.kmsp._ai_conversion_transition(self)
    
            else:
    
                # - preparations
                if 'beta_lasso' not in self.model_ori._intermediate:
                    self.model_ori._intermediate['beta_lasso'] = self.model_ori._lasso()
                beta_lasso = self.model_ori._intermediate['beta_lasso']
                Xb   = np.dot(self.X,beta_lasso)
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
        
        else:
            
            
            # obtain information concerning sample splitting
            if 'split_info' not in self._intermediate:
                self._split()
            
            # extract transitional elements across all models
            trans_list_pre = [[[],[]] for i in range(self.repeat)]
            trans_influence_list_pre = [[[],[]] for i in range(self.repeat)]
            for s in range(self.repeat):
                for sc in range(2):
                    
                    # -- extract current information concerning sample splitting
                    ai_        = self._intermediate['split_info']['ai'][s][sc]
                    select_id_ = self._intermediate['split_info']['select_id'][s][sc]
                    select_X_  = self._intermediate['split_info']['select_X'][s][sc]
                    select_X__ = select_X_ if index is None else np.union1d(index,select_X_)
    
                    # -- fitted results based on selected features
                    model_ = kmsp(y = self.y[select_id_],
                                  delta = self.delta[select_id_],
                                  X = self.X[select_id_,:][:,select_X__],
                                  ai = ai_,
                                  transition=self.transition,
                                  split=False,
                                  degree=self.degree)
                    trans_list_pre[s][sc],trans_influence_list_pre[s][sc] = model_._ai_conversion_transition()
         
            # combine results under each sample splitting
            trans_list = [[] for i in range(self.repeat)]
            trans_influence_list = [[] for i in range(self.repeat)]
            for s in range(self.repeat): 
                trans_list[s] = [(trans_list_pre[s][0][i]+trans_list_pre[s][1][i])/2 
                                 for i in range(self.group_num)]
                trans_influence_list[s] = [np.vstack([trans_influence_list_pre[s][0][i],
                                                      trans_influence_list_pre[s][1][i]]) 
                                           for i in range(self.group_num)]

        # returned values  
        return  trans_list,trans_influence_list

    ## synthesis of auxiliary information (general function for different target parameters)
    def _ais(self,tp,tp_influence,split=False,index=None):

        # process the auxiliary information into needed elements
        if split is False:
            
            # process the auxiliary information into needed elements
            if ('ai_conversion' not in self._intermediate) or (self.split is True):
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
            
        else:
            
            # obtain lists of transitional elements
            trans_list,trans_influence_list = self._ai_conversion_transition(index=index)
            
            # synthesis of auxiliary information
            tp_new_list = [[] for i in range(self.repeat)]
            tp_new_avar_list = [[] for i in range(self.repeat)]
            for s in range(self.repeat): 
                tp_ = (tp[s][0]+tp[s][1])/2
                tp_influence_ = np.vstack(tp_influence[s])
                self._intermediate['trans'] = trans_list[s],trans_influence_list[s]
                tp_new_,tp_new_se_ = self._ais(
                    tp=tp_,tp_influence=tp_influence_,split=False)
                tp_new_list[s] = tp_new_
                tp_new_avar_list[s] = (tp_new_se_**2)*self.n
    
            # combine results derived from sample splitting
            tp_new,tp_new_se = self._split_combine(tp_list=tp_new_list,tp_avar_list=tp_new_avar_list)

        # returned values
        return tp_new,tp_new_se



