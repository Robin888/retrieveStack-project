# -*- coding: utf-8 -*-
"""
Created on Thu Apr 06 20:19:01 2017

@author: Ruobing
"""
import pandas as pd
import numpy as np
from baseFunctions import stackKfold,evaluation
from scipy.optimize import minimize
from scipy import stats


class modelInfo(object):
    def __init__(self,model,train_x,train_y,test_x,
                 trainX_output,testX_output,metric,score):
        
        self.model = model
        self.train_x = train_x
        self.train_y = train_y
        self.test_x=test_x
        self.trainX_output = trainX_output
        self.testX_output=testX_output
        self.metric = metric
        self.score = score





def ensemble(layer_name,models,train_x,train_y,test_x,n_classes,n_folds,
             metric_for_classifier='log_loss',metric_for_regressor='mean_squared_error'):
    new_blend_train_classifier=[]
    new_blend_test_classifier=[]
    new_blend_train_regressor=[]
    new_blend_test_regressor=[]
    models_info=[]
    for i,model in enumerate(models):
        print i,model
        new_train,new_test,score=stackKfold(model,train_x,train_y,test_x,n_classes,n_folds,
                                            metric_for_classifier,metric_for_regressor)
        
        if 'Classi' in str(model): 
            print 'the',metric_for_classifier,'of',layer_name,'model',i,'is',score
            new_blend_train_classifier.append(new_train)
            new_blend_test_classifier.append(new_test)
            models_info.append(modelInfo(model=model,train_x=train_x,train_y=train_y,test_x=test_x,
                                         trainX_output=new_train,testX_output=new_test,
                                         metric=metric_for_classifier,score=score))
        
        if 'Regres' in str(model): 
            print 'the',metric_for_regressor,'of',layer_name,'model',i,'is',score
            new_blend_train_regressor.append(new_train)
            new_blend_test_regressor.append(new_test)
            models_info.append(modelInfo(model=model,train_x=train_x,train_y=train_y,test_x=test_x,
                                         trainX_output=new_train,testX_output=new_test,
                                         metric=metric_for_regressor,score=score))
        
            
        print '*******************************************************'
    
    if len(new_blend_train_classifier)!=0:
        cols_for_classifier=[]    
        for i in range(len(new_blend_train_classifier)):
            for label in range(n_classes):
                cols_for_classifier.append(layer_name+'_classifier_'+str(i)+'_class_'+str(label))    
        new_blend_train_classifier=pd.DataFrame(np.concatenate(new_blend_train_classifier,axis=1),columns=cols_for_classifier)
        new_blend_test_classifier=pd.DataFrame(np.concatenate(new_blend_test_classifier,axis=1),columns=cols_for_classifier)
    else:
        new_blend_train_classifier=pd.DataFrame()
        new_blend_test_classifier=pd.DataFrame()
    
    if len(new_blend_train_regressor)!=0:
        cols_for_regressor=[]
        for i in range(len(new_blend_train_regressor)):
            cols_for_regressor.append(layer_name+'_regressor_'+str(i))
        new_blend_train_regressor=pd.DataFrame(np.transpose(new_blend_train_regressor),columns=cols_for_regressor)
        new_blend_test_regressor=pd.DataFrame(np.transpose(new_blend_test_regressor),columns=cols_for_regressor)
    else:
        new_blend_train_regressor=pd.DataFrame()
        new_blend_test_regressor=pd.DataFrame()
        
    new_blend_train=pd.concat([new_blend_train_classifier,new_blend_train_regressor],axis=1)
    new_blend_test=pd.concat([new_blend_test_classifier,new_blend_test_regressor],axis=1)
    
    return  new_blend_train,new_blend_test,models_info


def retrieve(all_infoList_list,metric='log_loss',maximize=False):
    good=[e for e in all_infoList_list[-1] if e.metric==metric]
    if maximize==False:
        threshold=max([e.score for e in all_infoList_list[-1]])
        for info in all_infoList_list[0:len(all_infoList_list)-1]:
            for e in info:
                if e.metric==metric and e.score<threshold:
                    good.append(e)
    else:
        threshold=min([e.score for e in all_infoList_list[-1]])
        for info in all_infoList_list[0:len(all_infoList_list)-1]:
            for e in info:
                if e.metric==metric and e.score>threshold:
                    good.append(e)
    return good



def opt_weight(modelInfo_list,train_y,metric='log_loss',method='SLSQP',maximize=False):
    
    def weighted_eval_minimize(weights):
        ''' scipy minimize will pass the weights as a numpy array '''
        entry=[e.trainX_output for e in modelInfo_list]
        final_prediction= sum([weights[i]*entry[i] for i in range(len(weights))])
        return evaluation(train_y, final_prediction,metric)

    def weighted_eval_maximize(weights):
        ''' scipy minimize will pass the weights as a numpy array '''
        entry=[e.trainX_output for e in modelInfo_list]
        final_prediction= sum([weights[i]*entry[i] for i in range(len(weights))])
        return evaluation(train_y, final_prediction,metric)*-1   
    
    
    starting_values = np.array([0.2]*len(modelInfo_list))
    cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
    bounds = [(0,1)]*len(modelInfo_list)
    if maximize==False:   
        res = minimize(weighted_eval_minimize, starting_values, method=method, bounds=bounds, constraints=cons)
        best_score=res['fun']
        weights=res['x']
        print 'The score after weighted averaging is',best_score
        return best_score,weights
    else:
        res = minimize(weighted_eval_maximize, starting_values, method=method, bounds=bounds, constraints=cons)
        best_score=res['fun']
        weights=res['x']
        print 'The score after weighted averaging is',best_score
        return best_score, weights


def predict(modelInfo_list,weights,majority_voting=False):
    pred_of_all=[]
    for model_info in modelInfo_list:
        model_info.model.fit(model_info.train_x,model_info.train_y)
        pred_of_all.append(model_info.model.predict(model_info.test_x))
    if majority_voting==False:
        pred=sum([weights[i]*pred_of_all[i] for i in range(len(weights))])
    else:
        mode=stats.mode(np.matrix(pred_of_all))
        pred=mode[0][0]
    return pred
    
    
def predict_proba(modelInfo_list,weights):
    pred_of_all=[]
    for model_info in modelInfo_list:
        model_info.model.fit(model_info.train_x,model_info.train_y)
        pred_of_all.append(model_info.model.predict_proba(model_info.test_x))
    pred=sum([weights[i]*pred_of_all[i] for i in range(len(weights))])
    return pred