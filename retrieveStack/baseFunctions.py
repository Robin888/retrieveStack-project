from sklearn.metrics import accuracy_score,auc,f1_score,log_loss,roc_auc_score,mean_absolute_error,mean_squared_error,median_absolute_error
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from scipy.optimize import minimize



def evaluation(y_true,y_predict,metric):
    if metric=='accuracy_score':
        return accuracy_score(y_true,y_predict)
    if metric=='auc':
        return auc(y_true,y_predict)
    if metric=='f1_score':
        return f1_score(y_true,y_predict)
    if metric=='log_loss':
        return log_loss(y_true,y_predict)
    if metric=='roc_auc_score':
        return roc_auc_score(y_true,y_predict)
    if metric=='auc':
        return auc(y_true,y_predict)
    if metric=='mean_absolute_error':
        return mean_absolute_error(y_true,y_predict)
    if metric=='mean_squared_error':
        return mean_squared_error(y_true,y_predict)
    if metric=='median_absolute_error':
        return median_absolute_error(y_true,y_predict)
        

# output the new train,new test and CV score of a single model in stacking
def stackKfold(model,train_x,train_y,test_x,n_classes,n_folds,
               metric_for_classifier='log_loss',metric_for_regressor='mean_squared_error'):
    
    
    skf = StratifiedKFold(n_folds)
        
    
    if 'Classi' in str(model) or 'NB'  in str(model) or 'Logistic' in str(model):
        i=0 # fold number for each base model
        dataset_blend_train = np.zeros((train_x.shape[0], n_classes))
        dataset_blend_test_all = []
        scores_of_base_model=[]
        for train_id,test_id in skf.split(train_x,train_y):
            print "Fold",i,'-',
            X_train = train_x.iloc[train_id]
            y_train = train_y.iloc[train_id]
                
            X_test = train_x.iloc[test_id]
            y_test = train_y.iloc[test_id]
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)
            dataset_blend_train[test_id] = y_pred
            dataset_blend_test_all.append(model.predict_proba(test_x))
            i=i+1
            scores_of_base_model.append(evaluation(y_test,y_pred,metric_for_classifier))
        print 'All folds Done!'
        score=np.mean(scores_of_base_model)
        
        # each folds do a prediction of a single base model, store average as the prediction of the test_x
        
    
    if 'Regres' in str(model) and 'Logistic' not in str(model):
        i=0 # fold number for each base model
        dataset_blend_train = np.zeros((train_x.shape[0]))
        dataset_blend_test_all = []
        scores_of_base_model=[]
        for train_id,test_id in skf.split(train_x,train_y):
            print "Fold",i,'-',
            X_train = train_x.iloc[train_id]
            y_train = train_y.iloc[train_id]
                
            X_test = train_x.iloc[test_id]
            y_test = train_y.iloc[test_id]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            dataset_blend_train[test_id] = y_pred
            dataset_blend_test_all.append(model.predict(test_x))
            i=i+1
            scores_of_base_model.append(evaluation(y_test,y_pred,metric=metric_for_regressor))
        print 'All folds Done!'
        score=np.mean(scores_of_base_model)
        
        # each folds do a prediction of a single base model, store average as the prediction of the test_x  
    
    dataset_blend_test=sum(dataset_blend_test_all)/(1.0*len(dataset_blend_test_all))
        
    return dataset_blend_train,dataset_blend_test,score

'''
def ensemble(layer_name,models,train_x,train_y,test_x,n_classes,n_folds,metric_for_classifier='log_loss',metric_for_regressor='mean_squared_error'):
    new_blend_train_classifier=[]
    new_blend_test_classifier=[]
    new_blend_train_regressor=[]
    new_blend_test_regressor=[]
    for i,model in enumerate(models):
        print i,model
        new_train,new_test,score=stackKfold(model,train_x,train_y,test_x,n_classes,n_folds,
                                            metric_for_classifier,metric_for_regressor)
        
        if 'Classi' in str(model): 
            print 'the',metric_for_classifier,'of',layer_name,'model',i,'is',score
            new_blend_train_classifier.append(new_train)
            new_blend_test_classifier.append(new_test)
        
        if 'Regres' in str(model): 
            print 'the',metric_for_regressor,'of',layer_name,'model',i,'is',score
            new_blend_train_regressor.append(new_train)
            new_blend_test_regressor.append(new_test)
            
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
    
    return  new_blend_train,new_blend_test



def splitPred(data,n_classes):
    outcome=[]
    for i in range(data.shape[1]/n_classes):
        outcome.append(data.iloc[:,0:n_classes].values)
        data=data.iloc[:,n_classes:]
    return outcome
        

  
def opt_weight(train_output,train_y,n_classes,metric='log_loss',method='SLSQP',maximize=False):
    
    def weighted_eval_minimize(weights):
        #scipy minimize will pass the weights as a numpy array 
        entry=splitPred(train_output,n_classes)
        final_prediction= sum([weights[i]*entry[i] for i in range(len(weights))])
        return evaluation(train_y, final_prediction,metric)

    def weighted_eval_maximize(weights):
        #scipy minimize will pass the weights as a numpy array
        entry=splitPred(train_output,n_classes)
        final_prediction= sum([weights[i]*entry[i] for i in range(len(weights))])
        return evaluation(train_y, final_prediction,metric)*-1   
    
    
    starting_values = np.array([0.2]*(train_output.shape[1]/n_classes))
    cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
    bounds = [(0,1)]*(train_output.shape[1]/n_classes)
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


def predict(train_x,train_y,test_x,models,weights):
    pred_of_all=[]
    for model in models:
        model.fit(train_x,train_y)
        pred_of_all.append(model.predict(test_x))
    pred=sum([weights[i]*pred_of_all[i] for i in range(len(weights))])
    return pred
    
    
def predict_proba(train_x,train_y,test_x,models,weights):
    pred_of_all=[]
    for model in models:
        model.fit(train_x,train_y)
        pred_of_all.append(model.predict_proba(test_x))
    pred=sum([weights[i]*pred_of_all[i] for i in range(len(weights))])
    return pred

'''    
        


















