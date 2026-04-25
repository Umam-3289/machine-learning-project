import os
import sys
import pickle
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_name=os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(x_train,y_train,x_test,y_test,models):
    try:
        report={}
    
        for i in range(len(list(models))):
            model=list(models.values())[i]
            # para=param.get(model,{})

            # gs=GridSearchCV(model,param,cv=3)
            # gs.fit(x_train,y_train)

            # model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)


            model.fit(x_train,y_train)

            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)

            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]]=test_model_score

        return report
    except Exception as e:
        raise CustomException(e,sys)

