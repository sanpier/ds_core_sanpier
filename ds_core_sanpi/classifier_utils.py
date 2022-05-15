import lightgbm as lgbm
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
import seaborn as sns
import time
from azureml.core import Run
from catboost import CatBoostClassifier
from concurrent import futures 
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, recall_score, precision_score, roc_auc_score, f1_score
from sklearn.model_selection import cross_validate, cross_val_predict, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class Classifier:

    dict_classifiers = {
        #"LR": LogisticRegression(solver='liblinear'),
        "KNC": KNeighborsClassifier(),
        "SVC_linear": SVC(probability=True, kernel='linear'),
        "SVC_poly": SVC(probability=True, kernel='poly'),
        "SVC_default": SVC(probability=True),
        "GBC": GradientBoostingClassifier(),
        "LGBMC": lgbm.LGBMClassifier(),
        "DTC": DecisionTreeClassifier(),
        "RFC": RandomForestClassifier(),
        "GNB": GaussianNB(),
        "Bagging": BaggingClassifier(KNeighborsClassifier()),
        "Extra": ExtraTreesClassifier(),
        "CatBoost": CatBoostClassifier(silent=True)
    }

    def __init__(self, data, keep_cols, target, online=False, verbose=True):
        """ construction of Classifier class 
        """
        self.data = data
        self.target = target
        self.keep_cols = keep_cols
        self.split_x_and_y(verbose) 
        self.not_categorical_target = all(data[data[target].notnull()][target].apply(lambda x: str(x).isnumeric() | isinstance(x, (int, float))))
        if self.not_categorical_target:
            self.labels = [(int(i)) for i in sorted(data[target].unique().tolist())]
        else:
            self.le = LabelEncoder()
            self.encoded_y = self.le.fit_transform(self.y)
            if verbose:
                print("Categorical target is label encoded:", pd.Series(self.encoded_y).unique())
            self.labels = self.le.classes_
        if verbose:
            print("Classifier initialized with labels:", self.labels)
        self.online_run = online
        if self.online_run:
            self.run = Run.get_context()

    def split_x_and_y(self, verbose=True):
        """ split target from the data 
        """
        self.features = list(set(self.data.columns.tolist()) - set(self.keep_cols) - set([self.target]))
        if verbose:
            print("Training will be done using the following features:\n", self.features)
        self.X = self.data[self.features].copy()
        self.y = self.data[self.target].copy()
        if verbose:
            print("Data is split into X and y:\n",
                "\tX:", self.X.shape, "\n",
                "\ty:", self.y.shape)

    def experiment_models(self, cv=5):
        """ experiment the bunch of classification models for the 
            classification problem: return the dataframe of scores
        """
        # check model performances with parallel computing
        cores = multiprocessing.cpu_count()
        workers = round(cores/2)
        df_scores = pd.DataFrame(columns = ['model', 'accuracy', 'recall', 'precision', 'f1_score'])
        with futures.ProcessPoolExecutor(max_workers=workers) as executor:
            jobs = {}
            for model, model_instantiation in self.dict_classifiers.items():
                try:
                    print(f'{model} is training:')
                    job = executor.submit(self.cv_score_model, model = model_instantiation, cv=cv)
                except Exception as e:
                     print(model, "raises an exception while scoring:", e)
                     continue
                jobs[job] = model
                time.sleep(1)  # this is just to make the output look nicer
            for job in futures.as_completed(jobs):
                model = jobs[job]
                scores = job.result()
                print(f"{model} scores:")
                [print('\t', key,':', val) for key, val in scores.items()]
                score_entry = {'model': model, 'accuracy': scores["accuracy"], 
                               'recall': scores["recall"], 'precision': scores["precision"], 'f1_score': scores["f1_score"]}
                df_scores = df_scores.append(score_entry, ignore_index = True)
        # sort score dataframe by f1-values
        df_scores.sort_values('f1_score', ascending=False, inplace=True)
        # best models => base models for stacking
        self.base_models = [(df_scores.iloc[0].model, self.dict_classifiers[df_scores.iloc[0].model]),
                            (df_scores.iloc[1].model, self.dict_classifiers[df_scores.iloc[1].model]),
                            (df_scores.iloc[2].model, self.dict_classifiers[df_scores.iloc[2].model])]
        # set best model
        self.best_model = self.base_models[0]
        return df_scores

    def cv_score_model(self, model, cv=5, confusion=False):
        """ do a cross validation scoring with given model if no 
            model is given then a logistic regression will be tried
        """
        if model == "lr":
            model = LogisticRegression()
        elif model == "best":
            model = self.best_model
        elif model == "stack":
            model = self.stacking_model()
        y = self.y.copy()
        if not self.not_categorical_target:
            y = self.encoded_y.copy()
        preds = cross_val_predict(model, self.X, y, cv=cv)  
        scores = {}
        scores["accuracy"] = round(accuracy_score(y, preds, normalize=True), 3)  
        scores["recall"] = round(recall_score(y, preds, average='weighted'), 3) 
        scores["precision"] = round(precision_score(y, preds, average='weighted'), 3)    
        scores["f1_score"] = round(f1_score(y, preds, average='weighted'), 3)   
        if confusion:
            self.apply_confusion_matrix(preds)    
        return scores

    def apply_confusion_matrix(self, preds, name=""):
        """ confusion matrix of the classification 
        """        
        y = self.y.copy()
        if not self.not_categorical_target:
            y = self.encoded_y.copy()
        cf_matrix = confusion_matrix(y, preds, labels=self.labels)
        cf_matrix_percentage = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis] 
        fig_unit_size = len(self.labels)           
        sns.set(rc={'figure.figsize':(fig_unit_size*6, fig_unit_size*3)})
        fig, axs = plt.subplots(ncols=2)
        fig.suptitle(f"Confusion Matrix", fontsize=20)
        g1 = sns.heatmap(cf_matrix, annot=True, fmt='g', cmap='Blues', ax=axs[0])
        g2 = sns.heatmap(cf_matrix_percentage, annot=True, fmt='.1%', cmap='Blues', ax=axs[1])
        g1.set_xlabel('Predicted Values')
        g1.set_ylabel('Actual Values ')
        g1.xaxis.set_ticklabels(self.labels)
        g1.yaxis.set_ticklabels(self.labels)
        #g1.set_xticklabels(g1.get_xticklabels(), rotation=30)
        #g1.set_yticklabels(g1.get_xticklabels(), rotation=0)
        g2.set_xlabel('Predicted Values')
        g2.set_ylabel('Actual Values ')
        g2.xaxis.set_ticklabels(self.labels)
        g2.yaxis.set_ticklabels(self.labels)
        #g2.set_xticklabels(g2.get_xticklabels(), rotation=30)
        #g2.set_yticklabels(g2.get_xticklabels(), rotation=0)
        plt.show()
        # save figure if on cloud
        if self.online_run:
            filename=f'./outputs/confusion_matrix_{name}.png'
            plt.savefig(filename, dpi=600)
            plt.close()

    def define_base_models(self, base_list):
        """ give a list of model abbreviations to be used
            in stacking
        """
        self.base_models = []
        for name in base_list:
            self.base_models.append((name, self.dict_classifiers[name]))
        print("Base models are defined!")

    def stacking_model(self):
        """ create a stacking model using best 3 base models
            that are defined after experimenting all models
        """
        if hasattr(self, 'base_models'): 
            meta_model = LogisticRegression()
            final_model = StackingClassifier(estimators=self.base_models, final_estimator=meta_model, cv=5)
            return final_model
        else:
            raise AssertionError("Please first experiment models to set top 3 base models!")

    def train_model(self, model):
        """ train the given classification model on whole data
        """   
        if model == "lr":
            model = LogisticRegression()
        elif model == "best":
            model = self.best_model
        elif model == "stack":
            model = self.stacking_model()
        y = self.y.copy()
        if not self.not_categorical_target:
            y = self.encoded_y.copy()
        model.fit(self.X, y)
        print("Model is fit on whole data!")
        self.trained_model = model

    def predict_test(self, test):
        """ train the given classification model on whole data
        """   
        if hasattr(self, 'trained_model'): 
            if not self.not_categorical_target:
                preds = self.trained_model.predict(test[self.features])
                preds = self.le.inverse_transform(preds)
            else:
                preds = self.trained_model.predict(test[self.features]).astype(int)
            return preds
        else:
            raise AssertionError("Please first train a model then predict on the test data!")