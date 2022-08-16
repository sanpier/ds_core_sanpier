import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import time
from azureml.core import Run
from catboost import CatBoostClassifier
from concurrent import futures 
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, recall_score, precision_score, roc_auc_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate, cross_val_predict, KFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class Classifier:

    dict_classifiers = {
        "LR": LogisticRegression(random_state=42),
        "Ridge": RidgeClassifier(random_state=42),
        "KNC": KNeighborsClassifier(),
        "SVC": SVC(probability=True, random_state=42), # kernel == "sigmoid" | 'linear' | 'poly'
        "GBC": GradientBoostingClassifier(random_state=42),
        "LGBMC": LGBMClassifier(random_state=42),
        "DTC": DecisionTreeClassifier(random_state=42),
        "RFC": RandomForestClassifier(random_state=42),
        "GNB": GaussianNB(),
        "Bagging": BaggingClassifier(random_state=42), # base_estimator=KNeighborsClassifier()
        "Extra": ExtraTreesClassifier(max_depth=None, random_state=42),
        "CatBoost": CatBoostClassifier(silent=True, random_state=42)
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

    ### TRAIN / TEST SPLIT ###
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

    def generate_train_test(self):
        """ create train test sets for modeling
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42, shuffle=True)
        if not self.not_categorical_target:
            self.encoded_y_train = self.le.transform(self.y_train)
            self.encoded_y_test = self.le.transform(self.y_test)
        print("Train data size:", self.X_train.shape, "\nTrain target distribution:\n", self.y_train.value_counts())
        print("Test data size:", self.X_test.shape, "\nTest target distribution:\n", self.y_test.value_counts())

    def generate_train_test_by_given_gmlids(self, gmlids):
        """ create train test sets for modeling
        """
        train_indexes = self.data[~self.data.gmlid.isin(gmlids)].index
        test_indexes = self.data[self.data.gmlid.isin(gmlids)].index
        self.X_train = self.X.iloc[train_indexes]
        self.X_test = self.X.iloc[test_indexes]
        self.y_train = self.y.iloc[train_indexes]
        self.y_test = self.y.iloc[test_indexes]
        if not self.not_categorical_target:
            self.encoded_y_train = self.le.transform(self.y_train)
            self.encoded_y_test = self.le.transform(self.y_test)
        print("Train data size:", self.X_train.shape, "\nTrain target distribution:\n", self.y_train.value_counts())
        print("Test data size:", self.X_test.shape, "\nTest target distribution:\n", self.y_test.value_counts())

    ### OVERSAMPLING / DATA AUGMENTATION ###
    def oversampling(self, k):
        """ oversampling method for imbalanced data
        """
        if hasattr(self, 'X_train'): 
            over = SMOTE(k_neighbors=k, random_state=42)
            self.X_train, self.y_train = over.fit_resample(self.X_train, self.y_train)
            self.X = self.X_train.copy()
            self.y = self.y_train.copy()
            if not self.not_categorical_target:
                self.encoded_y_train = self.le.transform(self.y_train)
                self.encoded_y = self.le.transform(self.y)
            print("After oversampling:\n",
                  "\tTrain data size:", self.X_train.shape, "\n",
                  "\tTrain target distribution:\n", self.y_train.value_counts(), "\n")          
        else:
            raise AssertionError("Please first generate train & test datasets out of given data!")

    def randomized_smoothing_extend(self, noise, extend, rounding_cols):
        """ augment training data with synthetic data generated by
            realistic gaussian noise to make modeling more robust
        """
        if hasattr(self, 'X_train'):  
            # split x and y
            y_train = pd.Series(self.y_train.copy(), name=self.target).reset_index(drop=True)
            if not self.not_categorical_target:
                y_train = pd.Series(self.encoded_y_train.copy(), name=self.target).reset_index(drop=True)
            X_train = self.X_train.copy().reset_index(drop=True)
            df_train = pd.concat([X_train, y_train], axis=1)
            
            # syntetic data of X
            df_trainX_syntetic = pd.concat([X_train]*extend).reset_index(drop=True)
            df_trainy_syntetic = pd.Series(pd.concat([y_train]*extend), name=self.target).reset_index(drop=True)
            df_trainX_syntetic = np.random.normal(df_trainX_syntetic, X_train.std()*noise, size = df_trainX_syntetic.shape)
            df_trainX_syntetic = pd.DataFrame(data=df_trainX_syntetic, columns=X_train.columns)

            # postprocessing
            non_zero_cols = [i for i, value in (X_train < 0).any().items() if value==False]
            for i in non_zero_cols:
                df_trainX_syntetic.loc[df_trainX_syntetic[i] < 0, i] = -1*df_trainX_syntetic.loc[df_trainX_syntetic[i] < 0, i]                
            binary_cols = X_train.columns[X_train.isin([0,1]).all()].tolist()
            for i in binary_cols:
                df_trainX_syntetic[i] = df_trainX_syntetic[i].round()
                df_trainX_syntetic.loc[df_trainX_syntetic[i] > 1, i] = 1
                df_trainX_syntetic.loc[df_trainX_syntetic[i] < 0, i] = 0
            for i in rounding_cols:
                df_trainX_syntetic[i] = df_trainX_syntetic[i].round()

            # concat syntetic data with y + add original data into it
            df_train_syntetic = pd.concat([df_trainX_syntetic, df_trainy_syntetic], axis=1)
            df_train = pd.concat([df_train, df_train_syntetic]).reset_index(drop=True)
            self.X_train = df_train[self.features].copy()
            self.y_train = pd.Series(df_train[self.target].copy(), name=self.target)
            self.X = self.X_train.copy()
            self.y = pd.Series(self.y_train.copy(), name=self.target)
            
            # set y if the target is categorical
            if not self.not_categorical_target:
                self.encoded_y_train = pd.Series(self.y_train.copy(), name=self.target)
                self.encoded_y = pd.Series(self.y.copy(), name=self.target)
                self.y_train = pd.Series(self.le.inverse_transform(self.encoded_y_train), name=self.target)
                self.y = pd.Series(self.le.inverse_transform(self.encoded_y), name=self.target)
            print("After randomized smoothing:\n",
                  "\tTrain data size:", self.X_train.shape, "\n",
                  "\tTrain target distribution:\n", self.y_train.value_counts(), "\n")   
        else:
            raise AssertionError("Please first generate train & test datasets out of given data!")
    
    def randomized_smoothing_equalize(self, noise, rounding_cols):
        """ augment training data with synthetic data generated by
            realistic gaussian noise to make modeling more robust, 
            further handle the imbalanced distribution of target
        """
        if hasattr(self, 'X_train'):
            # split x and y
            y_train = pd.Series(self.y_train.copy(), name=self.target).reset_index(drop=True)
            if not self.not_categorical_target:
                y_train = pd.Series(self.encoded_y_train.copy(), name=self.target).reset_index(drop=True)
            X_train = self.X_train.copy().reset_index(drop=True)
            df_train = pd.concat([X_train, y_train], axis=1)
            
            # set columns
            non_zero_cols = [i for i, value in (X_train < 0).any().items() if value==False] 
            binary_cols = X_train.columns[X_train.isin([0,1]).all()].tolist()

            # syntetic data
            max_ = y_train.value_counts().max()
            frames = [df_train]
            for i, value in y_train.value_counts().items():
                expand = round(max_/value)
                df_train_target_i = df_train[df_train[self.target] == i]
                print(f"{i} portion of data (shape={df_train_target_i.shape}) needs to be expanded by:", expand)                
                if expand > 1:
                    df_syntetic_i = pd.concat([df_train_target_i]*(expand-1))
                    df_syntetic_i = np.random.normal(df_syntetic_i, df_train_target_i.std()*noise, size = df_syntetic_i.shape)
                    df_syntetic_i = pd.DataFrame(data=df_syntetic_i, columns=df_train_target_i.columns)
                    df_syntetic_i[self.target] = i
                    frames.append(df_syntetic_i)

            # concat syntetic data into original data
            df_train = pd.concat(frames).reset_index(drop=True)

            # postprocessing
            for i in non_zero_cols:
                df_train.loc[df_train[i] < 0, i] = -1*df_train.loc[df_train[i] < 0, i]
            for i in binary_cols:
                df_train[i] = df_train[i].round()
                df_train.loc[df_train[i] > 1, i] = 1
                df_train.loc[df_train[i] < 0, i] = 0
            for i in rounding_cols:
                df_train[i] = df_train[i].round()

            # split data into X and y
            self.X_train = df_train[self.features].copy()
            self.y_train = pd.Series(df_train[self.target].copy(), name=self.target)
            self.X = self.X_train.copy()
            self.y = pd.Series(self.y_train.copy(), name=self.target)
            
            if not self.not_categorical_target:
                self.encoded_y_train = pd.Series(self.y_train.copy(), name=self.target)
                self.encoded_y = pd.Series(self.y.copy(), name=self.target)
                self.y_train = pd.Series(self.le.inverse_transform(self.encoded_y_train), name=self.target)
                self.y = pd.Series(self.le.inverse_transform(self.encoded_y), name=self.target) 
            print("After randomized smoothing:\n",
                  "\tTrain data size:", self.X_train.shape, "\n",
                  "\tTrain target distribution:\n", self.y_train.value_counts(), "\n")   
        else:
            raise AssertionError("Please first generate train & test datasets out of given data!")

    ### SCORE MODELS ###
    def experiment_models(self, cv=5, score="weighted", in_test=False):
        """ experiment the bunch of classification models for the 
            classification problem: return the dataframe of scores
        """
        # check model performances with parallel computing
        cores = multiprocessing.cpu_count()
        workers = round(cores/2)
        df_scores = pd.DataFrame(columns = ['model', 'accuracy', 'recall', 'precision', 'f1_score', 'roc_auc'])
        with futures.ProcessPoolExecutor(max_workers=workers) as executor:
            jobs = {}
            for model, model_instantiation in self.dict_classifiers.items():
                try:
                    print(f'{model} is training:')
                    if in_test == True:
                        job = executor.submit(self.score_in_test, model = model_instantiation, score=score)
                    else:
                        job = executor.submit(self.cv_score_model, model = model_instantiation, cv=cv, score=score)
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
                               'recall': scores["recall"], 'precision': scores["precision"], 
                               'f1_score': scores["f1_score"], 'roc_auc': scores['roc_auc']}
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

    def cv_score_model(self, model=None, cv=5, score="weighted", confusion=False):
        """ do a cross validation scoring with given model if no 
            model is given then a logistic regression will be tried
        """
        if model is None:
            if hasattr(self, 'model'): 
                model = self.model
            else:
                raise AssertionError("Please pass over a model to proceed!")
        elif model == "lr":
            model = LogisticRegression(random_state=42)
        elif model == "best":
            model = self.best_model
        elif model == "stack":
            model = self.stacking_model()
        elif model == "vote":
            model = self.voting_model()
        y = self.y.copy()
        if (not self.not_categorical_target) & (not (str(type(model)) == "<class 'catboost.core.CatBoostClassifier'>")):
            y = self.encoded_y.copy()
        self.pred_test = cross_val_predict(model, self.X, y, cv=cv) 
        if (not self.not_categorical_target) & (not (str(type(model)) == "<class 'catboost.core.CatBoostClassifier'>")):
            self.pred_test = self.le.inverse_transform(self.pred_test)  
        self.model = model
        scores = self.classification_metrics(self.y, self.pred_test, score=score)  
        if confusion:
            self.apply_confusion_matrix(self.y, self.pred_test)    
        return scores

    def score_in_test(self, model=None, score="weighted", confusion=False):
        """ score the given classification model on the generated test data
        """
        if hasattr(self, 'X_train'): 
            if model is None:
                if hasattr(self, 'model'): 
                    model = self.model
                else:
                    raise AssertionError("Please pass over a model to proceed!")
            elif model == "lr":
                model = LogisticRegression(random_state=42)
            elif model == "best":
                model = self.best_model
            elif model == "stack":
                model = self.stacking_model()
            elif model == "vote":
                model = self.voting_model()
            y_train = self.y_train 
            if (not self.not_categorical_target) & (not (str(type(model)) == "<class 'catboost.core.CatBoostClassifier'>")):
                y_train = self.encoded_y_train
            model.fit(self.X_train, y_train)
            self.pred_test = model.predict(self.X_test)
            if (not self.not_categorical_target) & (not (str(type(model)) == "<class 'catboost.core.CatBoostClassifier'>")):
                self.pred_test = self.le.inverse_transform(self.pred_test)  
            self.model = model
            scores = self.classification_metrics(self.y_test, self.pred_test, score=score) 
            if confusion:
                self.apply_confusion_matrix(self.y_test, self.pred_test)    
            return scores
        else:
            raise AssertionError("Please first generate train & test datasets out of given data!")

    def classification_metrics(self, y_test, preds, score="weighted"):
        """ classification perforamce metrics 
        """
        scores = {}
        scores["accuracy"] = round(accuracy_score(y_test, preds, normalize=True), 3)
        scores["recall"] = round(recall_score(y_test, preds, average=score), 3)
        scores["precision"] = round(precision_score(y_test, preds, average=score), 3)
        scores["f1_score"] = round(f1_score(y_test, preds, average=score), 3)
        scores["roc_auc"] = round(roc_auc_score(y_test, preds, average=score, multi_class='ovr'), 3)   
        scores["test_sample"] = len(y_test)    
        return scores
        
    def probability_prediction(self, model=None, cv=5):
        """ do a cross validation predictions with probabilities
        """
        if model is None:
            if hasattr(self, 'model'): 
                model = self.model
            else:
                raise AssertionError("Please pass over a model to proceed!")
        elif model == "lr":
            model = LogisticRegression(random_state=42)
        elif model == "best":
            model = self.best_model
        elif model == "stack":
            model = self.stacking_model()
        elif model == "vote":
            model = self.voting_model()
        y = self.y.copy()
        if not self.not_categorical_target:
            y = self.encoded_y.copy()
        return cross_val_predict(model, self.X, y, cv=cv, method='predict_proba')

    def apply_confusion_matrix(self, y, preds, name="", rotate=False):
        """ confusion matrix of the classification 
        """        
        cf_matrix = confusion_matrix(y, preds, labels=self.labels)
        cf_matrix_percentage = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis] 
        fig_unit_size = len(self.labels)           
        sns.set(rc={'figure.figsize':(fig_unit_size*5, fig_unit_size*2)})
        fig, axs = plt.subplots(ncols=2)
        fig.suptitle(f"Confusion Matrix", fontsize=20)
        g1 = sns.heatmap(cf_matrix, annot=True, fmt='g', cmap='Blues', ax=axs[0])
        g2 = sns.heatmap(cf_matrix_percentage, annot=True, fmt='.1%', cmap='Blues', ax=axs[1])
        g1.set_xlabel('Predicted Values')
        g1.set_ylabel('Actual Values ')
        g1.xaxis.set_ticklabels(self.labels)
        g1.yaxis.set_ticklabels(self.labels)
        if rotate:
            g1.set_xticklabels(g1.get_xticklabels(), rotation=30)
            g1.set_yticklabels(g1.get_xticklabels(), rotation=0)
        g2.set_xlabel('Predicted Values')
        g2.set_ylabel('Actual Values ')
        g2.xaxis.set_ticklabels(self.labels)
        g2.yaxis.set_ticklabels(self.labels)
        if rotate:
            g2.set_xticklabels(g2.get_xticklabels(), rotation=30)
            g2.set_yticklabels(g2.get_xticklabels(), rotation=0)
        plt.show()
        # save figure if on cloud
        if self.online_run:
            filename=f'./outputs/confusion_matrix_{name}.png'
            plt.savefig(filename, dpi=600)
            plt.close()

    ### ENSEMBL MODELS ### 
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
            meta_model = LogisticRegression(random_state=42)
            final_model = StackingClassifier(estimators=self.base_models, final_estimator=meta_model, cv=5)
            return final_model
        else:
            raise AssertionError("Please first experiment models to set top 3 base models!")

    def voting_model(self):
        """ create a voting model using best 3 base models
            that are defined after experimenting all models
        """
        if hasattr(self, 'base_models'): 
            final_model = VotingClassifier(estimators=self.base_models, voting='soft')
            return final_model
        else:
            raise AssertionError("Please first experiment models to set top 3 base models!")

    ### TRAIN GIVEN MODEL & PREDICT GIVEN TEST ###
    def train_model(self, model=None):
        """ train the given classification model on whole data
        """   
        if model is None:
            if hasattr(self, 'model'): 
                model = self.model
            else:
                raise AssertionError("Please pass over a model to proceed!")
        elif model == "lr":
            model = LogisticRegression(random_state=42)
        elif model == "best":
            model = self.best_model
        elif model == "stack":
            model = self.stacking_model()
        elif model == "vote":
            model = self.voting_model()
        y = self.y.copy()
        if (not self.not_categorical_target) & (not (str(type(model)) == "<class 'catboost.core.CatBoostClassifier'>")):
            y = self.encoded_y.copy()
        model.fit(self.X, y)
        print("Model is fit on whole data!")
        self.trained_model = model

    def predict_test(self, test):
        """ train the given classification model on whole data
        """   
        if hasattr(self, 'trained_model'): 
            if (not self.not_categorical_target) & (not (str(type(self.trained_model)) == "<class 'catboost.core.CatBoostClassifier'>")):
                preds = self.trained_model.predict(test[self.features])
                preds = self.le.inverse_transform(preds)
            else:
                preds = self.trained_model.predict(test[self.features]).astype(int)
            return preds
        else:
            raise AssertionError("Please first train a model then predict on the test data!")

    ### EXPLAIN MODEL ###
    def explain_model_with_shap(self, model=None):
        """ show SHAP values to explain the output of the regression model
        """
        if model is None:
            if hasattr(self, 'model'): 
                model = self.model
            else:
                raise AssertionError("Please pass over a model to proceed!")
        # fit the model
        self.train_model(model)
        # set important features
        importances = model.feature_importances_
        sorted_idx = importances.argsort()[(-1*self.X.shape[1]):]
        important_features = self.X.columns[sorted_idx].tolist() 
        # get shap summary plot
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(self.X)
        shap.summary_plot(shap_values, self.X, feature_names = important_features)
