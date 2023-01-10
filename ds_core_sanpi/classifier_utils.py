import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import string
import time
from azureml.core import Run
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from pathos.helpers import cpu_count
from pathos.pools import ProcessPool
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class Classifier:

    dict_classifiers = {
        "LogR": LogisticRegression(random_state=42),
        "RidgeC": RidgeClassifier(random_state=42),
        "GNB": GaussianNB(),
        "AdaC": AdaBoostClassifier(random_state=42),
        "GBC": GradientBoostingClassifier(random_state=42),
        "XGBC": XGBClassifier(random_state=42),
        "LGBMC": LGBMClassifier(random_state=42),
        "BaggingC": BaggingClassifier(random_state=42), # base_estimator=KNeighborsClassifier()
        "SVC": SVC(probability=True, random_state=42), # kernel == "sigmoid" | 'linear' | 'poly'
        "KNC": KNeighborsClassifier(),
        "DTC": DecisionTreeClassifier(random_state=42),
        "RFC": RandomForestClassifier(random_state=42),
        "ExtraC": ExtraTreesClassifier(max_depth=None, random_state=42),
        "CatBC": CatBoostClassifier(silent=True, random_state=42)
    }
    
    def __init__(self, data, keep_cols, target, online=False):
        """ construction of Classifier class 
        """
        self.data = data[keep_cols + sorted(list(set(data.columns.tolist()) - set(keep_cols + [target]))) + [target]]
        self.target = target
        self.keep_cols = keep_cols
        self.split_x_and_y() 
        self.not_categorical_target = all(data[data[target].notnull()][target].apply(lambda x: str(x).isnumeric() | isinstance(x, (int, float))))
        if self.not_categorical_target:
            self.labels = [(int(i)) for i in sorted(data[target].unique().tolist())]
        else:
            self.le = LabelEncoder()
            self.encoded_y = self.le.fit_transform(self.y)
            print("Categorical target is label encoded:", pd.Series(self.encoded_y).unique())
            self.labels = self.le.classes_
        print("Classifier initialized with labels:", self.labels)
        self.online_run = online
        if self.online_run:
            self.run = Run.get_context()

    ### TRAIN / TEST SPLIT ###
    def split_x_and_y(self):
        """ split target from the data 
        """
        self.features = sorted(list(set(self.data.columns.tolist()) - set(self.keep_cols + [self.target])))
        if len(self.features) <= 25:
            print("Training will be done using the following features:\n", self.features)
        self.X = self.data[self.features].copy()
        self.y = self.data[self.target].copy()
        print("Data is split into X and y:\n",
              "\tX:", self.X.shape, "\n",
              "\ty:", self.y.shape)

    def generate_train_test(self, test_size=0.25, verbose=False):
        """ create train test sets for modeling
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42, stratify=self.y)
        if not self.not_categorical_target:
            self.encoded_y_train = self.le.transform(self.y_train)
            self.encoded_y_test = self.le.transform(self.y_test)
        print("Train data size:", self.X_train.shape)
        if verbose:
            print("Train target distribution:\n", self.y_train.value_counts())
        print("Test data size:", self.X_test.shape) 
        if verbose:
            print("Test target distribution:\n", self.y_test.value_counts())

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
    def experiment_models(self, by_test=True, cv=5, score="weighted"):
        """ check model performances with parallel computing
        """
        if by_test:            
            if not hasattr(self, 'X_train'): 
                self.generate_train_test() 
        cores = cpu_count()
        pool = ProcessPool(cores)
        model_names = list(self.dict_classifiers.keys())
        models = list(self.dict_classifiers.values())
        # experiment bunch of regression models
        print(f"Running models parallel with {cores} cores:", model_names)
        n_models = len(models)
        try:
            if by_test == True:
                scores_data = pool.amap(self.score_in_test, models, model_names, [score] * n_models)
            else:
                scores_data = pool.amap(self.cv_score_model, models, model_names, [cv] * n_models, [score] * n_models)
            while not scores_data.ready():
                time.sleep(5); print(".", end=' ')
            scores_data = scores_data.get()
            print("\n")
        except Exception as e:
            print(f"\nCouldn't run parallel because of the following exception:", e)
            scores_data = []
            for m_name, model in self.dict_classifiers.items():
                if by_test == True:
                    scores_data.append(self.score_in_test(model, m_name, score))
                else:
                    scores_data.append(self.cv_score_model(model, m_name, cv, score))      
        df_scores = pd.DataFrame(scores_data)                 
        # sort score dataframe by f1-values
        df_scores.sort_values('f1_score', ascending=False, inplace=True)
        # best models => base models for stacking
        self.base_models = [(df_scores.iloc[0].model, self.dict_classifiers[df_scores.iloc[0].model]),
                            (df_scores.iloc[1].model, self.dict_classifiers[df_scores.iloc[1].model]),
                            (df_scores.iloc[2].model, self.dict_classifiers[df_scores.iloc[2].model])]
        # set best model
        self.best_model = self.base_models[0]
        return df_scores

    def cv_score_model(self, model=None, model_name="", cv=5, score="weighted", confusion=False):
        """ do a cross validation scoring with given model if no 
            model is given then a logistic regression will be tried
        """
        if model is None:
            if hasattr(self, 'model'): 
                model = self.model
            else:
                raise AssertionError("Please pass over a model to proceed!")
        elif model == "logr":
            model = LogisticRegression(random_state=42)
        elif model == "best":
            model = self.best_model
        elif model == "stack":
            model = self.stacking_model()
        elif model == "vote":
            model = self.voting_model()
        if model_name == "":
            model_name = extract_model_name(model)
        y = self.y.copy()
        if (not self.not_categorical_target) & (not model_name in ["CatBoostClassifier", "CatBC"]):
            y = self.encoded_y.copy()
        self.pred_test = cross_val_predict(model, self.X, y, cv=cv) 
        if (not self.not_categorical_target) & (not model_name in ["CatBoostClassifier", "CatBC"]):
            self.pred_test = self.le.inverse_transform(self.pred_test)  
        self.model = model
        scores = classification_metrics(self.y, self.pred_test, score, model_name)  
        if confusion:
            apply_confusion_matrix(self.y, self.pred_test, self.labels, self.online_run, f"cv_{model_name}")     
        return scores

    def score_in_test(self, model=None, model_name="", score="weighted", confusion=False):
        """ score the given classification model on the generated test data
        """
        if hasattr(self, 'X_train'): 
            if model is None:
                if hasattr(self, 'model'): 
                    model = self.model
                else:
                    raise AssertionError("Please pass over a model to proceed!")
            elif model == "logr":
                model = LogisticRegression(random_state=42)
            elif model == "best":
                model = self.best_model
            elif model == "stack":
                model = self.stacking_model()
            elif model == "vote":
                model = self.voting_model()
            if model_name == "":
                model_name = extract_model_name(model)
            y_train = self.y_train 
            if (not self.not_categorical_target) & (not model_name in ["CatBoostClassifier", "CatBC"]):
                y_train = self.encoded_y_train
            model.fit(self.X_train, y_train)
            self.pred_test = model.predict(self.X_test)
            if (not self.not_categorical_target) & (not model_name in ["CatBoostClassifier", "CatBC"]):
                self.pred_test = self.le.inverse_transform(self.pred_test)  
            self.model = model
            scores = classification_metrics(self.y_test, self.pred_test, score, model_name) 
            if confusion:
                apply_confusion_matrix(self.y_test, self.pred_test, self.labels, self.online_run, f"test_{model_name}")    
            return scores
        else:
            raise AssertionError("Please first generate train & test datasets out of given data!")
        
    def cv_probability_prediction(self, model=None, cv=5):
        """ do a cross validation predictions with probabilities
        """
        if model is None:
            if hasattr(self, 'model'): 
                model = self.model
            else:
                raise AssertionError("Please pass over a model to proceed!")
        elif model == "logr":
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
        prob_pred = cross_val_predict(model, self.X, y, cv=cv, method='predict_proba')
        return pd.Series(map(lambda i: tuple(i), prob_pred))

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

    def cv_ensemble_models(self, models, score="weighted", confusion=False):
        """ create a ensemble of different classifiers manually given 
            a list of models
        """
        self.ensemble_models = models
        df_preds = pd.DataFrame(self.y)
        binary = self.data[self.target].nunique() == 2
        # individual model predictions
        for model in models:
            model_name = ''.join([char for char in extract_model_name(model) if char.isupper()])
            print(model_name, "scores\t:", self.cv_score_model(model=model, model_name=model_name, score=score, confusion=confusion))
            df_preds[f"{model_name.lower()}_pred"] = self.pred_test
            if binary:
                df_preds[f"{model_name.lower()}_pred_proba"] = self.cv_probability_prediction(model=model)
        # prepare proba columns
        if binary:
            pred_proba_cols = [i for i in df_preds.columns if i.endswith("_pred_proba")]
            for col_i in pred_proba_cols:
                df_preds[col_i] = df_preds[col_i].apply(lambda x: x[1]) # Is 1 the target always?
        # meta model
        meta_model = LinearRegression()
        pred_cols = [i for i in df_preds.columns if i.endswith("_pred")]
        meta_model.fit(df_preds[pred_cols], df_preds[self.target])
        coefs = dict(zip(pred_cols, meta_model.coef_/meta_model.coef_.sum()))
        print("Model coefficients:")
        print(coefs)
        self.ensemble_coefficients = coefs
        # combined predicitons
        if binary:
            df_preds["final_pred_proba"] = sum([df_preds[f"{i}_proba"]*coefs[i] for i in coefs.keys()])
            df_preds["final_pred"] = df_preds["final_pred_proba"].apply(lambda x: 1 if x >= 0.5 else 0)
        else:
            df_preds["final_pred"] = sum([df_preds[i]*coefs[i] for i in coefs.keys()]).apply(round)
        print("Ensemble scores\t:", classification_metrics(df_preds[self.target], df_preds.final_pred, score=score, model_name="Ensemble"))
        if confusion:
            apply_confusion_matrix(df_preds[self.target], df_preds.final_pred, [0,1])
        return df_preds

    def predict_test_by_ensemble(self, df):
        """ do prediction by ensembling the given models
        """
        if hasattr(self, 'ensemble_models'): 
            # train models and do individual predictions
            binary = self.data[self.target].nunique() == 2
            df_preds_test = pd.DataFrame()
            for model in self.ensemble_models:
                model_name = ''.join([char for char in extract_model_name(model) if char.isupper()])
                print(model_name, "training...")
                self.train_model(model)
                df_ = self.predict_test(df)
                df_preds_test[f"{model_name.lower()}_pred"] = df_["prediction"]
                df_preds_test[f"{model_name.lower()}_pred_proba"] = df_["prediction_prob"]
            # final prediciton
            if binary:
                pred_proba_cols = [i for i in df_preds_test.columns if i.endswith("_pred_proba")]
                for col_i in pred_proba_cols:
                    df_preds_test[col_i] = df_preds_test[col_i].apply(lambda x: float(x[1])) # Is 1 the target always?
                df_preds_test["final_pred_proba"] = sum([df_preds_test[f"{i}_proba"]*self.ensemble_coefficients[i] for i in self.ensemble_coefficients.keys()])
                df_preds_test["final_pred"] = df_preds_test["final_pred_proba"].apply(lambda x: 1 if x >= 0.5 else 0)
            else:      
                df_preds_test["final_pred"] = sum([df_preds_test[i]*self.ensemble_coefficients[i] for i in self.ensemble_coefficients.keys()]).apply(round)
            return df_preds_test
        else:
            raise AssertionError("Please first call cv_ensemble_models method and pass your models!")


    ### TRAIN GIVEN MODEL & PREDICT GIVEN TEST ###
    def train_model(self, model=None):
        """ train the given classification model on whole data
        """   
        if model is None:
            if hasattr(self, 'model'): 
                model = self.model
            else:
                raise AssertionError("Please pass over a model to proceed!")
        elif model == "logr":
            model = LogisticRegression(random_state=42)
        elif model == "best":
            model = self.best_model
        elif model == "stack":
            model = self.stacking_model()
        elif model == "vote":
            model = self.voting_model()
        model_name = extract_model_name(model)
        y = self.y.copy()
        if (not self.not_categorical_target) & (not model_name in ["CatBoostClassifier", "CatBC"]):
            y = self.encoded_y.copy()
        model.fit(self.X, y)
        print("Model is fit on whole data!")
        self.trained_model = model

    def predict_test(self, df):
        """ predict the given data with the trained model
            and return the same data including predictions
        """   
        # align data
        try:
            df_X = df[self.features]
        except Exception as e:
            print("Given data doesn't have the same set of features as training!")
        if hasattr(self, 'trained_model'): 
            model_name = extract_model_name(self.trained_model)
            preds = self.trained_model.predict(df_X)
            if (not self.not_categorical_target) & (not model_name in ["CatBoostClassifier", "CatBC"]):
                preds = self.le.inverse_transform(preds)
            df["prediction"] = preds
            df["prediction_prob"] = pd.Series(map(lambda i: tuple(i), self.trained_model.predict_proba(df_X)))
            return df
        else:
            raise AssertionError("Please first train a model then predict on the test data!")

    ### EXPLAIN MODEL ###
    def explain_model_with_shap(self, model=None):
        """ show SHAP values to explain the output of the classification model
        """            
        if model is None:
            if hasattr(self, 'model'): 
                model = self.model
            else:
                raise AssertionError("Please pass over a model to proceed!")
        model.fit(self.X, self.y)
        explainer = shap.Explainer(model)
        shap_values = explainer(self.X)
        shap.plots.beeswarm(shap_values)


### AUXILIARY FUNCTIONS ###
def classification_metrics(y_test, preds, score="weighted", model_name=""):
    """ classification perforamce metrics 
    """
    return {
        'model' : model_name,
        'accuracy' : round(accuracy_score(y_test, preds, normalize=True), 3),
        'recall' : round(recall_score(y_test, preds, average=score), 3),
        'precision' : round(precision_score(y_test, preds, average=score), 3),
        'f1_score' : round(f1_score(y_test, preds, average=score), 3),
        'sample_size' : len(y_test)  
    }

def apply_confusion_matrix(y, preds, labels, online_run=False, name=""):
    """ confusion matrix of the classification 
    """        
    cf_matrix = confusion_matrix(y, preds, labels=labels)
    cf_matrix_percentage = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis] 
    fig_unit_size = len(labels)           
    sns.set(rc={'figure.figsize':(fig_unit_size*5, fig_unit_size*2)})
    fig, axs = plt.subplots(ncols=2)
    fig.suptitle(f"Confusion Matrix", fontsize=20)
    g1 = sns.heatmap(cf_matrix, annot=True, fmt='g', cmap='Blues', ax=axs[0])
    g2 = sns.heatmap(cf_matrix_percentage, annot=True, fmt='.1%', cmap='Blues', ax=axs[1])
    g1.set_xlabel('Predicted Values')
    g1.set_ylabel('Actual Values ')
    g1.xaxis.set_ticklabels(labels)
    g1.yaxis.set_ticklabels(labels)    
    if not all(pd.Series(labels).apply(lambda x: str(x).replace(".", "").isnumeric())):
        g1.set_xticklabels(g1.get_xticklabels(), rotation=30)
        g1.set_yticklabels(g1.get_xticklabels(), rotation=0)
    g2.set_xlabel('Predicted Values')
    g2.set_ylabel('Actual Values ')
    g2.xaxis.set_ticklabels(labels)
    g2.yaxis.set_ticklabels(labels)
    if not all(pd.Series(labels).apply(lambda x: str(x).replace(".", "").isnumeric())):
        g2.set_xticklabels(g2.get_xticklabels(), rotation=30)
        g2.set_yticklabels(g2.get_xticklabels(), rotation=0)
    plt.show()
    # save figure if on cloud
    if online_run:
        filename=f'./outputs/confusion_matrix_{name}.png'
        plt.savefig(filename, dpi=600)
        plt.close()

def extract_model_name(model):
    model_name = str(type(model)).split(".")[-1]
    return "".join([char for char in model_name if char not in string.punctuation])