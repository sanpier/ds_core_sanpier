import gc
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
import warnings
from azureml.core import Run
from category_encoders import TargetEncoder, LeaveOneOutEncoder, WOEEncoder
from catboost import CatBoostRegressor
from classifier_utils import Classifier
from lightgbm import LGBMRegressor, LGBMClassifier
from lofo import LOFOImportance, FLOFOImportance, Dataset, plot_importance
from regressor_utils import Regressor
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, recall_score, precision_score, roc_auc_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_predict, KFold
warnings.filterwarnings("ignore")


class EDA_Preprocessor:
        
    def __init__(self, data, keep_cols, drop_cols, problem, target_col="", online_run=False):
        """ construction of EDA_Preprocessor class 
        """
        # fix column names
        data = data.rename(columns = lambda x: standardize_column_name(x))
        keep_cols = [standardize_column_name(i) for i in keep_cols]
        drop_cols = [standardize_column_name(i) for i in drop_cols]
        target_col = standardize_column_name(target_col)
        # target 
        self.target = target_col
        # problem
        if problem not in ["classification", "regression"]:
            raise AssertionError("Please define problem as one of classification or regression!")
        self.problem = problem
        # id cols 
        self.keep_cols = keep_cols
        # bool columns
        self.bool_cols = list(set(data.select_dtypes(include=['bool']).columns.tolist()) - set(drop_cols) - set(keep_cols) - set([target_col]))
        # convert hidden object numeric cols to float
        object_cols = data.select_dtypes(include=['object']).columns.tolist()
        num_cols = [i for i in object_cols if all(data[data[i].notnull()][i].apply(lambda x: str(x).isnumeric()))]
        data[num_cols] = data[num_cols].fillna("0").astype(float)
        # categorical ones
        self.categorical_cols = list(set(object_cols) - set(num_cols) - set(drop_cols) - set(keep_cols) - set([target_col]))
        # numeric columns
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        self.numeric_cols = list(set(numeric_cols) - set(drop_cols) - set(keep_cols) - set([target_col]))
        # convert bool to integer
        data[self.bool_cols] = data[self.bool_cols].astype(int)
        # report columns
        print("EDA_Preprocessor instance initialized with data:\n",
             f"\tKeeping columns: {len(self.keep_cols)}\n",
             f"\tNumeric features: {len(self.numeric_cols)}\n",
             f"\tCategorical features: {len(self.categorical_cols)}\n",
             f"\tBoolean features: {len(self.bool_cols)}")
        # set transformation columns: not boolean ones, target neither the binary ones
        self.transform_cols = list(set(self.numeric_cols) - set(self.bool_cols + data.columns[data.isin([0,1,np.nan]).all()].tolist()))
        # set the final columns and data
        if self.target != "": 
            all_cols = keep_cols + sorted(self.numeric_cols) + sorted(self.bool_cols) + sorted(self.categorical_cols) + [self.target]
        else:
            all_cols = keep_cols + sorted(self.numeric_cols) + sorted(self.bool_cols) + sorted(self.categorical_cols)
        self.df = data[all_cols]
        self.df.reset_index(inplace=True, drop=True)
        print("EDA data is now as follows:")
        print(self.df.info())
        self.online_run = online_run        
        if self.online_run:
            self.run = Run.get_context()

    ### ENGINEER DATA ###
    def align_cols(self, cols):
        """ align the dataframe with columns given 
        """
        list_of_cols = list(set(cols)-set(self.df.columns.tolist()))
        print("Data is filled with following zero columns:\n", list_of_cols)
        for i in list_of_cols:
            self.df[i] = 0
        self.df = self.df[cols]
        print("Shape after alignment:", self.df.shape)

    def pca_decomposition(self, number_of_pc=-1, name=""):
        """ PCA decomposition of the data using only the numeric features
        """
        if self.target != "": 
            pc_size = number_of_pc 
            if number_of_pc == -1:
                pc_size = int(math.sqrt(len(self.numeric_cols)))
            pca = PCA(n_components=pc_size)
            principalComponents = pca.fit_transform(self.df[self.numeric_cols])
            pc_columns = ['pc_' + str(i+1) for i in range(pc_size)]
            principalDf = pd.DataFrame(data = principalComponents, columns = pc_columns)
            self.df = pd.concat([principalDf, self.df[self.target]], axis = 1)
            print("Size of the new data:", self.df.shape)
            # explained variance in PCA by plot
            exp_var = pca.explained_variance_ratio_.round(3)
            sns.set(rc={'figure.figsize':(2.5*math.sqrt(pc_size), 1.5*math.sqrt(pc_size))})
            sns.barplot(y=exp_var, x=pc_columns)
            plt.show()
            if self.online_run:
                # save figure
                filename=f'./outputs/pca_decomposition_explained_variance_{name}.png'
                plt.savefig(filename, dpi=600)
                plt.close()
            print("Explained variance in PCA:\n", exp_var,
                "\nTotal variance explained:", sum(exp_var).round(3))
            return pca
        else:
            raise AssertionError("Please set a target feature first!")
    
    ### DISTRIBUTION PLOTS ###
    def check_outliers(self, cols=None):
        """ check numeric outliers by percentile analysis
        """   
        if cols is None:  
            cols = self.numeric_cols
        for col in cols:
            print(col.upper())
            print("\tStatistical outliers:\t", get_extreme_values(self.df[col]))
            vfunc = np.vectorize(lambda x: "{:.2f}".format(x))
            print("\t0-1-5-50-95-99-100 Percentiles:\t", vfunc(np.nanpercentile(self.df[col], [0,1,5,50,95,99,100])))

    def numeric_distributions(self, cols=None, name=""):
        """ show boxplot, density plot and histogram distribution
            of selected columns in dataframe 
        """
        if cols is None:  
            cols = self.numeric_cols
        for col in cols:
            sns.set(rc={'figure.figsize':(6, 8)})
            fig, axs = plt.subplots(nrows=3)
            fig.suptitle(f"Distribution plots of: {col}", fontsize=20, y=0.95)
            sns.boxplot(data=self.df, x=col, ax=axs[0])
            sns.kdeplot(data=self.df, x=col, ax=axs[1])
            sns.histplot(data=self.df, x=col, ax=axs[2], stat="count", discrete=True)
            plt.show()
            if self.online_run:
                # save figure
                filepath=f'./outputs/distribution_of_{col}_{name}.png'
                plt.savefig(filepath, dpi=600)
                plt.close(fig)
        # check outliers
        self.check_outliers(cols)

    def count_distributions(self, cols=None, name=""):
        """ show count plots of given columns in dataframe 
        """
        if cols is None:  
            cols = self.categorical_cols
        for col in cols:
            sns.set(rc={'figure.figsize':(6, 5)})
            sns.countplot(y=col, 
                          data=self.df, 
                          palette="Set3", 
                          order=self.df[col].value_counts().index).set_title(f"Count plot of: {col}", fontsize=20)
            plt.show()
            if self.online_run:
                # save figure
                filepath=f'./outputs/count_of_{col}_{name}.png'
                plt.savefig(filepath, dpi=600)
                plt.close() 

    ### IMPUTATION FUNTIONS & FILL MISSING VALUES ###
    def check_null_features(self, threshold=0):
        """ check nan values 
        """
        nans = self.df.isna().sum()
        print(nans[nans > threshold].sort_values(ascending=False))

    def replace_inf_values(self):
        """ get rid of inf values 
        """
        df = self.df[self.numeric_cols]
        count = np.isinf(df).values.sum()
        print("Infinity values total count: ", count)
        col_name = df.columns.to_series()[np.isinf(df).any()]
        print("Infinity value columns: ")
        for i in col_name:
            print(f"\tfixing column {i}:", np.isinf(df[col_name]).values.sum())
            max_value = self.df[~self.df[i].isin([np.inf])][i].max()
            min_value = self.df[~self.df[i].isin([-np.inf])][i].min()
            self.df[i] = np.where(self.df[i].isin([np.inf]), max_value, self.df[i]) 
            self.df[i] = np.where(self.df[i].isin([-np.inf]), min_value, self.df[i]) 
            
    def fill_missing_values(self, fill_by_zero_cols=None, strategy="mean"):
        """ fill missing numeric values by mean and categorical features 
            with 'Unknown' 
        """
        # filling nan values in given columns with zero
        if fill_by_zero_cols:
            self.df[fill_by_zero_cols] = self.df[fill_by_zero_cols].fillna(0)
        # filling nan values in numeric cols
        if strategy == "mean":
            self.df[self.numeric_cols] = self.df[self.numeric_cols].fillna(
                self.df[self.numeric_cols].mean()
            ).fillna(0)
        elif strategy == "median":
            self.df[self.numeric_cols] = self.df[self.numeric_cols].fillna(
                self.df[self.numeric_cols].median()
            ).fillna(0)
        elif strategy == "zero":
            self.df[self.numeric_cols] = self.df[self.numeric_cols].fillna(0)
        # filling categorical cols with unknown class
        self.df[self.categorical_cols] = self.df[self.categorical_cols].fillna("Unknown").replace(r'^\s*$', "Unknown", regex=True)
        print("After filling missing values in EDA data:")
        print(self.df.isna().sum())

    def fill_given_col_by_mode(self, fill_col, by_mode_col):
        """ fill column by other column's mode
        """
        self.df[fill_col] = self.df.apply(lambda row: self.fill_given_col_by_mode_row(row, fill_col, by_mode_col), axis=1)
        
    def fill_given_col_by_mode_row(self, row, fill_col, by_mode_col):
        """ fill column by other column's mode per row
        """
        if self.df[self.df[fill_col].isna()].shape[0] == 0:
            print(f"Column {fill_col} is already full!")
            return
        if (row[fill_col] == None) | (not row[fill_col]) | (row[fill_col] != row[fill_col]):
            df_ = self.df[self.df[by_mode_col] == row[by_mode_col]]
            if df_.shape[0] > 0:
                return df_[fill_col].mode().iloc[0]
            else: 
                return row[fill_col]
        else:
            return row[fill_col]

    def impute_all(self, estimator=LGBMRegressor(), missing_values=np.nan, initial_strategy="mean", imputation_order="ascending"):
        """ fill missing columns by IterativeImputer 
            + missing_values: int or np.nan
            + initial_strategy: {"mean", "median", "most_frequent", "constant"}
            + imputation_order: {"ascending", "descending", "roman", "arabic", "random"}
        """
        categorical_cols = self.df.columns[self.df.dtypes==object].tolist()
        if len(categorical_cols) == 0:
            cols = list(set(self.df.columns) - set(self.keep_cols) - set([self.target]))
            imp_num = IterativeImputer(estimator=estimator, 
                                       missing_values=missing_values, 
                                       initial_strategy=initial_strategy,
                                       imputation_order=imputation_order,
                                       random_state=0)
            self.df[cols] = imp_num.fit_transform(self.df[cols])
            self.imputed_cols = cols
            self.check_null_features()
            return imp_num
        else:
            raise AssertionError("The data has categorical features, please first handle them!")

    def impute_with_classification(self, col):
        """ fill missing numeric values in given columns with a
            classification model
        """
        if self.df[self.df[col].isna()].shape[0] == 0:
            print(f"Column {col} is already full!")
            return
        df_copy = self.df.copy()
        df_copy.drop(columns=self.target, axis=1, inplace=True)
        categoric_cols = list(set(self.categorical_cols) - set([col]))
        if len(categoric_cols) > 0:
            # filling categorical cols with unknown class
            df_copy[categoric_cols] = df_copy[categoric_cols].fillna("Unknown").replace(r'^\s*$', "Unknown", regex=True)
            # dummification
            df_copy = pd.get_dummies(df_copy, columns=categoric_cols)
            df_copy = df_copy.rename(columns = lambda x: standardize_column_name(x))
        # choose columns to do modeling 
        features = list(set(df_copy.columns[df_copy.notnull().all()]) - set(self.keep_cols))
        df_copy = df_copy[features + [col]]
        # impute by predictons
        train_indexes = df_copy[df_copy[col].notnull()].index
        test_indexes = df_copy[df_copy[col].isnull()].index
        le = LabelEncoder()
        encoded_y = le.fit_transform(df_copy.iloc[train_indexes][col])
        classifier = LGBMClassifier(random_state=42)
        # measure how good it is
        preds = cross_val_predict(classifier, df_copy.iloc[train_indexes][features], encoded_y, cv=5)  
        scores = {}
        scores["accuracy"] = round(accuracy_score(encoded_y, preds, normalize=True), 3)  
        scores["recall"] = round(recall_score(encoded_y, preds, average='weighted'), 3) 
        scores["precision"] = round(precision_score(encoded_y, preds, average='weighted'), 3)    
        scores["f1_score"] = round(f1_score(encoded_y, preds, average='weighted'), 3)       
        [print('\t', key,':', val) for key, val in scores.items()]
        # fit model
        classifier.fit(df_copy.iloc[train_indexes][features], encoded_y)
        preds = classifier.predict(df_copy.iloc[test_indexes][features])
        self.df.loc[test_indexes, col] = le.inverse_transform(preds)
        # release memory
        gc.collect()
        del df_copy

    def impute_with_regression(self, col):
        """ fill missing numeric values in given columns with a
            regression model
        """
        if self.df[self.df[col].isna()].shape[0] == 0:
            print(f"Column {col} is already full!")
            return
        df_copy = self.df.copy()
        df_copy.drop(columns=self.target, axis=1, inplace=True)
        if len(self.categorical_cols) > 0:
            # filling categorical cols with unknown class
            df_copy[self.categorical_cols] = df_copy[self.categorical_cols].fillna("Unknown").replace(r'^\s*$', "Unknown", regex=True)
            # dummification
            df_copy = pd.get_dummies(df_copy, columns=self.categorical_cols)
            df_copy = df_copy.rename(columns = lambda x: standardize_column_name(x))
        # choose columns to do modeling 
        features = list(set(df_copy.columns[df_copy.notnull().all()]) - set(self.keep_cols))
        df_copy = df_copy[features + [col]]
        # impute by predictons
        train_indexes = df_copy[df_copy[col].notnull()].index
        test_indexes = df_copy[df_copy[col].isnull()].index
        regressor = LGBMRegressor(random_state=42)
        # measure how good it is
        preds = cross_val_predict(regressor, df_copy.iloc[train_indexes][features], df_copy.iloc[train_indexes][col], cv=5) 
        scores = {}
        scores["MAE"] = round(mean_absolute_error(df_copy.iloc[train_indexes][col], preds), 3)  
        scores["RMSE"] = round(mean_squared_error(df_copy.iloc[train_indexes][col], preds, squared=False), 3) 
        scores["R2"] = round(r2_score(df_copy.iloc[train_indexes][col], preds), 3)    
        scores["MAPE"] = round(mean_absolute_percentage_error(df_copy.iloc[train_indexes][col], preds), 3) 
        [print('\t', key,':', val) for key, val in scores.items()]
        # fit model
        regressor.fit(df_copy.iloc[train_indexes][features], df_copy.iloc[train_indexes][col])
        self.df.loc[test_indexes, col] = regressor.predict(df_copy.iloc[test_indexes][features])
        # release memory
        gc.collect()
        del df_copy

    ### HANDLE CATEGORICAL FEATURES ###
    def target_encoding_on_column(self, col, value_threshold):
        """ target encode the given categorical column 
        """
        categories = self.df[col].unique()
        df_ground_truth = self.df[self.df[self.target].notnull()]
        for cat in categories:
            if self.df[self.df[col] == cat].shape[0] > value_threshold:
                self.df.loc[self.df[col] == cat, col] = df_ground_truth.loc[df_ground_truth[col] == cat, self.target].mean()
            else:
                self.df.loc[self.df[col] == cat, col] = df_ground_truth[self.target].mean()
        self.df[col] = self.df[col].astype("float")

    def target_encoding(self, category_threshold, value_threshold, all_of_them=False):
        """ target encode all categorical columns 
        """
        if (self.problem == "regression") | self.df[self.target].isin([0,1,np.nan]).all():
            if all_of_them==False:
                # dummifiction
                dummy_cols = [col for col in self.categorical_cols if len(self.df[col].unique()) <= category_threshold]
                print("Dummfying ones with less than category threshold:", len(dummy_cols), "\n", dummy_cols)
                self.df = pd.get_dummies(self.df, columns=dummy_cols)
                self.df = self.df.rename(columns = lambda x: standardize_column_name(x))
                # target encoding
                target_cols = list(set(self.categorical_cols) - set(dummy_cols))
                print("Selected categorical ones are now under the process of target encoding:", len(target_cols))
                for category_col in target_cols:
                    self.target_encoding_on_column(category_col, value_threshold)
                    print(category_col, "is target encoded now!")
            else:
                # target encode all
                print("All categorical ones are now under the process of target encoding:", len(self.categorical_cols))
                for category_col in self.categorical_cols:
                    self.target_encoding_on_column(category_col, value_threshold)
                    print(category_col, "is target encoded now!")
        else:
            raise AssertionError("The classification problem is not applicable for target encoding!")

    def target_encoding_by_lib(self, method="target"):
        """ target encoding by category_encoders library
        """
        if (self.problem == "regression") | self.df[self.target].isin([0,1,np.nan]).all():
            if method == "target":
                encoder = TargetEncoder(handle_missing="return_nan", handle_unknown="return_nan")
            elif method == "leave_one_out":
                encoder = LeaveOneOutEncoder(handle_missing="return_nan", handle_unknown="return_nan")
            elif method == "woe":
                encoder = WOEEncoder(handle_missing="return_nan", handle_unknown="return_nan")
            self.df[self.categorical_cols] = encoder.fit_transform(self.df[self.categorical_cols], self.df[self.target])
            return encoder
        else:
            raise AssertionError("The classification problem is not applicable for target encoding!")

    def dummification(self, value_threshold=1000):
        """ get into dummy cols out of categorical features 
        """
        print("Shape before dummification:", self.df.shape)
        # generate dummies
        for cat_col_i in self.categorical_cols:
            cats = self.df[cat_col_i].value_counts()[lambda x: x > value_threshold].index
            df_cat_col_dummified = pd.get_dummies(pd.Categorical(self.df[cat_col_i], categories=cats)).rename(columns = lambda x: str(f"{cat_col_i}_{x}").lower())
            self.df = pd.concat([self.df, df_cat_col_dummified], axis=1, join='inner')
        # drop categorical columns and standardize new dummy column names
        self.df.drop(columns=self.categorical_cols, axis=1, inplace=True)
        self.df = self.df.rename(columns = lambda x: standardize_column_name(x))
        print("Shape after dummification:", self.df.shape)

    ### TRANSFORMATION FUNCTIONS ###
    def power_transformation(self):
        """ do power transformation on the numeric columns to handle skewness 
            of the data and make it more closer to normal distribution
        """                          
        power = PowerTransformer(method='yeo-johnson', standardize=True)
        self.df[self.transform_cols] = pd.DataFrame(data = power.fit_transform(self.df[self.transform_cols]), columns=self.transform_cols)   
        print("Power transformation is done on the following numeric columns: total = ", len(self.transform_cols), 
              "\n", self.transform_cols)     
        return power 

    def standardizer(self):
        """ do standardization on the numeric columns
        """              
        scaler = StandardScaler()
        self.df[self.transform_cols] = pd.DataFrame(data = scaler.fit_transform(self.df[self.transform_cols]), columns=self.transform_cols)     
        print("Standardization is done on the following numeric columns: total = ", len(self.transform_cols), 
              "\n", self.transform_cols)     
        return scaler

    def apply_transformer(self, transformer):
        """ apply transformation given a transformer object 
        """  
        self.df[self.transform_cols] = pd.DataFrame(data = transformer.transform(self.df[self.transform_cols]), columns=self.transform_cols)   
        print("Transformation object is used to transform the following numeric columns: total = ", len(self.transform_cols), 
              "\n", self.transform_cols)    
              
    def show_skewness(self):
        """ calculate and print skewness of all transformation columns 
        """
        print("Skewness:")
        print(self.df[self.transform_cols].skew().sort_values(ascending=False))
    
    ### CORRELATION FUNCTIONS ###
    def resolve_correlation(self, drop=True):
        """ remove highly correlated features 
        """
        # create correlation matrix
        corr_matrix = self.df[self.numeric_cols + self.bool_cols].corr().abs().round(2)
        # select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # find features with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        print("Highly correlated columns are:", to_drop)
        # drop features 
        if drop:
            self.numeric_cols = list(set(self.numeric_cols) - set(to_drop))
            self.transform_cols = list(set(self.transform_cols) - set(to_drop))
            self.df.drop(to_drop, axis=1, inplace=True)
            print("Shape becomes:", self.df.shape)
        if self.online_run:
            self.run.log_list(name='Highly correlated columns are:', value=to_drop)

    def heatmap_correlation(self, name=""):
        """ show correlation heatmap of numeric features 
        """ 
        if self.target != "": 
            features = self.transform_cols + [self.target] 
        else:
            features = self.transform_cols
        unit_size = np.sqrt(len(features)*3)
        df_analysis = self.df[features].corr().round(2)
        sns.set(rc={'figure.figsize':(unit_size*3, unit_size*2.5)})
        sns.heatmap(df_analysis, annot=True, cmap="YlOrRd", linewidths=1)
        plt.show()
        if self.online_run:
            # save figure
            filename=f'./outputs/heatmap_correlation_{name}.png'
            plt.savefig(filename, dpi=600)
            plt.close()
    
    def categorical_correlation(self, cols=None, name="", rotate=False):
        """ show correlation of given categorical features
            wrt. to target feature
        """
        if cols is None:  
            cols = self.categorical_cols
        if self.target != "": 
            if (self.problem == "regression") | self.df[self.target].isin([0,1,np.nan]).all():
                for col in cols:
                    sns.set(rc={'figure.figsize':(6, 5)})
                    ax = sns.catplot(x=col, y=self.target, kind="bar", data=self.df)
                    ax.fig.suptitle(f"Correlation plot of: {col} to {self.target}", fontsize=20, y=1.05)
                    if rotate:
                        ax.set_xticklabels(rotation=30)
                    print(self.df.groupby(col)[self.target].mean())
                    plt.show()
                    if self.online_run:
                        # save figure
                        filename=f'./outputs/categorical_correlation_{col}_{name}.png'
                        plt.savefig(filename, dpi=600)
                        plt.close()
            else:
                for col in cols:
                    for unique_val in self.df[col].unique().tolist():
                        value_counts = self.df[self.df[col] == unique_val][self.target].value_counts(normalize=True)
                        sns.set(rc={'figure.figsize':(6, 5)})
                        ax = sns.catplot(x=value_counts.index, y=value_counts.values, kind="bar", data=self.df)
                        ax.fig.suptitle(f"Correlation plot of: {col} of {unique_val}", fontsize=20, y=1.05)
                        if rotate:
                            ax.set_xticklabels(rotation=30)
                        plt.show()
                        if self.online_run:
                            # save figure
                            filename=f'./outputs/categorical_correlation_of_{col}_in_{unique_val}.png'
                            plt.savefig(filename, dpi=600)
                            plt.close()
        else:
            raise AssertionError("Please set a target feature first!")

    ### PERFORMANCE METRICS & PLOTS ###
    def regression_metrics(self, col1, col2):
        """ show metrics for selected two continuous columns to be
            compared with each other 
        """      
        MAE = mean_absolute_error(self.df[col1], self.df[col2])
        RMSE = mean_squared_error(self.df[col1], self.df[col2], squared=False)
        R2 = r2_score(self.df[col1], self.df[col2])
        LOG = np.sqrt(mean_squared_log_error(self.df[col1], self.df[col2]))
        MAPE = mean_absolute_percentage_error(self.df[col1], self.df[col2])
        print("Mean Absolute Error:", MAE)
        print("Root Mean Square Error:", RMSE)
        print("R2 Score:", R2)
        print("Mean Squared Log Error:", LOG)
        print("Mean Absolute Percentage Error:", MAPE)

    def reg_plot(self, col, name=""):
        """ relationship btw. target and given column
        """
        if self.target != "": 
            plt.figure(figsize=(15, 12))
            ax = sns.regplot(x=self.df[col], y=self.df[self.target], order=2)
            ax.set_xlabel(f'given column: {col}')
            ax.set_ylabel(f'target value: {self.target}')
            plt.show()
            if self.online_run:
                # save figure
                filepath=f'./outputs/reg_plot_of_{col}_{name}.png'
                plt.savefig(filepath, dpi=600)
                plt.close() 
        else:
            raise AssertionError("Please set a target feature first!")

    def how_close_two_continious_features_are(self, col1, col2, lower_threshold, higher_threshold, res=0.05):
        """ show histogram distribution of the ratio between
            selected two continuous columns in the dataframe
            that wanted to be same in ideal case 
        """
        # take the ratio
        df_ratio = self.df.copy()
        df_ratio["error"] = df_ratio[col1] - df_ratio[col2]
        df_ratio["percentage_error"] = df_ratio["error"]/df_ratio[col2] 
        df_ratio["abs_percentage_error"] = round(100*(np.absolute(df_ratio["percentage_error"])), 1) 
        # regression metrics
        print("Regression metrics:")
        self.regression_metrics(col1, col2)    
        # take the ratio
        df_ratio["ratio"] = df_ratio[col1] / df_ratio[col2]
        df_ratio_ = df_ratio[(df_ratio["ratio"] > lower_threshold) & (df_ratio["ratio"] < higher_threshold)]
        # counts of observation in different absolute percentage errors
        percentage_errors = [5, 10, 15, 25, 50]
        print("\nNumber of observations wrt. absolute percentage errors:")
        [print(f'<={val}%:', df_ratio[df_ratio["abs_percentage_error"] <= val].shape[0]) for val in percentage_errors]
        # set evaluation criteria based on absolute percentage errors
        df_ratio["evaluation"] = df_ratio["abs_percentage_error"].apply(evaluate)
        # histogram distribution
        plt.figure(figsize=(12, 9))
        sns.histplot(data=df_ratio.iloc[df_ratio_.index], x="ratio", hue="evaluation", binwidth=res,
                     palette={"5%":  "#205072", 
                              "10%": "#33709c", 
                              "15%": "#329D9C",
                              "25%": "#56C596",
                              "50%": "#7BE495",
                              "off": "#CFF4D2"}, hue_order = ["5%", "10%", "15%", "25%", "50%", "off"])
        plt.show()
        # return data
        return df_ratio

    ### SPLIT DATA & FEATURE IMPORTANCE ###
    def split_x_and_y(self):
        """ split target from the data 
        """
        if self.target != "": 
            df_ground_truth = self.df[self.df[self.target].notnull()]
            self.X = df_ground_truth.loc[:, list(set(df_ground_truth.columns) - set(self.keep_cols) - set([self.target]))].copy()
            self.y = df_ground_truth[self.target].copy()
            print("ID columns are removed from data:", len(self.keep_cols))
            print("Data is split into X and y:\n",
                  "\tX:", self.X.shape, "\n",
                  "\ty:", self.y.shape)
        else:
            raise AssertionError("Please set a target feature first!")

    def set_test_data(self):
        """ set test dataset """
        self.test = self.df[self.df[self.target].isnull()]
        self.test.reset_index(drop=True, inplace=True)

    def get_feature_importance(self, number_of_features=-1, name=""):
        """ get feature importance by LGBMClassifier model
        """
        if self.target != "": 
            # split data into X and y haven't been done yet
            if not hasattr(self, 'X'): 
                self.split_x_and_y()
            n_features = number_of_features 
            if number_of_features == -1:
                n_features = self.df.shape[1]
            # define the model
            if self.problem == "classification":
                model = LGBMClassifier(random_state=42) 
            elif self.problem == "regression":
                model = LGBMRegressor(random_state=42) 
            # fit the model
            model.fit(self.X, self.y)
            # get importances
            importances = model.feature_importances_
            importances = np.round(importances / sum(importances), 4)
            sorted_idx = importances.argsort()[(-1*n_features):]
            # summarize feature importance
            plt.figure(figsize=(np.sqrt(n_features)*2.5, np.sqrt(n_features*1.5)))
            plt.barh(self.X.columns[sorted_idx].astype(str), importances[sorted_idx])
            plt.show()
            # set important features
            important_features = self.X.columns[sorted_idx].tolist()
            unimportant_features = list(set(self.X.columns) - set(important_features))
            if self.online_run:
                self.run.log_list(name='Important features', value=important_features)
                # save figure
                filename=f'./outputs/feature_importance_{name}.png'
                plt.savefig(filename, dpi=600)
                plt.close()
            print("Feature importance is calculated, the important features are:\n", important_features)
            self.df.drop(unimportant_features, axis=1, inplace=True)
            self.X.drop(unimportant_features, axis=1, inplace=True)
            self.numeric_cols = list(set(self.numeric_cols) - set(unimportant_features))
            self.transform_cols = list(set(self.transform_cols) - set(unimportant_features))
            self.categorical_cols = list(set(self.categorical_cols) - set(unimportant_features))
            self.bool_cols = list(set(self.bool_cols) - set(unimportant_features))
            return important_features
        else:
            raise AssertionError("Please set a target feature first!")
    
    def get_lofo_importance(self, classification_score="weighted", fast=False, recursive_check=False, drop=True):
        """ get feature importance by LOFO
        """
        if self.target != "": 
            # split data into X and y haven't been done yet
            if not hasattr(self, 'X'): 
                self.split_x_and_y()
            # calcluate the LOFO importance dataframe
            df_lofo = pd.concat([self.X, self.y], axis=1)
            dataset = Dataset(df=df_lofo, target=self.target, features=self.X.columns)
            # define the metric
            if self.problem == "classification":
                if classification_score == "weighted":
                    metric='f1_weighted'
                elif classification_score == "macro":
                    metric='f1_macro'
                score_metric='f1_score'
            elif self.problem == "regression":
                metric='r2'
                score_metric='R2'
            if fast:
                lofo_imp = FLOFOImportance(dataset, scoring=metric)
            else:
                lofo_imp = LOFOImportance(dataset, scoring=metric)
            importance_df = lofo_imp.get_importance()
            # plot importance
            plot_importance(importance_df, figsize=(12, 12))
            # find features that needs to be dropped!
            if recursive_check:
                features = importance_df.feature.iloc[::-1].tolist()
                # define the model
                if self.problem == "classification":
                    modeling = Classifier(df_lofo, [], self.target, self.online_run, False)
                    base = modeling.cv_score_model(LGBMClassifier(random_state=42), score=classification_score)[score_metric]
                elif self.problem == "regression":
                    modeling = Regressor(df_lofo, [], self.target, self.online_run, False)
                    base = modeling.cv_score_model(LGBMRegressor(random_state=42))[score_metric]
                score = 1
                print("Total features before LOFO:", len(features), "\tBase score:", base)
                features_to_drop = []
                while score >= base:
                    feature_to_drop = features[0]
                    print(f"\tDo we drop feature: {feature_to_drop}?")
                    features_ = features[1:]
                    # score it according to problem and model
                    if self.problem == "classification":
                        modeling = Classifier(df_lofo[features_ + [self.target]], [], self.target, self.online_run, False)
                        score = modeling.cv_score_model(LGBMClassifier(random_state=42))[score_metric]
                    elif self.problem == "regression":
                        modeling = Regressor(df_lofo[features_ + [self.target]], [], self.target, self.online_run, False)
                        score = modeling.cv_score_model(LGBMRegressor(random_state=42))[score_metric]
                    if score >= base:
                        base = score
                        print("\tYes since score now is:", score)
                        features = features[1:]
                        features_to_drop.append(feature_to_drop)
                    else:
                        print("\tNo!")            
                print("Number of features after LOFO:", len(features), "\tNew score:", base)
            else:
                features_to_drop = importance_df[importance_df.importance_mean <= 0].feature.tolist()
            print("Following features are dropped:\t", features_to_drop)
            # drop features
            if drop:
                self.df.drop(features_to_drop, axis=1, inplace=True)
                self.X.drop(features_to_drop, axis=1, inplace=True)
                self.numeric_cols = list(set(self.numeric_cols) - set(features_to_drop))
                self.transform_cols = list(set(self.transform_cols) - set(features_to_drop))
                self.categorical_cols = list(set(self.categorical_cols) - set(features_to_drop))
                self.bool_cols = list(set(self.bool_cols) - set(features_to_drop))
            return importance_df
        else:
            raise AssertionError("Please set a target feature first!")
    

### AUXILIARY FUNCTIONS ###
def evaluate(x):
    """ evaluation intervals of the residual ratio between
        two continious features
    """
    if (x >= 50):
        value = "off"
    if (x <= 50):
        value = "50%"
    if (x <= 25):
        value = "25%"
    if (x <= 15):
        value = "15%"
    if (x <= 10):
        value = "10%"
    if (x <= 5):
        value = "5%"
    return value

def standardize_column_name(x):
    """ standardize given column string 
    """
    x = str(x).replace(" ", "_")
    x = re.sub("ä","ae", x)
    x = re.sub("ö","oe", x)
    x = re.sub("ü","ue", x)
    x = re.sub("ß","ss", x)
    return re.sub('[^A-Za-z0-9_]+', '', x)

def get_extreme_values(serie):
    """ find borders for detecting extreme values
    """
    q75, q25 = np.nanpercentile(serie, [75,25])
    intr_qr = q75 - q25 
    upper = q75+(1.5*intr_qr)
    lower = q25-(1.5*intr_qr)
    return ("{:.2f}".format(lower), "{:.2f}".format(upper))