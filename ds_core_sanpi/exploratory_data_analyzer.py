import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
from azureml.core import Run
from ds_core_sanpi import Regressor, Classifier
from lightgbm import LGBMRegressor, LGBMClassifier
from lofo import LOFOImportance, Dataset, plot_importance
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_percentage_error
from sklearn.preprocessing import PowerTransformer, StandardScaler


class EDA_Preprocessor:
        
    def __init__(self, data, keep_cols, drop_cols, problem, target_col="", online_run=False):
        """ construction of EDA_Preprocessor class 
        """
        # fix column names
        data = data.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        keep_cols = [re.sub('[^A-Za-z0-9_]+', '', i) for i in keep_cols]
        drop_cols = [re.sub('[^A-Za-z0-9_]+', '', i) for i in drop_cols]
        target_col = re.sub('[^A-Za-z0-9_]+', '', target_col)
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
        self.transform_cols = list(set(self.numeric_cols) - set(self.bool_cols + data.columns[data.isin([0,1]).all()].tolist()))
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

    def fill_missing_values(self, fill_by_zero_cols=None):
        """ fill missing numeric values by mean and categorical features 
            with 'Unknown' 
        """
        # filling nan values in given columns with zero
        if fill_by_zero_cols:
            self.df[fill_by_zero_cols] = self.df[fill_by_zero_cols].fillna(0)
        # filling nan values in numeric cols
        self.df[self.numeric_cols] = self.df[self.numeric_cols].fillna(
            self.df[self.numeric_cols].mean()
        ).fillna(0)
        # filling categorical cols with unknown class
        self.df[self.categorical_cols] = self.df[self.categorical_cols].fillna("Unknown").replace(r'^\s*$', "Unknown", regex=True)
        print("After filling missing values in EDA data:")
        print(self.df.isna().sum())

    def align_cols(self, cols):
        """ align the dataframe with columns given 
        """
        list_of_cols = list(set(cols)-set(self.df.columns.tolist()))
        print("Data is filled with following zero columns:\n", list_of_cols)
        for i in list_of_cols:
            self.df[i] = 0
        self.df = self.df[cols]
        print("Shape after alignment:", self.df.shape)

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

    def dummification(self):
        """ get into dummy cols out of categorical features 
        """
        print("Shape before dummification:", self.df.shape)
        self.df = pd.get_dummies(self.df, columns=self.categorical_cols)
        self.df = self.df.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        print("Shape after dummification:", self.df.shape)

    def power_transformation(self):
        """ do power transformation on the numeric columns to handle skewness 
            of the data and make it more closer to normal distribution
        """                          
        power = PowerTransformer(method='yeo-johnson', standardize=True)
        self.df[self.transform_cols] = pd.DataFrame(data = power.fit_transform(self.df[self.transform_cols]), columns=self.transform_cols)   
        print("Power transformation is done on the following numeric columns: total = ", len(self.transform_cols), 
              "\n", self.transform_cols)     
        return power

    def apply_power_transformer(self, transformer):
        """ do power transformation given a transformer object 
        """  
        self.df[self.transform_cols] = pd.DataFrame(data = transformer.transform(self.df[self.transform_cols]), columns=self.transform_cols)   
        print("Power transformation object is used to transform the following numeric columns: total = ", len(self.transform_cols), 
              "\n", self.transform_cols)     

    def standardizer(self):
        """ do standardization on the numeric columns
        """              
        scaler = StandardScaler()
        self.df[self.transform_cols] = pd.DataFrame(data = scaler.fit_transform(self.df[self.transform_cols]), columns=self.transform_cols)     
        print("Standardization is done on the following numeric columns: total = ", len(self.transform_cols), 
              "\n", self.transform_cols)     
        return scaler

    def show_skewness(self):
        """ calculate and print skewness of all transformation columns 
        """
        print("Skewness:")
        print(self.df[self.transform_cols].skew().sort_values(ascending=False))
    
    def distribution_columns(self, cols, name=""):
        """ show boxplot, density plot and histogram distribution
            of selected columns in dataframe 
        """
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

    def count_columns(self, cols, name=""):
        """ show count plots of given columns in dataframe 
        """
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
    
    def categorical_correlation(self, cols, name="", rotate=False):
        """ show correlation of given categorical features
            wrt. to target feature
        """
        if self.target != "": 
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
            raise AssertionError("Please set a target feature first!")

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

    def how_close_two_continious_features_are(self, col1, col2, lower_threshold, higher_threshold, res=0.05):
        """ show histogram distribution of the ratio between
            selected two continuous columns in the dataframe
            that wanted to be same in ideal case 
        """
        # take the ratio
        ratio = self.df[col1] / self.df[col2]
        ratio = ratio[(ratio > lower_threshold) & (ratio < higher_threshold)]
        print(f"{len(ratio)} of the entries are within given ratio threshold {lower_threshold} - {higher_threshold}")
        # evaluate
        df_ratio = pd.DataFrame({"ratio": ratio})
        df_ratio["evaluation"] = df_ratio["ratio"].apply(evaluate)
        print("Distribution according to the errors:")
        print(df_ratio["evaluation"].value_counts())
        # regression metrics
        self.regression_metrics(col1, col2)
        # histogram distribution
        plt.figure(figsize=(12, 9))
        sns.histplot(data=df_ratio, x="ratio", hue="evaluation", binwidth=res,
                     palette={"5%":  "#205072", 
                              "10%": "#33709c", 
                              "15%": "#329D9C",
                              "25%": "#56C596",
                              "50%": "#7BE495",
                              "off": "#CFF4D2"}, hue_order = ["5%", "10%", "15%", "25%", "50%", "off"])
        plt.show()
        # merge ratio data with self.data
        return pd.merge(self.df, df_ratio, left_index=True, right_index=True)

    def split_x_and_y(self):
        """ split target from the data 
        """
        if self.target != "":  
            data = self.df[self.df[self.target].notnull()]
            self.X = data.loc[:, list(set(data.columns) - set(self.keep_cols) - set([self.target]))].copy()
            self.y = data[self.target].copy()
            print("ID columns are removed from data:", len(self.keep_cols))
            print("Data is split into X and y:\n",
                  "\tX:", self.X.shape, "\n",
                  "\ty:", self.y.shape)
        else:
            raise AssertionError("Please set a target feature first!")

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
                model = LGBMClassifier() 
            elif self.problem == "regression":
                model = LGBMRegressor() 
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
    
    def get_lofo_importance(self):
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
                metric='f1_weighted'
                score_metric='f1_score'
            elif self.problem == "regression":
                metric='r2'
                score_metric='R2'
            lofo_imp = LOFOImportance(dataset, scoring=metric)
            importance_df = lofo_imp.get_importance()
            # plot importance
            plot_importance(importance_df, figsize=(12, 12))
            # find features that needs to be dropped!
            features = importance_df.feature.iloc[::-1].tolist()
            # define the model
            if self.problem == "classification":
                modeling = Classifier(df_lofo, [], self.target, self.online_run, False)
                base = modeling.cv_score_model(LGBMClassifier())[score_metric]
            elif self.problem == "regression":
                modeling = Regressor(df_lofo, [], self.target, self.online_run, False)
                base = modeling.cv_score_model(LGBMRegressor())[score_metric]
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
                    score = modeling.cv_score_model(LGBMClassifier())[score_metric]
                elif self.problem == "regression":
                    modeling = Regressor(df_lofo[features_ + [self.target]], [], self.target, self.online_run, False)
                    score = modeling.cv_score_model(LGBMRegressor())[score_metric]
                if score >= base:
                    base = score
                    print("\tYes since score now is:", score)
                    features = features[1:]
                    features_to_drop.append(feature_to_drop)
                else:
                    print("\tNo!")            
            print("Following features are dropped:\t", features_to_drop)
            print("Total features after LOFO:", len(features), "\tNew score:", base)
            self.df.drop(features_to_drop, axis=1, inplace=True)
            self.X.drop(features_to_drop, axis=1, inplace=True)
            self.numeric_cols = list(set(self.numeric_cols) - set(features_to_drop))
            self.transform_cols = list(set(self.transform_cols) - set(features_to_drop))
            self.categorical_cols = list(set(self.categorical_cols) - set(features_to_drop))
            self.bool_cols = list(set(self.bool_cols) - set(features_to_drop))
            return importance_df
        else:
            raise AssertionError("Please set a target feature first!")
    

# functions out of the class
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