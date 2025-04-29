import gc
import math
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import pingouin as pg
import re
import seaborn as sns
import statsmodels.api as sm
import sweetviz as sv
import warnings
from azureml.core import Run
from category_encoders import TargetEncoder, LeaveOneOutEncoder, WOEEncoder
from classifier_utils import Classifier, classification_metrics
from collections import defaultdict
from featimp import get_feature_importances, display_feature_importances
from lightgbm import LGBMRegressor, LGBMClassifier
from lofo import LOFOImportance, FLOFOImportance, Dataset, plot_importance
from natsort import natsorted
from regressor_utils import Regressor, regression_metrics
from scipy import stats
from scipy.special import inv_boxcox
from scipy.stats import boxcox, norm
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, PowerTransformer, QuantileTransformer, StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict, train_test_split, KFold
from statsmodels.graphics.gofplots import qqplot_2samples
warnings.filterwarnings("ignore")


class EDA_Preprocessor:
        
    def __init__(self, data, keep_cols, drop_cols, problem, target_col="", verbose=True, online_run=False):
        """ construction of EDA_Preprocessor class 
        """
        # fix column names
        data = data.rename(columns = lambda x: standardize_column_name(x))
        keep_cols = [standardize_column_name(i) for i in keep_cols]
        drop_cols = [standardize_column_name(i) for i in drop_cols]
        target_col = standardize_column_name(target_col)
        # target 
        self.target = target_col
        # verbose
        self.verbose = verbose
        # problem
        if problem not in ["classification", "regression"]:
            raise AssertionError("Please define problem as one of classification or regression!")
        self.problem = problem
        # id cols 
        self.keep_cols = keep_cols
        # bool columns
        bool_cols = []
        for i in list(set(data.columns) - set(drop_cols) - set(keep_cols) - set([target_col])):
            serie = data[i][data[i].notna()].astype("str")
            if serie.isin(["False", "True"]).all():
                bool_cols.append(i)
        data[bool_cols] = data[bool_cols].astype(float)
        #bool_cols = list(set(data[data.notna()].select_dtypes(include=['bool']).columns.tolist()) - set(drop_cols) - set(keep_cols) - set([target_col]))
        # convert hidden object numeric cols to float
        object_cols = data.select_dtypes(include=['object']).columns.tolist()
        num_cols = [i for i in object_cols if all(data[data[i].notnull()][i].apply(lambda x: str(x).isnumeric()))]
        data[num_cols] = data[num_cols].apply(lambda x: pd.to_numeric(x, errors='coerce'), axis=0) 
        # categorical ones
        self.categorical_cols = natsorted(list(set(object_cols) - set(num_cols) - set(drop_cols) - set(keep_cols) - set([target_col])))
        # convert bool to integer and define binary columns
        self.binary_cols = data.columns[data.isin([0,1,np.nan]).all()].tolist()
        self.binary_cols = natsorted(list(set(self.binary_cols) - set(self.categorical_cols) - set(drop_cols) - set(keep_cols) - set([target_col])))
        # numeric columns
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        self.numeric_cols = natsorted(list(set(numeric_cols) - set(self.binary_cols) - set(self.categorical_cols) - set(drop_cols) - set(keep_cols) - set([target_col])))
        # report columns
        if verbose:
            print("EDA_Preprocessor instance initialized with data:\n",
                f"\tKeeping columns: {len(self.keep_cols)}\n",
                f"\tNumeric features: {len(self.numeric_cols)}\n",
                f"\tCategorical features: {len(self.categorical_cols)}\n",
                f"\tBinary features: {len(self.binary_cols)}")
        # set the final columns and data
        all_cols = self.keep_cols + self.numeric_cols + self.binary_cols + self.categorical_cols
        if self.target != "": 
            all_cols = all_cols + [self.target]
        self.df = data[all_cols]
        if (self.target != ""):
            if (self.df[self.df[self.target].isnull()].shape[0] > 0): 
                self.df = self.df[self.target.notnull()]
                if verbose:
                    print("Rows with null target values are dropped:", self.df.shape)
        self.df.reset_index(inplace=True, drop=True)
        if verbose:
            print("EDA data is now as follows:")
            print(self.df.info(verbose=True, show_counts=True))
        self.online_run = online_run        
        if self.online_run:
            self.run = Run.get_context()

    ### COLUMN ENGINEERING ###
    def align_cols(self, cols):
        """ align the dataframe with columns given 
        """
        list_of_cols = list(set(cols)-set(self.df.columns.tolist()))
        print("Data is filled with following zero columns:\n", list_of_cols)
        for i in list_of_cols:
            self.df[i] = 0
        self.df = self.df[cols]
        print("Shape after alignment:", self.df.shape)

    def drop_cols(self, cols):
        """ drop given columns from the data
        """
        # drop given columns
        cols = list(set(cols).intersection(set(self.df.columns)))
        self.df.drop(cols, axis=1, inplace=True)
        self.numeric_cols = natsorted(list(set(self.numeric_cols) - set(cols)))
        self.categorical_cols = natsorted(list(set(self.categorical_cols) - set(cols)))
        self.binary_cols = natsorted(list(set(self.binary_cols) - set(cols)))
        # report columns now
        print(f"Data shape: {self.df.shape}\n",
              f"\tKeeping columns: {len(self.keep_cols)}\n",
              f"\tNumeric features: {len(self.numeric_cols)}\n",
              f"\tCategorical features: {len(self.categorical_cols)}\n",
              f"\tBinary features: {len(self.binary_cols)}")

    def convert_to_categoric(self, cols):
        """ convert given numeric columns to categorical
        """
        cols = list(set(cols).difference(set(self.binary_cols).union(set(self.categorical_cols))))
        if len(cols) > 0:
            self.numeric_cols = natsorted(list(set(self.numeric_cols).difference(set(cols))))
            self.categorical_cols = natsorted(list(set(self.categorical_cols).union(set(cols))))        
            for i in cols:
                if i in list(self.df.columns):
                    self.df[i] = self.df[i].apply(lambda v: str(int(v)) if not pd.isnull(v) else None)
            all_cols = self.keep_cols + self.numeric_cols + self.binary_cols + self.categorical_cols
            if self.target != "": 
                all_cols = all_cols + [self.target]
            self.df = self.df[all_cols]

    ### DIMENSIONALITY REDUCTION ###
    def pca_decomposition(self, cols=None, n_components=-1, prefix="", name=""):
        """ PCA decomposition of the data for dimensionality
            reduction
        """
        if cols is None:  
            cols = self.numeric_cols
        if n_components == -1:
            n_components = int(math.sqrt(len(cols)))
        pca = PCA(n_components=n_components)
        pca_clusters = pca.fit_transform(self.df[cols])
        pca_columns = ['pca_' + str(i+1) for i in range(n_components)]
        if prefix != "":
            pca_columns = [prefix + "_" + col for col in pca_columns]
        df_pca = pd.DataFrame(data = pca_clusters, columns = pca_columns)
        self.df = pd.concat([self.df, df_pca], axis = 1)
        self.numeric_cols = self.numeric_cols + pca_columns
        print("Size of the new data:", self.df.shape)
        # explained variance in PCA by plot
        exp_var = pca.explained_variance_ratio_.round(3)
        sns.set(rc={'figure.figsize':(2.5*math.sqrt(n_components), 1.5*math.sqrt(n_components))})
        sns.barplot(y=exp_var, x=pca_columns)
        plt.show()
        if self.online_run:
            # save figure
            filename=f'./outputs/pca_decomposition_explained_variance_{name}.png'
            plt.savefig(filename, dpi=600)
            plt.close()
        print("Explained variance in PCA:\n", exp_var,
              "\nTotal variance explained:", sum(exp_var).round(3))
        return pca
    
    def linear_discriminant_analysis(self, cols=None, n_components=-1, prefix="", name=""):
        """ Linear discriminant analysis of the data for dimensionality
            reduction
        """
        if self.target != "":
            if cols is None:  
                cols = self.numeric_cols
            if n_components == -1:
                n_components = int(math.sqrt(len(cols)))
            print(n_components)
            lda = LinearDiscriminantAnalysis(n_components=n_components)
            lda_clusters = lda.fit_transform(self.df[cols], self.df[self.target])
            lda_columns = ['lda_' + str(i+1) for i in range(n_components)]
            if prefix != "":
                lda_columns = [prefix + "_" + col for col in lda_columns]
            df_lda = pd.DataFrame(data = lda_clusters, columns = lda_columns)
            self.df = pd.concat([self.df, df_lda], axis = 1)
            self.numeric_cols = self.numeric_cols + lda_columns
            print("Size of the new data:", self.df.shape)
            # explained variance in LDA by plot
            exp_var = lda.explained_variance_ratio_.round(3)
            sns.set(rc={'figure.figsize':(2.5*math.sqrt(n_components), 1.5*math.sqrt(n_components))})
            sns.barplot(y=exp_var, x=lda_columns)
            plt.show()
            if self.online_run:
                # save figure
                filename=f'./outputs/lda_explained_variance_{name}.png'
                plt.savefig(filename, dpi=600)
                plt.close()
            print("Explained variance in LDA:\n", exp_var,
                "\nTotal variance explained:", sum(exp_var).round(3))
            return lda
        else:
            raise AssertionError("A target is needed to use this function!")

    def truncated_svd(self, cols=None, n_components=-1, prefix="", name=""):
        """ Dimensionality reduction by singular value decomposition
        """
        if cols is None:  
            cols = self.numeric_cols
        if n_components == -1:
            n_components = int(math.sqrt(len(cols)))
        svd = TruncatedSVD(n_components=n_components)
        svd_clusters = svd.fit_transform(self.df[cols])
        svd_columns = ['svd_' + str(i+1) for i in range(n_components)]
        if prefix != "":
            svd_columns = [prefix + "_" + col for col in svd_columns]
        df_svd = pd.DataFrame(data = svd_clusters, columns = svd_columns)
        self.df = pd.concat([self.df, df_svd], axis = 1)
        self.numeric_cols = self.numeric_cols + svd_columns
        print("Size of the new data:", self.df.shape)
        # explained variance in SVD by plot
        exp_var = svd.explained_variance_ratio_.round(3)
        sns.set(rc={'figure.figsize':(2.5*math.sqrt(n_components), 1.5*math.sqrt(n_components))})
        sns.barplot(y=exp_var, x=svd_columns)
        plt.show()
        if self.online_run:
            # save figure
            filename=f'./outputs/svd_explained_variance_{name}.png'
            plt.savefig(filename, dpi=600)
            plt.close()
        print("Explained variance in SVD:\n", exp_var,
              "\nTotal variance explained:", sum(exp_var).round(3))
        return svd

    def apply_dimension_reduction(self, df, reductor, prefix, decomposition_cols):
        """ apply dimension reduction to the data by the
            object that is fit before
        """
        if df is None:
            df = self.df.copy()
        clusters = reductor.transform(df[decomposition_cols])
        columns = [f'{prefix}_' + str(i+1) for i in range(reductor.n_components)]
        df_ = pd.DataFrame(data=clusters, columns=columns)
        df = pd.concat([df, df_], axis = 1)
        self.numeric_cols = self.numeric_cols + columns
        return df


    ### OUTLIER DETECTION ###    
    def get_outliers(self, cols=None):
        """ check numeric outliers by percentile analysis
        """  
        outlier_df = pd.DataFrame(columns=['Feature', 'Total Outliers', 'Lower limit', 'Upper limit', 
                                           'Min', '1th', '5th', 'Median', '95th', '99th', 'Max']) 
        if cols is None:  
            cols = self.numeric_cols
        for col in cols:
            lower, upper, total = calc_interquartile(self.df, col)
            if total != 0:
                row_df = pd.DataFrame([{
                    'Feature': col, 
                    'Total Outliers': total,
                    'Lower limit': lower,
                    'Upper limit': upper, 
                    'Min': "{:.2f}".format(np.nanpercentile(self.df[col], 0)),
                    '1th': "{:.2f}".format(np.nanpercentile(self.df[col], 1)),
                    '5th': "{:.2f}".format(np.nanpercentile(self.df[col], 5)),
                    'Median': "{:.2f}".format(np.nanpercentile(self.df[col], 50)),
                    '95th': "{:.2f}".format(np.nanpercentile(self.df[col], 95)),
                    '99th': "{:.2f}".format(np.nanpercentile(self.df[col], 99)),
                    'Max': "{:.2f}".format(np.nanpercentile(self.df[col], 100))
                }])
                outlier_df = pd.concat([outlier_df, row_df], axis=0, ignore_index=True)
        return outlier_df

    def isolation_forest_anomaly_detection(self, cols=None, contamination=0.05):
        """ apply isolation forest model to detect anomaly/outliers 
            in the data. identify if a point is: 
                + an outlier (-1) or 
                + an inlier (1) with the value specified in anomaly column
        """
        isof = IsolationForest(contamination=contamination,random_state=42)
        df_anomaly = self.df.copy()
        if cols is None:  
            cols = self.numeric_cols + self.binary_cols
        isof.fit(df_anomaly[cols])
        df_anomaly['anomaly_scores'] = isof.decision_function(df_anomaly[cols])
        df_anomaly['anomaly'] = isof.predict(df_anomaly[cols])
        # drop outliers
        anomaly_indexes = df_anomaly[df_anomaly['anomaly'] == -1].index
        self.df = self.df.drop(index=anomaly_indexes).reset_index(drop=True)
        return df_anomaly

    ### DISTRIBUTION PLOTS ###
    def distribution_vs_gaussian(self, col=None, name=""):
        """ show the distribution of given column vs gaussian
            if no column given then check for the target column
        """     
        if col is None:
            col = self.target
            if self.problem != "regression":  
                raise AssertionError("Target needs to be a continuous numeric feature!")
        _, axes = plt.subplots(1, 2, figsize=(20, 7))
        sns.histplot(self.df[col], kde=True, stat="density", ax=axes[0])
        mu, std = norm.fit(self.df[col])
        xmin, xmax = axes[0].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        axes[0].plot(x, p, 'k', linewidth=2)
        axes[0].set(title='Distribution of target vs Gaussian')
        pg.qqplot(self.df[col], dist='norm', ax=axes[1]) 
        axes[1].set(title='QQ-Plot of Target')
        if self.online_run:
            # save figure
            filepath=f'./outputs/distribution_vs_gaussian_{name}.png'
            plt.savefig(filepath, dpi=600)
            plt.close() 

    def feature_analysis(self, cols=None, method='kde', hue=None, sample=None, name=""):
        """
        Feature analysis function:
        - For numerical features: Shows distribution and rolling window correlation as two separate plots.
        - For categorical features: Shows distribution with NULL as a category and average target.
        """
        if hue == "target":
            hue = self.target
        if sample:
            data = self.df.sample(sample)
        # remove categorical columns with too many unique values
        cat_cols_nunique = data[self.categorical_cols].nunique()
        too_much_cat_values = cat_cols_nunique[cat_cols_nunique > 10].index.tolist()
        if cols is None:  
            cols = self.numeric_cols + self.binary_cols + self.categorical_cols
            cols = natsorted([i for i in cols if i in self.numeric_cols]) + \
                   natsorted([i for i in cols if i in self.binary_cols]) +  \
                   natsorted([i for i in cols if (i in self.categorical_cols) & (i not in too_much_cat_values)])
        else: 
            cols = natsorted([i for i in cols if(i not in too_much_cat_values)])
        numeric_cols = [i for i in cols if i in self.numeric_cols]
        categorical_cols = [i for i in cols if i in self.binary_cols + self.categorical_cols]
            
        #### 1. numerical features: Create 4 plots per row
        nrows = int(len(numeric_cols) / 2) + (len(numeric_cols) % 2 > 0)
        fig, axes = plt.subplots(nrows, 4, figsize=(20, nrows * 5))
        axes = axes.ravel()
        rolling_num = round(len(data) / 5)
        for idx, col in enumerate(numeric_cols):
            # 1.1. left plot: distribution
            ax_dist = axes[idx * 2]
            if method == 'kde':
                sns.kdeplot(data=data, x=col, hue=hue, ax=ax_dist, common_norm=False)
            elif method == 'cdf':
                sns.ecdfplot(data=data, x=col, hue=hue, ax=ax_dist)
            elif method == 'hist':
                sns.histplot(data=data, x=col, alpha=0.8, hue=hue, ax=ax_dist, common_norm=False, stat="density", discrete=True)
            elif method == 'box':
                sns.boxplot(data=data, x=hue, y=col, ax=ax_dist)
                if hue:
                    ax_dist.set_xlabel(hue)
            ax_dist.set_title(f"Distribution of {col}")
            # 1.2. right plot: rolling window correlation
            ax_roll = axes[idx * 2 + 1]
            temp = data.sort_values(col)
            temp.reset_index(inplace=True)
            ax_roll.scatter(temp.index, temp[self.target].rolling(rolling_num).mean(), s=1, alpha=0.5, label="Rolling Target Mean")
            # null value analysis for rolling window
            null_mask = temp[col].isnull()
            if null_mask.sum() > 0:
                null_target_mean = temp.loc[null_mask, self.target].mean()
                null_proportion = null_mask.mean()
                ax_roll.axhline(null_target_mean, color='red', linestyle='--', label=f"Null ({null_proportion:.1%}) Target Mean: {null_target_mean:.2f}")
            ax_roll.set_title(f"Rolling Correlation of {col}")
            ax_roll.legend(loc='best')
            del temp
            gc.collect()
        # hide unused numerical subplots
        for ax in axes[len(numeric_cols) * 2:]:
            ax.set_visible(False)
        fig.tight_layout()
        plt.show()

        #### 2. categorical features: 3 plots per row, each showing histogram + target distribution
        nrows = int(len(categorical_cols) / 3) + (len(categorical_cols) % 3 > 0)
        if nrows > 0:
            fig, axes = plt.subplots(nrows, 3, figsize=(20, nrows * 5))
            axes = axes.ravel()
            for idx, col in enumerate(categorical_cols):
                temp_df = self.df.copy()
                # add a category for null values
                temp_df[col] = temp_df[col].fillna('NULL')
                labels = temp_df[col].value_counts().index.tolist()
                # histogram on the left y-axis
                sns.countplot(x=col, data=temp_df, palette="Set3", order=labels, ax=axes[idx])
                #if len(' '.join([str(x) for x in labels])) > 40:
                #    axes[idx].set_xticklabels(labels, rotation=30)
                axes[idx].set_ylabel('Counts')
                # average target on the right y-axis
                if self.target != "":
                    if (self.problem == "regression") or self.df[self.target].isin([0, 1, np.nan]).all():
                        temp = temp_df.groupby(col)[self.target].agg(['mean', 'size']).reset_index().sort_values(by=['size'], ascending=False)
                        temp[col] = temp[col].astype("object")
                        temp = temp.set_index(col).loc[labels].reset_index()
                        ax2 = axes[idx].twinx()
                        ax2.plot(temp.index, temp['mean'], color='m', marker='o', label='Avg. Target')
                        ax2.tick_params(axis='y', colors='m')
                        ax2.set_ylabel('Avg. Target', color='m')                    
                axes[idx].set_xlabel(col)
                axes[idx].set_title(f"Distribution and Avg. Target of {col}")
            # hide unused categorical subplots
            for ax in axes[len(categorical_cols):]:
                ax.set_visible(False)
            fig.tight_layout()
            plt.show()
        if self.online_run:
            filepath = f'./outputs/combined_feature_analysis_{name}.png'
            plt.savefig(filepath, dpi=600)
            plt.close()

    def compare_two_data(self, df, label1, label2, cols=None, method="cdf", name=""):
        """ checking whether distributions of the given columns are similar 
            or not in two datasets: this can be used to detect possible
            covariance shift btw. training and test sets
        """
        if cols is None:  
            cols = self.numeric_cols + self.binary_cols + self.categorical_cols
        # remove categorical columns with too much unique values
        cat_cols_nunique = self.df[self.categorical_cols].nunique()
        too_much_cat_values = cat_cols_nunique[cat_cols_nunique > 12].index.tolist()
        cols = natsorted([i for i in cols if i in self.numeric_cols]) + \
               natsorted([i for i in cols if i in self.binary_cols]) +  \
               natsorted([i for i in cols if (i in self.categorical_cols) & (not i in too_much_cat_values)])
        # create one data out of both
        temp = pd.concat([self.df[cols], df[cols]], axis=0).reset_index(drop=True)
        temp["label"] = pd.Series([label1] * len(self.df) + [label2] * len(df))
        nrows = int(len(cols) / 4) + 1
        _, axes = plt.subplots(nrows, 4, figsize=(16, round(nrows*14/4)))
        for ax, col in zip(axes.ravel()[:len(cols)], cols):
            if col in self.numeric_cols:
                # numeric distribution
                if method == 'kde':
                    sns.kdeplot(data=temp, x=col, hue="label", ax=ax, fill=True, common_norm=False)
                elif method == 'cdf':
                    sns.ecdfplot(data=temp, x=col, hue="label", ax=ax)
                elif method == 'hist':
                    sns.histplot(data=temp, x=col, alpha=0.8, hue="label", ax=ax, common_norm=False, stat="density", discrete=True)
                elif method == 'box':
                    sns.boxplot(data=temp, x="label", y=col, ax=ax)  
                    ax.set_xlabel("datasets")   
                elif method == "qqplot":
                    pp_x = sm.ProbPlot(self.df[col])
                    pp_y = sm.ProbPlot(df[col])
                    qqplot_2samples(pp_x, pp_y, f"{col} quantiles in 1st data", f"{col} quantiles in 2nd data", line="45", ax=ax)      
            else:     
                # categoric distribution
                labels = temp[col].value_counts().index
                df = (temp.groupby(['label'])[col].value_counts(normalize=True)*100).unstack().reset_index()
                melted_df = pd.melt(df, id_vars=['label'], var_name=col, value_name='percentage')
                sns.barplot(data=melted_df, x=col, y='percentage', hue='label', hue_order=temp['label'].unique(), estimator=sum, ax=ax)
                if not all(pd.Series(labels).apply(lambda x: str(x).replace(".", "").isnumeric())) | \
                    (pd.Series(labels).apply(lambda x: len(str(x))).max() == 1):
                    ax.set_xticklabels(labels, rotation=30)
                ax.set_xlabel(col)
                ax.set_ylabel('percentage')
        for ax in axes.ravel()[len(cols):]:
            ax.set_visible(False)
        plt.tight_layout(w_pad=1)
        plt.suptitle(f'Distribution comparison of two datasets: {label1} vs {label2}', fontsize=20, y=1.02)
        plt.show()
        if self.online_run:
            filepath=f'./outputs/qqplot_{name}.png'
            plt.savefig(filepath, dpi=600)
            plt.close() 
        del temp
        gc.collect()
    
    def html_eda_report(self, df=None):
        """ sweetviz EDA report as an html page
        """
        if df is None:
            html_report = sv.analyze(self.df)
        else:
            feature_config = sv.FeatureConfig(skip=self.keep_cols)
            all_cols = self.keep_cols + self.numeric_cols + self.binary_cols + self.categorical_cols
            df = df[all_cols]
            html_report = sv.compare([self.df, "Train"], [df, "Test"], self.target, feature_config)
        html_report.show_html("EDA_report.html")

    ### IMPUTATION FUNTIONS & FILL MISSING VALUES ###
    def check_missing_values(self, threshold = 50, name=""):
        """ test how target is changing for the samples where the features are null
        """
        if self.target != "":  
            if (self.problem == "regression") | self.df[self.target].isin([0,1,np.nan]).all():
                # missing values
                nans = self.df.isna().sum()
                nans = nans[nans > 50].sort_values(ascending=False)
                if len(nans) > 0:
                    # start by plotting the bell curve
                    plt.figure(figsize=(15, 6))
                    z_ticks = np.linspace(-3.5, 3.5, 61)
                    pdf = stats.norm.pdf(z_ticks)
                    plt.plot(z_ticks, pdf)
                    # what is average of target
                    avg_target_population = self.df[self.target].mean()
                    # calculate the conditional average target for every missing feature
                    print('feature                                         #_missing     avg_target         z    p-value')
                    for f in nans.index.tolist():
                        sample_size = self.df[f].isna().sum()
                        avg_target_sample = self.df[self.df[f].isna()][self.target].mean()
                        z = (avg_target_sample - avg_target_population) / (self.df[self.target].std() / np.sqrt(sample_size))
                        plt.scatter([z], [stats.norm.pdf(z)], c='r' if abs(z) > 2 else 'g', s=100)
                        print(f"{f:45} :   {sample_size:7}        {avg_target_sample:7.3f}     {z:5.2f}      {2*stats.norm.cdf(-abs(z)):.3f}")
                        if abs(z) > 1: plt.annotate(f"{f}: {avg_target_sample:.3f}",
                                                    (z, stats.norm.pdf(z)),
                                                    xytext=(0,10), 
                                                    textcoords='offset points', ha='left' if z > 0 else 'right',
                                                    color='r' if abs(z) > 2 else 'g')
                    # annotate the center (z=0)
                    plt.vlines([0], 0, 0.05, color='g')
                    plt.annotate(f"z_score = 0\naverage target: {avg_target_population:.3f}",
                                                        (0, 0.05),
                                                        xytext=(0,10), 
                                                        textcoords='offset points', ha='center',
                                                        color='g')
                    plt.title('Average target when feature is missing')
                    plt.yticks([])
                    plt.xlabel('z_score')
                    plt.show()
                    if self.online_run:
                        filepath=f'./outputs/null_vs_target_{name}.png'
                        plt.savefig(filepath, dpi=600)
                        plt.close() 
        print("Null value counts per feature:")
        nans = self.df.isna().sum()
        print(nans[nans > threshold].sort_values(ascending=False))

    def nullity_plot(self, method):
        """ various plots to explain and identify nullity or completeness
            of the data
             1) method == heatmap: correlation ranges from:
                + -1: if one variable appears the other definitely does not
                +  0: variables appearing or not appearing have no effect on one another
                +  1: if one variable appears the other definitely also does
             2) method == dendogram: hierarchical clustering algorithm
             3) method == bar: number of non-null values per each feature
             4) method == matrix: shows how missing data is distributed
        """
        if method == "heatmap":
            msno.heatmap(self.df) # figsize=(x, y)
        elif method == "dendrogram":
            msno.dendrogram(self.df)
        elif method == "bar":
            msno.bar(self.df)
        elif method == "matrix":
            msno.matrix(self.df)

    def replace_inf_values(self, value=None):
        """ get rid of inf values 
        """
        df = self.df[self.numeric_cols]
        count = np.isinf(df).values.sum()
        print("Infinity values total count: ", count)
        col_name = df.columns.to_series()[np.isinf(df).any()]
        print("Infinity value columns: ")
        for i in col_name:
            print(f"\tfixing column {i}:", np.isinf(df[col_name]).values.sum())
            if value:
                self.df[i] = np.where(self.df[i].isin([np.inf]), value, self.df[i]) 
                self.df[i] = np.where(self.df[i].isin([-np.inf]), value, self.df[i]) 
            else:
                max_value = self.df[~self.df[i].isin([np.inf])][i].max()
                min_value = self.df[~self.df[i].isin([-np.inf])][i].min()
                self.df[i] = np.where(self.df[i].isin([np.inf]), max_value, self.df[i]) 
                self.df[i] = np.where(self.df[i].isin([-np.inf]), min_value, self.df[i]) 
            
    def generate_null_indicators(self, cols=None):
        """ create null indicator columns for given cols, or for all 
        """
        if cols is None:  
            cols = self.numeric_cols + self.binary_cols + self.categorical_cols
        # check given cols for null values
        for col in cols:
            if col in self.numeric_cols + self.binary_cols:
                if self.df[self.df[col].isna()].shape[0] > 0:
                    self.df[f"is_{col}_there"] = np.where(self.df[col].notna(), 1, 0)
                    self.binary_cols = natsorted(self.binary_cols + [f"is_{col}_there"])
            else:
                self.df[col] = self.df[col].fillna("Unknown").replace(r'^\s*$', "Unknown", regex=True)

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
        # filling binary cols with median    
        self.df[self.binary_cols] = self.df[self.binary_cols].fillna(
            self.df[self.binary_cols].median()
        ).fillna(0)
        # filling categorical cols with unknown class
        self.df[self.categorical_cols] = self.df[self.categorical_cols].fillna("Unknown").replace(r'^\s*$', "Unknown", regex=True)
        if self.verbose:
            print("All missing values in the data is imputed now!")

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
        nans = self.df.isna().sum()
        cols = nans[nans > 0].index.tolist()
        cols = list(set(cols) - set(self.keep_cols) - set([self.target]))
        for i in cols:
            self.df[i + "_missing"] = np.where(self.df[i].isna(), 1, 0)
        categorical_cols = self.df[cols].columns[self.df[cols].dtypes==object].tolist()
        if len(categorical_cols) == 0:
            imp_num = IterativeImputer(estimator=estimator, 
                                       missing_values=missing_values, 
                                       initial_strategy=initial_strategy,
                                       imputation_order=imputation_order,
                                       random_state=0)
            self.df[cols] = imp_num.fit_transform(self.df[cols])
            print("Imputation is done on the following columns: total = ", len(cols), 
                  "\n", cols) 
            self.count_nulls()
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
        if self.target != "":
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
        scores = classification_metrics(encoded_y, preds, score="weighted", model_name="LGBMC")     
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
        if self.target != "":
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
        scores = regression_metrics(df_copy.iloc[train_indexes][col], preds, model_name="LGBMR")
        [print('\t', key,':', val) for key, val in scores.items()]
        # fit model
        regressor.fit(df_copy.iloc[train_indexes][features], df_copy.iloc[train_indexes][col])
        self.df.loc[test_indexes, col] = regressor.predict(df_copy.iloc[test_indexes][features])
        # release memory
        gc.collect()
        del df_copy

    ### HANDLE CATEGORICAL FEATURES ###
    def fit_target_encoding(self, cols=None, smooth=20, n_folds=5, postfix=""):
        """ fit smoothed target encoding on multiple features with k-fold cross-validation.
            parameters:
                cols: list of feature columns to encode.
                smooth: smoothing factor (higher = closer to global mean).
                n_folds: number of folds for cross-validation.
            returns:
                encoded training data and dictionary of encoding mappings.
        """
        if (self.problem == "regression") | self.df[self.target].isin([0,1,np.nan]).all():
            # set columns to encode
            if cols is None:  
                cols = self.categorical_cols
            if self.verbose:
                print(f"The following {len(cols)} columns are now under the process of target encoding:\n", cols)
            # set global mean
            global_mean = self.df[self.target].mean() 
            # set k-folder
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            # store categorical encoder mappings            
            encoding_dict = {}
            # for all given columns
            for col in cols:
                encoded_feature = pd.Series(index=self.df.index, dtype=np.float32)
                fold_mappings = []
                # prevents leakage
                for train_idx, val_idx in kf.split(self.df):
                    train_fold = self.df.iloc[train_idx]
                    val_fold = self.df.iloc[val_idx]
                    # compute category statistics
                    category_stats = train_fold.groupby(col)[self.target].agg(["mean", "count"])
                    # apply smoothing
                    category_stats["smooth_mean"] = (category_stats["mean"] * category_stats["count"] + 
                                                    global_mean * smooth) / (category_stats["count"] + smooth)
                    fold_mappings.append(category_stats["smooth_mean"])
                    # assign fold values
                    encoded_feature.iloc[val_idx] = val_fold[col].map(category_stats["smooth_mean"])
                # final mapping: average across folds
                final_mapping = pd.concat(fold_mappings, axis=1).mean(axis=1).to_dict()
                # default for unseen categories
                final_mapping["Unknown"] = global_mean
                encoding_dict[col] = final_mapping
                # assign to encoded features                
                target_col = col + "_" + postfix  
                if postfix == "": 
                    target_col = col 
                self.df[target_col] = encoded_feature.fillna(global_mean)
                print(col, "is target encoded now!")
            return encoding_dict
        else:
            raise AssertionError("The classification problem is not applicable for target encoding!")
    
    def apply_target_encoding(self, encoder, df=None, postfix=""):
        """ given the dataframe and encoder dictionary this function applies 
            the categorical encoding to the data provided
        """
        # if no dataframe provided then do it to the instance of this class
        if df is None:  
            for col in list(encoder.keys()):
                if col in self.df.columns:
                    target_col = col + "_" + postfix  
                    if postfix == "": 
                        target_col = col   
                    self.df[target_col] = self.df[col].map(encoder[col]).fillna(encoder[col]["Unknown"])
        else:
            for col in list(encoder.keys()):
                if col in df.columns:
                    target_col = col + "_" + postfix  
                    if postfix == "": 
                        target_col = col   
                    df[target_col] = df[col].map(encoder[col]).fillna(encoder[col]["Unknown"])
            return df

    def target_encoding_by_library(self, cols=None, method="target", smoothing=None, postfix=""):
        """ target encoding by category_encoders library with three options:
            + target: replaces each category with the mean of the target variable.
            + leave_one_out: similar to Target Encoding, but excludes the current row's 
            target value when computing the mean.
            + woe: calculates the log-odds ratio of the target within each category.
            best for logistic regression and binary classification problems where features 
            need monotonic transformations.
        """
        if cols is None:  
            cols = self.categorical_cols
        if (self.problem == "regression") | self.df[self.target].isin([0,1,np.nan]).all():
            # choose methodology
            if method == "target":
                encoder = TargetEncoder(handle_missing="return_nan", handle_unknown="return_nan", smoothing=smoothing)
            elif method == "leave_one_out":
                encoder = LeaveOneOutEncoder(handle_missing="return_nan", handle_unknown="return_nan", sigma=smoothing)
            elif method == "woe":
                encoder = WOEEncoder(handle_missing="return_nan", handle_unknown="return_nan", sigma=smoothing)
            # fit target encoding
            encoded_values = encoder.fit_transform(self.df[cols], self.df[self.target].values)
            # rename columns with postfix
            if postfix:
                encoded_values.columns = [col + "_" + postfix for col in cols]
            else:
                self.df.drop(columns=cols, inplace=True)
            # join back encoded values
            self.df = self.df.join(encoded_values)
            print("Target encoding is done on the following columns: total = ", len(cols), "\n", cols) 
            return encoder
        else:
            raise AssertionError("The classification problem is not applicable for target encoding!")

    def dummification(self, cols, value_threshold=1000, drop=False):
        """ get into dummy cols out of categorical features 
        """
        print("Shape before dummification:", self.df.shape)
        # generate dummies
        if cols is None:  
            cols = self.categorical_cols
        for cat_col_i in cols:
            cats = self.df[cat_col_i].value_counts()[lambda x: x > value_threshold].index
            df_cat_col_dummified = pd.get_dummies(pd.Categorical(self.df[cat_col_i], categories=cats)).rename(
                columns = lambda x: standardize_column_name(str(f"{cat_col_i}_{x}").lower()))
            self.df = pd.concat([self.df, df_cat_col_dummified], axis=1, join='inner')
            self.binary_cols = natsorted(self.binary_cols + df_cat_col_dummified.columns.tolist())
        # drop categorical columns and standardize new dummy column names
        if drop:
            self.df.drop(columns=cols, axis=1, inplace=True)
            if cols:
                self.categorical_cols = natsorted(list(set(self.categorical_cols) - set(cols)))
        print("Shape after dummification:", self.df.shape)

    ### TRANSFORMATION FUNCTIONS ###
    def calculate_vs_gaussian_r2(self, series=None):
        """ calculate the r-squared between the given series and a normal distribution.
        """
        if series is None:  
            series = self.df[self.target]
        series = series.dropna().sort_values()
        # generate theoretical normal quantiles
        mean, std = series.mean(), series.std()
        theoretical_quantiles = norm.ppf(np.linspace(0.01, 0.99, len(series)), loc=mean, scale=std)
        return round(r2_score(series, theoretical_quantiles), 4)

    def convert_object_to_category(self):
        """ return a copy of the self.data with category types instead of objects 
        """
        df_copy = self.df.copy()
        for col in self.categorical_cols:
            df_copy[col] = df_copy[col].astype('category')
        return df_copy
    
    def check_best_target_transformation(self, method="r2", metric="R2"):
        """ check best possible target transformation method for closest-to-gaussian
            + r2: check how close the transformed target to normal (gaussian) distribution by checking R2-score
            + ml: let LGBM decides which target gets better predicted
            the metric can be any of the regression metrics below:
            + R2 / MAE / MAPE / RMSE / RMSLE
        """
        norm = self.df[self.target].copy()
        log = np.log1p(self.df[self.target])
        sqrt = np.sqrt(self.df[self.target])
        cbrt = np.cbrt(self.df[self.target])
        box, lam = boxcox(self.df[self.target] + 1)
        transformer = PowerTransformer(method='yeo-johnson')
        yeo = transformer.fit_transform(self.df[self.target].values.reshape(-1, 1)).flatten()
        # set transformer parameters
        self.box_lam = lam
        self.yeo_transformer = transformer
        # dict of different approaches
        series_dict = {'norm': norm, 'log': log, 'sqrt': sqrt, 'cbrt': cbrt, 'box': box, 'yeo': yeo}
        if method == "r2":
            # calculate r-squared for each
            r2_scores = {method: self.calculate_vs_gaussian_r2(pd.Series(data)) for method, data in series_dict.items()}
            print(r2_scores)
            # find best methodology
            best_method = max(r2_scores, key=r2_scores.get)
            best_r2 = r2_scores[best_method]
            print(f"The best transformation is '{best_method}' with an R-squared of {best_r2:.4f}")
        elif method == "ml":
            ml_scores = {}
            df_copy = self.convert_object_to_category()
            # get ml scores for each
            for i in series_dict:
                X_train = df_copy.drop(self.keep_cols + [self.target], axis=1)
                y_train = series_dict[i]
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
                model = LGBMRegressor(random_state=42, verbose=-1)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                scores = regression_metrics(
                    self.inverse_transformation(y_val, i), 
                    self.inverse_transformation(y_pred, i)
                )
                ml_scores[i] = scores
            print(ml_scores)
            # find best methodology
            optimizer = max if metric == "R2" else min
            best_method = optimizer(ml_scores.items(), key=lambda x: x[1][metric])
            print(f"The best transformation is '{best_method[0]}' with {metric} of {best_method[1][metric]:.4f}")

    def inverse_transformation(self, series, method):
        """ do inverse transformation on the given series. methods can be as follows:
             + log:     logarithmic transformation
             + sqrt:    square root
             + cbrt:    cubic root
             + box:     box-cox transformation
             + yeo:     yeo-johnson transformation
        """   
        if method == "norm":
            return series
        elif method == "log":
            return np.expm1(series)
        elif method == "sqrt":
            return np.square(series)
        elif method == "cbrt":
            return np.power(series, 3)
        elif method == "box":
            return inv_boxcox(series, self.box_lam) - 1
        elif method == "yeo":
            return self.yeo_transformer.inverse_transform(series.reshape(-1, 1)).flatten()

    def target_transformation(self, method="log"):
        """ do transformation on the target variable. methods can be as follows:
             + log:     logarithmic transformation
             + sqrt:    square root
             + cbrt:    cubic root
             + box:     box-cox transformation
             + yeo:     yeo-johnson transformation
        """   
        if (self.problem == "regression"): 
            new_target = self.target + f"_{method}"
            if method == "log":
                self.df[new_target] = np.log1p(self.df[self.target])
            elif method == "sqrt":
                self.df[new_target] = np.sqrt(self.df[self.target])
            elif method == "cbrt":
                self.df[new_target] = np.cbrt(self.df[self.target])
            elif method == "box":
                self.df[new_target], lam = boxcox(self.df[self.target] + 1)
                self.box_lam = lam
            elif method == "yeo":
                transformer = PowerTransformer(method='yeo-johnson')
                self.df[new_target] = transformer.fit_transform(self.df[self.target].values.reshape(-1, 1)).flatten()
                self.yeo_transformer = transformer
            self.target = new_target
        else:
            raise AssertionError("Target needs to be a continuous numeric feature!")

    def power_transformation(self, verbose=False):
        """ do power transformation on the numeric columns to handle skewness 
            of the data and make it more closer to normal distribution
        """                          
        power = PowerTransformer(method='yeo-johnson')
        self.df[self.numeric_cols] = pd.DataFrame(data = power.fit_transform(self.df[self.numeric_cols]), columns=self.numeric_cols)   
        print("Power transformation is done: number of columns transformed = ", len(self.numeric_cols))
        if verbose: 
            print(self.numeric_cols) 
        return power 

    def quantile_transformation(self, n_quantiles=1000, output_distribution='uniform', verbose=False):
        """ do quantile transformation on the numeric columns to handle skewness 
            of the data and make it more closer to normal distribution or uniform
            distribution as you like
        """                          
        quantile = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution)
        self.df[self.numeric_cols] = pd.DataFrame(data = quantile.fit_transform(self.df[self.numeric_cols]), columns=self.numeric_cols)   
        print("Quantile transformation is done: number of columns transformed = ", len(self.numeric_cols))
        if verbose: 
            print(self.numeric_cols)     
        return quantile 

    def standardization(self, verbose=False):
        """ do standardization on the numeric columns
        """              
        scaler = StandardScaler()
        self.df[self.numeric_cols] = pd.DataFrame(data = scaler.fit_transform(self.df[self.numeric_cols]), columns=self.numeric_cols)     
        print("Standardization is done: number of columns transformed = ", len(self.numeric_cols))
        if verbose: 
            print(self.numeric_cols) 
        return scaler

    def apply_transformer(self, transformer):
        """ apply transformation given a transformer object 
        """  
        self.df[self.numeric_cols] = pd.DataFrame(data = transformer.transform(self.df[self.numeric_cols]), columns=self.numeric_cols)   
        print("Transformation object is applied: number of columns transformed = ", len(self.numeric_cols), 
              "\n", self.numeric_cols)    
              
    def show_skewness(self):
        """ calculate and print skewness of all transformation columns 
        """
        print("Skewness:")
        print(self.df[self.numeric_cols].skew().sort_values(ascending=False))
    
    ### CORRELATION FUNCTIONS ###
    def resolve_correlation(self, threshold=0.95, drop=False):
        """ remove highly correlated features 
        """
        # create correlation matrix
        corr_matrix = self.df[self.numeric_cols + self.binary_cols].corr().abs().round(2)
        # select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # find features with correlation greater than threshold
        highly_corr = [column for column in upper.columns if any(upper[column] > threshold)]
        print("Highly correlated columns are:", highly_corr)
        # drop features 
        if drop:
            self.numeric_cols = list(set(self.numeric_cols) - set(highly_corr))
            self.binary_cols = list(set(self.binary_cols) - set(highly_corr))
            self.df.drop(highly_corr, axis=1, inplace=True)
            print("Shape becomes:", self.df.shape)
        if self.online_run & (len(highly_corr) > 0):
            self.run.log_list('Highly correlated columns are:', highly_corr)
        return highly_corr

    def heatmap_correlation(self, cols=None, name=""):
        """ show correlation heatmap of numeric features 
        """ 
        if cols is None:  
            if (self.target != "") & (self.problem == "regression"): 
                cols = self.numeric_cols + self.binary_cols + [self.target] 
            else:
                cols = self.numeric_cols + self.binary_cols
        cols = self.df[cols].loc[:, self.df[cols].nunique() > 1].columns.tolist()
        unit_size = np.sqrt(len(cols)*3)
        df_analysis = self.df[cols].corr().round(2)
        sns.set(rc={'figure.figsize':(unit_size*3, unit_size*2.5)})
        sns.heatmap(df_analysis, annot=True, cmap="YlOrRd", linewidths=1)
        plt.show()
        if self.online_run:
            # save figure
            filename=f'./outputs/heatmap_correlation_{name}.png'
            plt.savefig(filename, dpi=600)
            plt.close()

    ### PERFORMANCE METRICS & PLOTS ###
    def how_similar_two_features(self, col1, col2, lower_threshold=0.4, higher_threshold=2.5, res=0.05):
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
        scores = regression_metrics(self.df[col1], self.df[col2], model_name="No Model")
        [print('\t', key,':', val) for key, val in scores.items()]  
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
            self.X = self.df[self.numeric_cols + self.binary_cols + self.categorical_cols].copy()
            self.y = self.df[self.target].copy()
            print("ID columns are removed from data:", len(self.keep_cols))
            print("Data is split into X and y:\n",
                  "\tX:", self.X.shape, "\n",
                  "\ty:", self.y.shape)
        else:
            raise AssertionError("Please set a target feature first!")

    def get_feature_importance(self, model=None, no_features_in_plot=30, name=""):
        """ get feature importance by LGBMClassifier model
        """
        if self.target != "": 
            # split data into X and y
            self.split_x_and_y()
            # define the model
            if model is None: 
                if self.problem == "classification":
                    model = LGBMClassifier(random_state=42, verbose=-1) 
                elif self.problem == "regression":
                    model = LGBMRegressor(random_state=42, verbose=-1) 
            # fit the model
            model.fit(self.X, self.y)
            # get importances
            importances = model.feature_importances_
            importances = np.round(importances / sum(importances), 4)
            sorted_idx = importances.argsort()[::-1]
            df_importance = pd.DataFrame(data={"feature": self.X.columns[sorted_idx].astype(str), "importance": importances[sorted_idx]})
            # summarize feature importance
            plt.figure(figsize=(15, no_features_in_plot/4))
            #plt.barh(df_importance.feature[:no_features_in_plot], df_importance.importance[:no_features_in_plot])
            sns.barplot(x='importance', y='feature', data=df_importance[:no_features_in_plot], orient="h", palette="Set3")
            plt.show()
            if self.online_run:
                # save figure
                filename=f'./outputs/feature_importance_{name}.png'
                plt.savefig(filename, dpi=600)
                plt.close()
            return df_importance
        else:
            raise AssertionError("Please set a target feature first!")
    
    def get_lofo_importance(self, classification_score="weighted", fast=False, recursive_check=False, drop=False):
        """ get feature importance by LOFO
        """
        if self.target != "": 
            # split data into X and y
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
                self.numeric_cols = natsorted(list(set(self.numeric_cols) - set(features_to_drop)))
                self.categorical_cols = natsorted(list(set(self.categorical_cols) - set(features_to_drop)))
                self.binary_cols = natsorted(list(set(self.binary_cols) - set(features_to_drop)))
            return importance_df
        else:
            raise AssertionError("Please set a target feature first!")

    def get_featimp(self):
        """ get feature importances using featimp library 
        """
        if self.df.isna().sum().sum() > 0:
            raise AssertionError("Please first impute the missing values of the dataframe!")
        if self.problem == "regression":
            task = 'reg'
        elif self.problem == "classification":
            task = 'clf_multiable'
            if self.df[self.target].nunique() == 2:
                task = 'clf_binary'
        fi_df = get_feature_importances(data=self.df, num_features=self.numeric_cols, cat_features=self.binary_cols + self.categorical_cols, 
                                        target=self.target, task=task, method='all')
        display_feature_importances(data=fi_df)
        fi_df = fi_df.reset_index().rename(columns={"index": "feature"})
        return fi_df


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

def calc_interquartile(df, column):
    """ detecting extreme values statistically
    """
    q75, q25 = np.nanpercentile(df[column], [75,25])
    intr_qr = q75 - q25 
    upper = q75+(1.5*intr_qr)
    lower = q25-(1.5*intr_qr)
    upper_outliers = df[df[column] > upper]
    lower_outliers = df[df[column] < lower]
    return "{:.2f}".format(lower), "{:.2f}".format(upper), lower_outliers.shape[0]+upper_outliers.shape[0]


### MEMORY REDUCTION ###
def reduce_memory_usage(df):
    """ reduce memory used by the dataframe by converting into 
        more memory friendly data types
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        # numerical
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        # categorical
        elif col_type == object:
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:  # Heuristic: low cardinality
                df[col] = df[col].astype('category')
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print('Memory usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df