# Gini Impurity Calculation for Random Split of Data
import numpy as np
import pandas as pd
import  seaborn as sns
from pandas.core.common import random_state
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt



class GiniImpurityCalculation:
    """
    Class for Gini Impurity Calculation
    """
    def __init__(self):
        """

        """
        self.X=None
        self.y=None
        self.dataframe=None
        self.dataframe_corr=None
        self.names=['x1', 'x2', 'x3', 'x4','x5']

    def calculate_gini_impurity_random_split(self, data:np.array)->float:
        """

        Args:
            data: numpy 2D array of predictors & labels, where labels are
                  assumed to be in the last column

        Returns:
            gini_impurity: gini impurity value

        """
        # initialize the output
        G = 0
        # iterate through the unique classes
        # for c in np.unique(data[:, :-1]):
        for c in np.unique(data[:, -1]):
            # compute p for the current c
            print(c)
            p = data[data[:, -1] == c].shape[0] / data.shape[0]
            # compute term for the current c
            G += p * (1 - p)
        # return gini impurity
        return (G)

    def create_classification_dataset(self, n_samples:int, n_features:int,n_informative:int, n_classes:int,
                                      weights, random_state:int)->DataFrame:
        """
        Create a Classification Dataset
        Args:
            n_samples:
            n_features:
            n_informative:
            n_classes:
            weights:
            random_state:

        Returns:
            DataFrame

        """
        self.X, self.y=make_classification(
                                           n_samples=n_samples,
                                           n_features=n_features,
                                           n_informative=n_informative,
                                           weights=weights,
                                           random_state=random_state
        )
        # Create the Dataframe with Features, Predictor and labels
        self.dataframe = pd.DataFrame(self.X, columns=self.names)
        self.dataframe["y"] = self.y

        # Visualize the classification data
        self.visualize_date()
        return self.dataframe

    def visualize_date(self):
        """

        Returns:

        """
        ## Plot the distribution of the Predictor features
        plt.boxplot(self.X)
        plt.xlabel('features')
        plt.ylabel('values')
        plt.title('Distribution in Features')
        plt.xticks([1, 2, 3, 4, 5],['x1', 'x2', 'x3', 'x4','x5'])
        plt.show()

        ## Create a HeatMap of the Features and labels
        self.dataframe_corr = self.dataframe.corr()
        sns.heatmap(self.dataframe_corr)
        plt.title('Heat Map of Correlation Values')
        plt.show()

if __name__ == "__main__":
    gini_impurity_calc= GiniImpurityCalculation()
    dataframe_ch:DataFrame = gini_impurity_calc.create_classification_dataset(100, 5, 2, 2,
                                                     [0.4, 0.6], 42)
    print(f'Original DataFrame:\n{dataframe_ch.head()}')
    print(f'Original DataFrame Shape:\n{dataframe_ch.shape}')
    df_1=dataframe_ch.sample(frac=0.4, random_state=42)
    print(f'40 percent of the Original DataFrame:\n{df_1.head()}')
    print(f'40 percent of the DataFrame Shape:\n{df_1.shape}')
    df_2 = pd.merge(dataframe_ch, df_1, indicator=True, how='outer').query('_merge=="left_only"') \
        .drop('_merge', axis=1)


    print(f'60 percent of the Original DataFrame:\n{df_2.head()}')
    print(f'60 percent of the DataFrame Shape:\n{df_2.shape}')
    df_1_gini = gini_impurity_calc.calculate_gini_impurity_random_split(df_1.to_numpy())
    print(df_1_gini)
    df_2_gini = gini_impurity_calc.calculate_gini_impurity_random_split(df_2.to_numpy())
    print(df_2_gini)
    #print(gini_impurity_calc.calculate_gini_impurity_random_split(df2.values))

    # weighted average to measure net 'quality' of the split
    weighted_gini = 0.4 * df_1_gini + 0.6 * df_2_gini
    print(weighted_gini)
    # Make a split based upon the median of x4 ##
    median_df=dataframe_ch.x4.median()
    df_1_informed = dataframe_ch[dataframe_ch.x4> median_df].copy()
    df_2_informed = dataframe_ch[dataframe_ch.x4 <= median_df].copy()

    # Compute the Gini Impurity with Informed Split of Data
    df_1_informed_gini = gini_impurity_calc.calculate_gini_impurity_random_split(df_1_informed.to_numpy())
    print(df_1_informed_gini)
    df_2_informed_gini = gini_impurity_calc.calculate_gini_impurity_random_split(df_2_informed.to_numpy())
    print(df_2_informed_gini)

    print(len(dataframe_ch))
    print(dataframe_ch.shape[0])
    print(df_1_informed.shape[0])
    print(df_2_informed.shape[0])

    df_1_informed_ratio = df_1_informed.shape[0]/dataframe_ch.shape[0]
    df_2_informed_ratio = df_2_informed.shape[0]/dataframe_ch.shape[0]
    weighted_gini_informed = df_1_informed_ratio*df_1_informed_gini + df_2_informed_ratio*df_2_informed_gini
    print(weighted_gini_informed)

# https://insidelearningmachines.com/gini_impurity/


