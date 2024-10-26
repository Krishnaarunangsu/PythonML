# Dimensionality Reduction through Principal Component Analysis
# importing required libraries
import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt
from pandas.core.common import random_state
from sklearn.linear_model import LogisticRegression

from src.mlops_oops.data_loader import DataLoader
from src.mlops_oops.dataset_details import DatasetDetails
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

class PrincipalComponentAnalysis:
    """
    Class for Feature Reduction
    """
    def __init__(self):
        self.dataset = None
        self.X = None
        self.X_train = None
        self.X_test = None
        self.y = None
        self.y_train = None
        self.y_test = None
        self.explained_variance = None
        self.regressor =  None
        self.y_predict = None
        self.confusion_matrix = None

    def load_dataset(self, file_path_details: json):
        """

        Args:
            file_path_details:

        Returns:

        """
        try:
            if file_path_details["data_loc"] is not None:
                data_loader = DataLoader()
                self.dataset = data_loader.read_excel_csv_dataframe(file_path_details["data_loc"])

                # Show the Train Data and Test Data
                if self.dataset is not None:
                    # print(self.data_train)
                    dataset_details = DatasetDetails()
                    dataset_details.show_dataset_details(self.dataset)
        except KeyError as ke:
            (
                print('Key Not Found in Wine Dictionary:', ke))

    def divide_dataset_x_y(self):
        """
        Divide the Dataset into X Columns and Y column
        Returns:

        """
        self.X = self.dataset.iloc[:, 0:13].values
        print(f'Predictors Columns:\n{self.X}')
        self.y = self.dataset.iloc[:, 13].values
        print(f'Target Column:\n{self.y}')

    def train_test_split(self):
        """
        Divide the X and y into the
        Training Set and Testing Set
        Returns:

        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=0)
        print(f'Train set for X Columns:\n{self.X_train}')
        print(f'Test set for X Columns:\n{self.X_test}')
        print(f'Train set for y Column:\n{self.y_train}')
        print(f'Test set for y Columns:\n{self.y_test}')


    def preprocess_data(self):
        """
        Preprocess the Predictor Data with StandardScaler
        Returns:

        """
        sc=StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        print(f'Scaled Train Data of X Columns:{self.X_train}')
        self.X_test = sc.fit_transform(self.X_test)
        print(f'Scaled Test Data of X Columns:{self.X_test}')


    def apply_pca(self):
        """
        Apply PCA of Train Data and Test Data of X
        Returns:

        """
        pca = PCA()
        self.X_train = pca.fit_transform(self.X_train)
        print(f'After applying PCA Value of Train set of X:\n{self.X_train}')
        self.X_test = pca.fit_transform(self.X_test)
        print(f'After applying PCA Value of Test set of X:\n{self.X_test}')
        self.explained_variance = pca.explained_variance_ratio_
        print(f'Explained Variance:\n{self.explained_variance}')

    def build_model_logistic_regression(self):
        """
        Fitting Logistic Regression To the Training Set
        Returns:

        """
        self.regressor = (LogisticRegression(random_state=0)
                          .fit(self.X_train, self.y_train))

        #self.regressor.fit(self.X_train, self.y_train)

    def predict_result(self):
        """
        Predicting the test set result using the
        predict function under LogisticRegression
        Returns:

        """
        self.y_predict = self.regressor.predict(self.X_test)
        print(f'The Predicted Set:\n{self.y_predict}')
        print(f'The Actual Set:\n{self.y_test}')

    def build_confusion_matrix(self):
        """
        Making Confusion Matrix between
        the test set of y and the predicted value
        Returns:

        """
        self.confusion_matrix = confusion_matrix(self.y_test, self.y_predict)
        print(f'Confusion Matrix:\n{self.confusion_matrix}')

    def plot_training_result(self):
        """

        Returns:

        """
        X_set, y_set = self.X_train, self.y_train
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                                       stop=X_set[:, 0].max() + 1, step=0.01),
                             np.arange(start=X_set[:, 12].min() - 1,
                                       stop=X_set[:, 12].max() + 1, step=0.01))

        print(f'X1:\n{X1}')
        print(f'X2:\n{X2}')
        plt.scatter(X1,X2)
        # set labels and title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('meshgrid() to create grid')
        y_predict_2 = self.regressor.predict(np.array(X1.ravel()))

        #plt.contourf(X1, X2, self.regressor.predict(np.array([X1.ravel(),
        #                                                   X2.ravel()]).T).reshape(X1.shape), alpha=0.75,
        #              cmap=ListedColormap(('yellow', 'white', 'aquamarine')))
        #
        # plt.xlim(X1.min(), X1.max())
        # plt.ylim(X2.min(), X2.max())
        #
        # for i, j in enumerate(np.unique(y_set)):
        #     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
        #                 color=ListedColormap(('red', 'green', 'blue'))(i), label=j)
        #
        # plt.title('Logistic Regression (Training set)')
        # plt.xlabel('PC1')  # for Xlabel
        # plt.ylabel('PC2')  # for Ylabel
        # plt.legend()  # to show legend
        #
        # # show scatter plot
        # plt.show()


if __name__ == "__main__":
    principal_component_analysis = PrincipalComponentAnalysis()
    data_path_dir_initial = "C:\\Arunangsu\\PythonML\\data\\"
    data_path = f'{data_path_dir_initial}' + "Wine.csv"

    file_details_json: json = \
         {
             "data_loc": data_path,

         }
    principal_component_analysis.load_dataset(file_details_json)
    principal_component_analysis.divide_dataset_x_y()
    principal_component_analysis.train_test_split()
    principal_component_analysis.preprocess_data()
    principal_component_analysis.apply_pca()
    principal_component_analysis.build_model_logistic_regression()
    principal_component_analysis.predict_result()
    principal_component_analysis.build_confusion_matrix()
    principal_component_analysis.plot_training_result()