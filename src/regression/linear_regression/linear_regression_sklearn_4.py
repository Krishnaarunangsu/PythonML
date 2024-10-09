import numpy as np
import pandas as pd
import pickle
import json

from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from src.mlops_oops.data_loader import DataLoader
from src.mlops_oops.dataset_details import DatasetDetails

class SimpleLinearRegression:
    """
    Class for Linear Regression
    """
    def __init__(self):
        """

        """
        self.data_train = None
        self.data_test = None

    def load_dataset(self, file_path_details:json):
        """

        Args:
            file_path_details:


        Returns:

        """
        try:
            if file_path_details["train_data"] is not None and file_path_details["test_data"] is not None:
                data_loader = DataLoader()
                self.data_train=data_loader.read_excel_csv_dataframe(file_path_details["train_data"])
                self.data_test = data_loader.read_excel_csv_dataframe(file_path_details["test_data"])

                # Show the Train Data and Test Data
                if self.data_train is not None and self.data_test is not None:
                    #print(self.data_train)
                    dataset_details=DatasetDetails()
                    dataset_details.show_dataset_details(self.data_train)
                    dataset_details.show_dataset_details(self.data_test)
        except KeyError as ke:
            print('Key Not Found in Employee Dictionary:', ke)

if __name__=="__main__":
     simple_linear_regression=SimpleLinearRegression()
     data_path_dir_initial = "C:\\Arunangsu\\PythonML\\data\\"
     train_data_path = f'{data_path_dir_initial}'+"train_gender_submission.csv"
     test_data_path = f'{data_path_dir_initial}'+"test_gender_submission.csv"
     # train_data_path = "C:\\Arunangsu\\PythonML\\data\\train_gender_submission.csv"
     # test_data_path = "C:\\Arunangsu\\PythonML\\data\\test_gender_submission.csv"
     file_details_json:json=\
         {
         "train_data": train_data_path,
         "test_data": test_data_path

     }
     simple_linear_regression.load_dataset(file_details_json)







# Reading data from csv file
