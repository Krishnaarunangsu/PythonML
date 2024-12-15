import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from src.mlops_oops.data_loader import DataLoader


# Explore and Visualize the data

class KMeans:
    """ Class for K-Means Clustering"""

    def __init__(self):
        """
        Initialisation
        """
        self.home_data = None

    def get_data(self):
        """

        Returns:

        """
        data_loader = DataLoader()
        url = "C:\\Arunangsu\\PythonML\\data\\housing_data\\housing.csv"
        #self.home_data=data_loader.read_url_dataframe(url)
        self.home_data = pd.read_csv(url, usecols=['longitude', 'latitude', 'housing_median_age'])
        print(self.home_data.head())
        print(self.home_data.describe())


    def plot_data(self):
        """

        Returns:

        """
        sns.scatterplot(data=self.home_data, x='longitude', y='latitude', hue='housing_median_age')
        plt.show()



if __name__ == "__main__":
    k_means = KMeans()
    k_means.get_data()
    k_means.plot_data()
