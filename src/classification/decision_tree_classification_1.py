# Importing the required packages
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib import pyplot as plt
from src.mlops_oops.data_loader import DataLoader
from src.mlops_oops.dataset_details import DatasetDetails


class DecisionTreeClassification:
    """
    Class to implement Decision Tree Classification
    """
    def __init__(self):
        """

        """
        self.X = None
        self.Y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test =  None
        self.y_pred =  None
        self.dataframe=None
        self.dataset_details=None
        self.dataframe_column_length = None
        self.feature_names = []
        self.target_name =  None
        self.class_unique_values = None
        # self.clf_using_gini = None
        # self.clf_using_entropy= None
        self.clf_object = None
        self.metrics_keys = ['Prediction', 'Actual', 'Confusion_Matrix','Accuracy', 'Classification_Report']
        self.metrics:dict = dict.fromkeys(self.metrics_keys)
        self.criteria = None
        self.data_loader=DataLoader()

    def import_data(self,url:str, sep:str):
        """

        Args:
            url:
            sep:

        Returns:
            DataFrame

        """
        self.dataframe=self.data_loader.read_url_dataframe(url=url, sep=sep,header=None)
        print(f'DataFrame Summary:\n{self.dataframe.head()}')
        self.dataframe_column_length = len(self.dataframe.columns)
        print(f'Length of the DataFrame Columns:{len(self.dataframe.columns)}')
        # Get the Feature Column Names
        self.get_feature_names_by_column_index()
        # print(f'Feature Columns:{self.feature_columns}')

        # Get the Target Column Name
        self.get_target_column_by_index(0)
        print(f'Target Column Name:{self.target_name}')

        # Get the Unique Class Values
        self.get_class_values_unique()
        print(f'Unique Class Values:{self.class_unique_values}')
        self.visualize_data()
        return self.dataframe

    def get_feature_names_by_column_index(self):
        """

        Returns:

        """
        # First Column(index 0 is the target column
        column_indices_to_get = []
        for column_index in range(self.dataframe_column_length):
            if column_index!=0:
                column_indices_to_get.append(column_index)
                print(f'Feature Column Name:{self.dataframe.columns[column_index]}')
                self.feature_names.append(self.dataframe.columns[column_index])
        print(f'Column Indices:{column_indices_to_get}')
        # self.feature_columns = self.dataframe.columns[column_indices_to_get]
        print(f'Feature Column Names:{self.feature_names}')

    def get_target_column_by_index(self, target_column_index:int)->str:
        """

        Args:
            target_column_index:

        Returns:
            target_column
        Raises


        """
        self.target_name=self.dataframe.columns[target_column_index]
        return self.target_name

    def get_class_values_unique(self):
        """

        Returns:

        """
        self.class_unique_values = self.dataframe[self.target_name].unique()

    def visualize_data(self):
        """

        Args:


        Returns:

        """
        self.dataset_details=DatasetDetails()
        self.dataset_details.show_dataset_details(self.dataframe)

    def split_data(self):
        """
        Function to split the dataset into features and target variables
        Returns:

        """
        # Feature Variables
        self.X = self.dataframe.values[:, 1:5]

        # Target Variable
        self.Y = self.dataframe.values[:, 0]

        # Split the dataset into train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y,
                                                            test_size=0.3, random_state=100)

    def train_using_gini(self):
        """

        Returns:

        """
        # Split the Data into Train and Test Data Set
        self.split_data()

        # Set the Split Criteria as Gini Impurity
        self.criteria = 'gini'

        # Creating the Classifier Model
        self.clf_object = DecisionTreeClassifier(criterion = self.criteria,
                                            random_state = 100, max_depth = 3,
                                            min_samples_leaf = 5)
        # Performing Training
        self.clf_object.fit(self.X_train, self.y_train)

    def predict_using_gini(self):
        """

        Returns:

        """
        self.train_using_gini()
        self.y_pred = self.clf_object.predict(self.X_test)
        # print(f'Prediction Using Gini:{self.y_pred}')
        self.metrics = self.view_metrics()
        #print(self.metrics["Prediction"])



    def train_using_entropy(self):
        """

        Returns:

        """
        # Split the Data into Train and Test Data Set
        self.split_data()

        # Set the Split Criteria as Gini Impurity
        self.criteria = 'entropy'

        # Creating the Classifier Model
        self.clf_object = DecisionTreeClassifier(criterion = self.criteria,
                                            random_state = 100, max_depth = 3,
                                            min_samples_leaf = 5)
        # Performing Training
        self.clf_object.fit(self.X_train, self.y_train)

    def predict_using_entropy(self):
        """

        Returns:

        """
        self.train_using_entropy()
        self.y_pred = self.clf_object.predict(self.X_test)
        # print(f'Prediction Using Entropy:{self.y_pred}')
        self.metrics = self.view_metrics()
        # return self.y_pred

    def view_metrics(self) ->dict:
        """

        Returns:
            Metrics: Dictionary of Classification Metrics

        """
        self.metrics["Prediction"] = self.y_pred
        print(f'Prediction:\n{self.metrics["Prediction"]}')
        self.metrics["Actual"] = self.y_test
        self.metrics['Confusion_Matrix'] = confusion_matrix(self.y_test, self.y_pred)
        self.metrics['Accuracy'] = accuracy_score(self.y_test, self.y_pred)
        self.metrics['Classification_Report'] = classification_report(y_true=self.y_test, y_pred=self.y_pred, zero_division=0.0)
        print(f'Report:\n{self.metrics['Classification_Report']}')
        for key, value in self.metrics.items():
            print(f'{key}:\n{value}')
        # print(self.metrics.values())
        self.plot_decision_tree()
        self.save_model()
        return self.metrics

    def plot_decision_tree(self):
        """

        Args:


        Returns:

        """
        plt.figure(figsize=(15, 10))
        plot_tree(self.clf_object, filled=True, feature_names=self.feature_names, class_names=self.class_unique_values, rounded=True)
        # Save the plot as image
        plot_image_Name = self.criteria + '_decision_tree.png'
        plt.savefig(plot_image_Name)
        plt.show()

    def save_model(self):
        """
        Save the model in the current directory
        Returns:

        """
        file_name = 'decision-tree-model-' + self.criteria + '.pkl'
        with open(file_name, "wb") as fout:
            pickle.dump(file_name, fout)




if __name__=="__main__":
        file_url_main:str='https://archive.ics.uci.edu/ml/machine-learning-'
        file_url_sub:str='databases/balance-scale/balance-scale.data'
        file_url=file_url_main+file_url_sub
        data_sep=","
        decision_tree_classifier=DecisionTreeClassification()
        decision_tree_classifier.import_data(file_url,data_sep)
        decision_tree_classifier.predict_using_gini()
        decision_tree_classifier.predict_using_entropy()