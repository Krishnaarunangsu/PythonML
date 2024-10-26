# Data Loading / ingestion class
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame


class DataLoader:
    """
    Data Loader/ Data Ingestion class
    """
    def __init__(self):
        """
        Initialization
        """
        self.dataframe:DataFrame= None

    def read_excel_csv_dataframe(self,file_path:str)->DataFrame:
        """
        Read/Excel or CSV from the path and
        create the dataframe
        Args:
            file_path:

        Returns:
            Dataframe

        """
        self.dataframe= pd.read_csv(file_path)
        return self.dataframe

    def read_json_dataframe(self, file_path:str)->DataFrame:
        """

        Args:
            file_path:

        Returns:
            Dataframe
        """
        self.dataframe = pd.read_csv(file_path)
        return self.dataframe

    def read_url_dataframe(self,url:str,sep:str,header):
        """
        Read the url to get the data
        and build the dataframe
        Args:
            url:
            sep:
            header:

        Returns:
            Dataframe

        """
        self.dataframe=pd.read_csv(url, sep=sep, header=header)
        return self.dataframe