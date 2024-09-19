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
        self.dataframe:DataFrame=None

    def read_excel_csv_dataframe(self,file_path:str):
        """
        Read/Excel or CSV from the path and
        create the dataframe
        Args:
            file_path:

        Returns:
            Dataframe

        """
        self.dataframe