class DatasetDetails:
    """
    Dataset Details
    """
    def __init__(self):
        """
        Initialization
        """
        self.dataset=None

    def show_dataset_details(self,dataset):
        """

        Args:
            dataset:

        Returns:

        """
        self.dataset = dataset
        print(f'Dataset Shape:{self.dataset.shape}')
        print(f'Dataset Summary:{self.dataset.head()}')
        print(f'Dataset Information:{self.dataset.info()}')
        print(f'Dataset Description:{self.dataset.describe()}')