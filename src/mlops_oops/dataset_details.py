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
        print(f'Dataset Summary:\n{self.dataset.head()}')
        print('Dataset Information')
        print('************************************')
        self.dataset.info()
        print(f'Dataset Description')
        print('************************************')
        print(self.dataset.describe())