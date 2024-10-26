import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


## function that implements the gini impurity ##
def gini_impurity(Data: np.array) -> float:
    """
    Function to compute the Gini Impurity for a set of input data

    Inputs:
        Data => numpy 2D array of predictors & labels, where labels are assumed to be in the last column

    Outputs:
        G => gini impurity value
    """
    # initialize the output
    G = 0
    # iterate through the unique classes
    for c in np.unique(Data[:, -1]):
        # compute p for the current c
        p = Data[Data[:, -1] == c].shape[0] / Data.shape[0]
        # compute term for the current c
        G += p * (1 - p)
    # return gini impurity
    return (G)

## create a classification dataset ##
X,y = make_classification(n_samples=100,
                          n_features=5,
                          n_informative=2,
                          n_classes=2,
                          weights=[0.4,0.6],
                          random_state=42)

## make a random 60%, 40% split of the data ##
names   = ["x1","x2","x3","x4","x5"]
df      = pd.DataFrame(X,columns=names)
df["y"] = y
df1 = df.sample(frac=0.4, random_state=42)
df2 = pd.merge(df,df1,indicator=True,how='outer').query('_merge=="left_only"') \
                                                 .drop('_merge', axis=1)
# compute the gini impurity for df1
print(gini_impurity(df1.values))
print(gini_impurity(df2.values))