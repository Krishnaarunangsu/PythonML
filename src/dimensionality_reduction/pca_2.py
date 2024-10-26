import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# Fetch the data samples
breast = load_breast_cancer()
print(breast)
breast_data =  breast.data
print(f'Breast Data:\n{breast_data}')
breast_features = breast["feature_names"]
print(f'Breast Data Feature Names:\n{breast_features}')

# Shape of the features
print(f'Shape of the data:\n{breast_data.shape}')

# Target
breast_labels = breast.target
print(f'Target:\n{breast_labels}')

# Shape of the features
print(f'Shape of the Target:\n{breast_labels.shape}')

# Target Columns
target_columns = breast["target_names"]
print(f'Target Columns:\n{target_columns}')
labels = np.reshape(breast_labels,(569,1))
final_breast_data = np.concatenate([breast_data,labels],axis=1)

print(f'Breast Data Shape after Reshaping:\n{final_breast_data.shape}')
breast_dataset = pd.DataFrame(final_breast_data)
print(f'Breast Data Feature Names:\n{breast_features}')
# print(f'Breast Data Feature Names:\n{len(breast_feature_names)}')
features_labels = np.append(breast_features,'label')
breast_dataset.columns = features_labels
print(breast_dataset.head())

breast_dataset['label'].replace(0, 'Benign',inplace=True)
breast_dataset['label'].replace(1, 'Malignant',inplace=True)

print(breast_dataset.tail())


x = breast_dataset.loc[:, breast_features].values
print(x)
x = StandardScaler().fit_transform(x) # normalizing the features

print(x.shape)

#Check whether the normalized data has a mean of zero and a standard deviation of one.
print(f'Current Mean:\n{np.mean(x),}')
print(f'Current Standard Deviation:\n{np.std(x)}')

# Let's convert the normalized features into a tabular format with the help of DataFrame.
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
print(f'Feature Columns:\n{feat_cols}')
print(f'Total no of Feature Columns:\n{len(feat_cols)}')

normalised_breast = pd.DataFrame(x,columns=feat_cols)
print(f'Summary of Breast Data Features:\n{normalised_breast.tail()}')

# Now comes the critical part, the next few lines of code will be projecting
# the thirty-dimensional Breast Cancer data to two-dimensional principal components.
# You will use the sklearn library to import the PCA module, and in the PCA method,
# you will pass the number of components (n_components=2) and finally call
# fit_transform on the aggregate data. Here, several components
# represent the lower dimension in which you will project your higher dimension data.


pca_breast = PCA(n_components=2)
principalComponents_breast = pca_breast.fit_transform(x)

# create a DataFrame that will have the principal component values
# for all 569 samples.
principal_breast_Df = pd.DataFrame(data = principalComponents_breast
             , columns = ['principal component 1', 'principal component 2'])

print(f'Tail of the transformed dataframe:\n{principal_breast_Df.tail()}')
print(f'Shape:\n{principal_breast_Df.shape}')

principal_breast_Df_with_target = principal_breast_Df
principal_breast_Df_with_target.insert(2, 'Target', breast_dataset['label'], True)
print(f'Tail of the transformed dataframe:\n{principal_breast_Df_with_target.tail()}')

# Once you have the principal components, you can find the explained_variance_ratio.
# It will provide you with the amount of information or variance
# each principal component holds after projecting the data
# to a lower dimensional subspace.

print(f'Explained variability per principal component:{pca_breast.explained_variance_ratio_}')

plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=20)
targets = ['Benign', 'Malignant']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = breast_dataset['label'] == target
    plt.scatter(principal_breast_Df.loc[indicesToKeep, 'principal component 1']
               , principal_breast_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})
plt.show()

