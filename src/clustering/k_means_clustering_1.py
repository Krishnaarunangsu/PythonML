from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

# plt.scatter(x, y)
#plt.show()

data = list(zip(x, y))
print(f'Dataset for K-Means Algorithm:{data}')
inertias=[]

# There are 10 data points
for i in range(1, 11):
    kmeans=KMeans(n_clusters=i)
    print(f'KMeans[{i}]={kmeans}')
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

print(f'Inertia is:{inertias}')

# plt.plot(range(1,11), inertias, marker='o')
# plt.title('Elbow method')
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')

# k=2 is optimized
kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

plt.scatter(x, y, c=kmeans.labels_)
plt.show()
