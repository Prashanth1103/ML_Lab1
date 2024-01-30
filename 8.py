import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
data = {
 'VAR1': [1.713, 0.180, 0.353, 0.940, 1.486, 1.266, 1.540, 0.459, 0.773],
 'VAR2': [1.586, 1.786, 1.240, 1.566, 0.759, 1.106, 0.419, 1.799, 0.186],
 'CLASS': [0, 1, 1, 0, 1, 0, 1, 1, 1]
}
df = pd.DataFrame(data)
X = df[['VAR1', 'VAR2']]
kmeans_model = KMeans(n_clusters=3, random_state=42)
kmeans_model.fit(X)
new_case = np.array([[0.906, 0.606]])
predicted_cluster = kmeans_model.predict(new_case)
print("Predicted cluster for the new case:", predicted_cluster[0])
