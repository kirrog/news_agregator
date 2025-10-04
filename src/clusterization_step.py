from typing import List

from numpy import ndarray
from sklearn.cluster import DBSCAN


class Clasterizator:
    def __init__(self):
        self.clustering = DBSCAN(eps=3, min_samples=2)

    def fit_predict(self, embeddings: List[ndarray]):
        self.data = embeddings
        return self.clustering.fit_predict(embeddings)
