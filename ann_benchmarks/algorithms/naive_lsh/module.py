import similarities.lsh  # My LSH implementation, see dockerfile for installation

from ..base.module import BaseANN


# This is my adapter from ANNBenchmarks to my LSH implementation
class NaiveLSH(BaseANN):

    # Optionally in config.yml I can specify index_params
    # the metric might be euclidean, angular, etc
    # Here LSH models cosine similarity, so angular is the only thing we've specified
    # in config.yml to run this on
    def __init__(self, metric, index_params):
        super().__init__()

    # "Fit" or index the data, whatever your parlance you prefer :)
    def fit(self, X, y=None):
        vectors = X
        hashes, projs = similarities.lsh.load_or_build_index(vectors)
        return self

    # Given query vector v, return n nearest neighbors
    def query(self, v, n):
        return self.lsh.query(v, n)

    # Return the name of the algorithm
    def __str__(self):
        return 'NaiveLSH'
