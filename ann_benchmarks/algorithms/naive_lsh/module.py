import similarities.lsh

from ..base.module import BaseANN


class NaiveLSH(BaseANN):

    def __init__(self, metric, index_params):
        super().__init__(metric, index_params)

    def fit(self, X, y=None):
        self.lsh = similarities.lsh.LSH()
        vectors = X
        hashes, projs = similarities.lsh.load_or_build_index(vectors)
        return self

    def query(self, v, n):
        return self.lsh.query(v, n)

    def __str__(self):
        return 'NaiveLSH'
