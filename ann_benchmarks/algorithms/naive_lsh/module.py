import np_sims.lsh  # My LSH implementation, see dockerfile for installation

from ..base.module import BaseANN


# This is my adapter from ANNBenchmarks to my LSH implementation
class NaiveLSH(BaseANN):

    # Optionally in config.yml I can specify index_params
    # the metric might be euclidean, angular, etc
    # Here LSH models cosine similarity, so angular is the only thing we've specified
    # in config.yml to run this on
    def __init__(self, metric, index_params):
        # These are params about how to build the index passed from config.yml
        #     run_groups:
        #       test:
        #         arg_groups: [{num_projections: 128, num_projections: 512, num_projections: 1024}]

        self.num_projections = index_params['num_projections']
        super().__init__()

    # "Fit" or index the data, whatever your parlance you prefer :)
    def fit(self, X, y=None):
        vectors = X
        self.projs = np_sims.lsh.create_projections(num_projections=self.num_projections, dims=vectors.shape[1])
        self.hashes = np_sims.lsh.index(vectors, self.projs)
        return self

    # Given query vector v, return n nearest neighbors
    def query(self, v, n):
        return np_sims.lsh.query(v, self.hashes, self.projs)[0]

    # Return the name of the algorithm
    def __str__(self):
        return 'NaiveLSH'
