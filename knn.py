class knn:

    def __init__(self, neighbors_number):
        self.neighbors_number = neighbors_number
        self.features = None
        self.labels = None


    def fit(self, X, y):
        self.features = X
        self.labels = y

    def predict(self, X):

        dists = X
        # numpy.linalg.norm(a-b)