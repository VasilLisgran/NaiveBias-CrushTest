import numpy as np
import matplotlib.pyplot as plt

class QuadraticDiscriminantClassificator():
    def fit(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)

        # All unique classes
        self.classes = np.unique(Y)
        self.mean = {}
        self.cov = {}
        self.priors = {}
        for c in self.classes:
            X_c = X[Y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.cov[c] = np.cov(X_c, rowvar=False)
            self.priors[c] = len(X_c) / len(X)
        
        return self.mean, self.cov, self.priors

    def predict(self, X):
        X = np.array(X)
        scores = np.zeros((len(X), len(self.classes)))

        for idx, c in enumerate(self.classes):
            mean_vector = self.mean[c]
            aprior = self.priors[c]
            cov_matrix = self.cov[c]

            diff = X - mean_vector
            first_num = -0.5 * np.log(np.linalg.det(Cov))
            second_num = -0.5 * np.sum((diff @ np.linalg.inv(cov_matrix) * diff), axis=1)
            third_num = np.log(aprior)

            QDA = first_num + second_num + third_num
            scores[:, idx] = QDA
        predicted = np.argmax(scores, axes=1)
        return self.classes[predicted]

class GaussianClassificator():
    def fit(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)

        # All unique classes
        self.classes = np.unique(Y)

        self.mean = {}
        self.var = {}

        for c in self.classes:
            X_c = X[Y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)

        return self.mean, self.var

    def predict(self, X):
        X = np.array(X)
        likelihoods = []

        for c in self.classes:
            coef = 1/np.sqrt(2*np.pi*(self.var[c]))
            exp = np.exp(- (X - self.mean[c])**2 / (2*(self.var[c])))
            # Probabilities of x1, x2
            P_c = coef * exp

            # it works because of x1 and x2 independence 
            total_likelihood = np.prod(P_c, axis=1)
            likelihoods.append(total_likelihood)

        likelihoods = np.array(likelihoods).T

        return np.argmax(likelihoods, axis=1)

def train(X, Y):
    split_index = int(len(X)*0.8)
    train_X, test_X = np.split(X, [split_index])

    train_Y, test_Y = np.split(Y, [split_index])
    return train_X, test_X, train_Y, test_Y

centers = range (0, 10)
correlations = np.linspace(0.0, 0.9, 10)
dimensions = [2, 4, 6, 8]
res_matrix = np.zeros((len(centers), len(correlations)))

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for d, dimension in enumerate(dimensions): 
    for i, center in enumerate(centers):
        for j, correlation in enumerate(correlations):
            
            first = center/10 * np.ones(dimension)
            second = -first

            Var = 1
            Cov = np.full((dimension, dimension), correlation)
            np.fill_diagonal(Cov, 1.0)

            rng = np.random.Generator(np.random.PCG64(12345))
            samples_first = rng.multivariate_normal(mean=first, cov=Cov, size=1000)
            samples_second = rng.multivariate_normal(mean=second, cov=Cov, size=1000)

            x_first = samples_first[:, 0]
            y_first = samples_first[:, 1]

            x_second = samples_second[:, 0]
            y_second = samples_second[:, 1]

            X = np.vstack([samples_first, samples_second])

            # TARGET
            Y = np.concatenate([
                np.zeros(len(samples_first)),
                np.ones(len(samples_first))
                ], axis=0)

            # make random indexes and apply it to X and Y
            # It it necessary or first 80% of Data will be 1600 zeros and 20% of ones
            indices = rng.permutation(len(X))
            X = X[indices]
            Y = Y[indices]

            train_X, test_X, train_Y, test_Y = train(X, Y)
            classificator = GaussianClassificator()
            classificator.fit(train_X, train_Y)
            predictions = classificator.predict(test_X)

            accuracy = np.mean(predictions == test_Y)
            res_matrix[i, j] = accuracy

    ax = axes[d]
    im = ax.imshow(res_matrix, cmap='viridis', origin='lower', aspect='auto')
    ax.set_title(f'Dimension D = {dimension}')
    ax.set_xlabel(r'Correlation ($\rho$ index)')
    ax.set_ylabel('Center distance')
    fig.colorbar(im, ax=ax)    

plt.tight_layout()
plt.show()
