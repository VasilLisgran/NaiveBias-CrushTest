import numpy as np
import matplotlib.pyplot as plt

first = [-1, -1]
second = [1, 1]

Var = 1
Cov = [[1, 0], [0, 1]]

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

class GaussianClassificator():
    def fit(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)

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

train_X, test_X, train_Y, test_Y = train(X, Y)
classificator = GaussianClassificator()
classificator.fit(train_X, train_Y)
predictions = classificator.predict(test_X)

accuracy = np.mean(predictions == test_Y)
print(accuracy)



fig, axes = plt.subplots(2)
axes[0].scatter(x_first, y_first)
axes[0].set_aspect('equal', adjustable='box')
axes[1].scatter(x_second, y_second)
axes[1].set_aspect('equal', adjustable='box')
plt.show()