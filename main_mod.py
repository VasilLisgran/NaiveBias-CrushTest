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
            first_num = -0.5 * np.log(np.linalg.det(cov_matrix))
            second_num = -0.5 * np.sum((diff @ np.linalg.inv(cov_matrix) * diff), axis=1)
            third_num = np.log(aprior)

            QDA = first_num + second_num + third_num
            scores[:, idx] = QDA
        predicted = np.argmax(scores, axis=1)
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

correlations = np.linspace(0.0, 0.9, 10)
dimensions = [2, 4, 6, 8]
data = [50, 100, 200, 400, 800, 1600, 3200]    # points per class
n_seeds = 20
tag = 'cov_diff'

acc_nb  = np.full((len(data), n_seeds, len(dimensions), len(correlations)), np.nan)
acc_qda = np.full((len(data), n_seeds, len(dimensions), len(correlations)), np.nan)

for idx, inf in enumerate(data):
    for d, dimension in enumerate(dimensions):
        mean = np.zeros(dimension)
        prop_nums = np.arange(dimension)
        distance = np.abs(np.subtract.outer(prop_nums, prop_nums))

        for j, correlation in enumerate(correlations):
            Cov_first = correlation ** distance          # Toeplitz, +rho
            Cov_second = (-correlation) ** distance      # Toeplitz, -rho (alternating signs)

            for s in range(n_seeds):
                rng = np.random.Generator(np.random.PCG64(s))
                samples_first = rng.multivariate_normal(mean=mean, cov=Cov_first, size=inf)
                samples_second = rng.multivariate_normal(mean=mean, cov=Cov_second, size=inf)

                X = np.vstack([samples_first, samples_second])
                Y = np.concatenate([np.zeros(inf), np.ones(inf)])

                # shuffle, otherwise the first 80% would be one class only
                indices = rng.permutation(len(X))
                X = X[indices]
                Y = Y[indices]

                train_X, test_X, train_Y, test_Y = train(X, Y)

                classificator1 = GaussianClassificator()
                classificator1.fit(train_X, train_Y)
                acc_nb[idx, s, d, j] = np.mean(classificator1.predict(test_X) == test_Y)

                classificator2 = QuadraticDiscriminantClassificator()
                classificator2.fit(train_X, train_Y)
                acc_qda[idx, s, d, j] = np.mean(classificator2.predict(test_X) == test_Y)

# mean and standard error over seeds -> shape (n, D, rho)
mean_nb  = acc_nb.mean(axis=1)
se_nb    = acc_nb.std(axis=1, ddof=1) / np.sqrt(n_seeds)
mean_qda = acc_qda.mean(axis=1)
se_qda   = acc_qda.std(axis=1, ddof=1) / np.sqrt(n_seeds)

np.save(f'mean_nb_{tag}.npy', mean_nb)
np.save(f'se_nb_{tag}.npy', se_nb)
np.save(f'mean_qda_{tag}.npy', mean_qda)
np.save(f'se_qda_{tag}.npy', se_qda)

n_show = [3, 6] # indexes in data: 400 (dashed), 3200 (solid)
styles = ['--', '-']

fig, axes = plt.subplots(1, len(dimensions), figsize=(4 * len(dimensions), 4),
                         sharey=True, constrained_layout=True)

for d, (ax, dimension) in enumerate(zip(axes, dimensions)):
    for k, ls in zip(n_show, styles):
        for mean_acc, se_acc, color, name in [(mean_nb, se_nb, '#eb6834', 'NB'),
                                              (mean_qda, se_qda, '#2a78d6', 'QDA')]:
            y = mean_acc[k, d, :]
            err = 2 * se_acc[k, d, :]
            ax.plot(correlations, y, ls, color=color, marker='o', ms=4, label=f'{name}, n = {data[k]}')
            ax.fill_between(correlations, y - err, y + err, color=color, alpha=0.15)
    ax.axhline(0.5, color='k', lw=1, ls=':')
    ax.set_xlabel('ρ  (class 1: +ρ, class 2: −ρ)')
    ax.set_title(f'D = {dimension}')

axes[0].set_ylabel('accuracy')
axes[0].legend(fontsize=8)
fig.suptitle(f'Same means, covariance Toeplitz(+ρ) vs Toeplitz(−ρ)  [{tag}]')
fig.savefig(f'acc_vs_rho_{tag}.png', dpi=150, bbox_inches='tight')
plt.show()