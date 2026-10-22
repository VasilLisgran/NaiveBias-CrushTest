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

centers = range (0, 10)
correlations = np.linspace(0.0, 0.9, 10)
dimensions = [2, 4, 6, 8]
data = [50, 100, 200, 400, 800, 1600, 3200]
res_matrix1 = np.zeros((len(centers), len(correlations)))
res_matrix2 = np.zeros((len(centers), len(correlations)))

# Naive-Bias and QDA -one-seed show part
'''fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
axes1 = axes1.flatten()
axes2 = axes2.flatten()'''

n_seeds = 20
axis_k = 0
tag = f'toep_e{axis_k + 1}'

final_matrix = np.full((len(data), n_seeds, len(dimensions), len(centers), len(correlations)), np.nan)

for idx, inf in enumerate(data):
    for d, dimension in enumerate(dimensions): 
        for i, center in enumerate(centers):
            for j, correlation in enumerate(correlations):

                direction = np.zeros(dimension)
                direction[axis_k] = 1     
                first = center/10 * direction
                second = -first

                prop_nums = np.arange(dimension)
                distance = np.abs(np.subtract.outer(prop_nums, prop_nums))
                Cov = correlation ** distance

                for s in range(n_seeds):
                    rng = np.random.Generator(np.random.PCG64(s))
                    samples_first = rng.multivariate_normal(mean=first, cov=Cov, size=inf)
                    samples_second = rng.multivariate_normal(mean=second, cov=Cov, size=inf)

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

                    classificator1 = GaussianClassificator()
                    classificator1.fit(train_X, train_Y)
                    predictions1 = classificator1.predict(test_X)

                    accuracy1 = np.mean(predictions1 == test_Y)
                    res_matrix1[i, j] = accuracy1

                    classificator2 = QuadraticDiscriminantClassificator()
                    classificator2.fit(train_X, train_Y)
                    predictions2 = classificator2.predict(test_X)

                    accuracy2 = np.mean(predictions2 == test_Y)
                    res_matrix2[i, j] = accuracy2

                    final_matrix[idx, s, d, i, j] = accuracy2 - accuracy1

        # Naive-Bias and QDA -one-seed show part
        '''im1 = axes1[d].imshow(res_matrix1, cmap='viridis', origin='lower', aspect='auto')
        axes1[d].set_title(f'Dimension D = {dimension}')
        axes1[d].set_xlabel(r'Correlation ($\rho$ index)')
        axes1[d].set_ylabel('Center distance')
        fig1.colorbar(im1, ax=axes1[d])  

        im2 = axes2[d].imshow(res_matrix2, cmap='viridis', origin='lower', aspect='auto')
        axes2[d].set_title(f'Dimension D = {dimension}')
        axes2[d].set_xlabel(r'Correlation ($\rho$ index)')
        axes2[d].set_ylabel('Center distance')
        fig2.colorbar(im2, ax=axes2[d])  '''

mean_diff = final_matrix.mean(axis=1)
# how spread out they are around the center in terms of seeds
se_diff = final_matrix.std(axis=1, ddof=1) / np.sqrt(n_seeds)
print(mean_diff)
print(se_diff)

# Naive-Bias and QDA -one-seed show part
'''fig1.tight_layout()
plt.tight_layout()

fig1.savefig('Naive-Bias-one-seed.png', dpi=150, bbox_inches='tight')
fig2.savefig('QDA-one-seed.png', dpi=150, bbox_inches='tight')'''

# diff_heatmaps_all
'''center_values = np.array(centers) / 10
vmax = np.abs(mean_diff).max() # 28 panels
fig, axes = plt.subplots(len(dimensions), len(data),
                         figsize=(2.4 * len(data), 2.2 * len(dimensions)),
                         sharex=True, sharey=True, constrained_layout=True)
for k, n_train in enumerate(data):
    for d, dimension in enumerate(dimensions):
        ax = axes[d, k]
        im = ax.imshow(mean_diff[k, d], cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       origin='lower', aspect='auto',
                       extent=[-0.05, 0.95, -0.05, 0.95])

        not_sig = np.abs(mean_diff[k, d]) < 2 * se_diff[k, d]
        ii, jj = np.where(not_sig)
        ax.scatter(correlations[jj], center_values[ii], s=3, c='k')

        if d == 0:
            ax.set_title(f'n = {n_train}')
        if k == 0:
            ax.set_ylabel(f'D = {dimension}\nCenter shift')
        if d == len(dimensions) - 1:
            ax.set_xlabel('ρ')'''

cells = [(3, 7), (3, 0), (3, 9), (7, 7), (0, 7)] 

fig, axes = plt.subplots(1, len(cells), figsize=(4 * len(cells), 4),
                         sharey=True, constrained_layout=True)

for ax, (ci, cj) in zip(axes, cells):          # one panel per cell
    for d, dimension in enumerate(dimensions):
        y = mean_diff[:, d, ci, cj]
        err = 2 * se_diff[:, d, ci, cj]
        ax.plot(data, y, marker='o', label=f'D = {dimension}')
        ax.fill_between(data, y - err, y + err, alpha=0.2)
    ax.axhline(0, color='k', lw=1)
    ax.set_xscale('log')
    ax.set_xlabel('n')
    ax.set_title(f'Shift {ci/10}, ρ = {correlations[cj]:.1f}')

axes[0].set_ylabel('QDA − NB (accuracy)')
axes[0].legend()
fig.suptitle(f'QDA − NB vs n: shift along x{axis_k + 1}, Toeplitz Σ  [{tag}]')
fig.savefig(f'diff_vs_n_{tag}.png', dpi=150, bbox_inches='tight')

#fig.colorbar(im, ax=axes, label='QDA − NB (accuracy)', shrink=0.8)
#fig.suptitle('QDA − NB, mean on seeds (points: |mean| < 2·SE)')
#fig.savefig('diff_heatmaps_all.png', dpi=150, bbox_inches='tight')
plt.show()

np.save(f'mean_diff_{tag}.npy', mean_diff)
np.save(f'se_diff_{tag}.npy', se_diff)