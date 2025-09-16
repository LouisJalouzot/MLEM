import torch


class PairwiseDataloader:
    def __init__(self, X, Y, n_pairs=4096):
        self.X = X
        self.Y = Y
        self.n_samples = X.shape[0]
        self.n_pairs = n_pairs
        self.device = X.device

    def __iter__(self):
        for _ in range(len(self)):
            yield self.sample()

    def __len__(self):
        return self.n_samples * (self.n_samples - 1) // (2 * self.n_pairs)

    def sample(self):
        ind_1 = torch.randint(0, self.n_samples, (self.n_pairs,), device=self.device)
        ind_2 = torch.randint(0, self.n_samples, (self.n_pairs,), device=self.device)

        # Ensure ind_1 and ind_2 are different
        mask = ind_1 == ind_2
        while mask.any():
            ind_2[mask] = torch.randint(
                0, self.n_samples, (mask.sum(),), device=self.device
            )
            mask = ind_1 == ind_2

        X_pairs = torch.stack([self.X[ind_1], self.X[ind_2]], dim=1)

        Y_pairs = torch.stack([self.Y[ind_1], self.Y[ind_2]], dim=1)
        true_dist = torch.cdist(Y_pairs[:, 0], Y_pairs[:, 1]).squeeze(1)

        return X_pairs, true_dist
