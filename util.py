"""
util.py - Core implementations for Causal Representation Learning on ColoredMNIST
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
from sklearn.metrics import mutual_info_score
from torchvision import datasets
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass

class ColoredMNIST(Dataset):
    """ColoredMNIST dataset implementing anti-causal structure Y → X ← E"""
    def __init__(self, env: str, root='./data', train=True):
        self.env = env
        mnist = datasets.MNIST(root, train=train, download=True)
        self.images = mnist.data.float() / 255.0
        self.labels = mnist.targets
        self.colored_images = self._create_measure_space()

    def _create_measure_space(self) -> torch.Tensor:
        """Create measure space (D_ei, σ_D_ei) for environment ei"""
        n_images = len(self.images)
        colored = torch.zeros((n_images, 3, 28, 28))

        for i, (img, label) in enumerate(zip(self.images, self.labels)):
            p_red = 0.75 if ((label % 2 == 0) == (self.env == 'e1')) else 0.25
            is_red = torch.rand(1) < p_red

            if is_red:
                colored[i, 0] = img  # Red channel
            else:
                colored[i, 1] = img  # Green channel
            colored[i, 2] = torch.randn_like(img) * 0.01  # Small noise

        return colored

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.colored_images[idx], self.labels[idx]

@dataclass
class MeasurableSet:
    """Represents a measurable set in the σ-algebra"""
    data: torch.Tensor
    name: str


class CausalKernel:
    def __init__(self, sample_space: torch.Tensor, Y: torch.Tensor, E: torch.Tensor):
        """Initialize with smaller batch size for memory efficiency"""
        # Store data in smaller chunks
        self.batch_size = min(1000, len(Y))
        self.sample_space = sample_space[:self.batch_size]
        self.Y = Y[:self.batch_size]
        self.E = E[:self.batch_size]
        # Process data in manageable chunks
        self._process_data()

    def _process_data(self):
        """Process data in batches to manage memory"""
        self.processed_indicators = []
        chunk_size = 100  # Process in smaller chunks

        for start_idx in range(0, self.batch_size, chunk_size):
            end_idx = min(start_idx + chunk_size, self.batch_size)
            chunk_data = self.sample_space[start_idx:end_idx]

            # Create indicators for this chunk
            chunk_indicators = []
            for dim in range(chunk_data.shape[1]):
                values = torch.unique(chunk_data[:, dim])
                for val in values:
                    indicator = (chunk_data[:, dim] == val).float()
                    chunk_indicators.append(indicator)

            self.processed_indicators.extend(chunk_indicators)


class ColoredMNISTKernel:
    def __init__(self, dataset_e1: ColoredMNIST, dataset_e2: ColoredMNIST):
        """Initialize with memory-efficient data handling"""
        self.e1_data = dataset_e1
        self.e2_data = dataset_e2

        # Use subset of data for kernel computations
        batch_size = 1000
        e1_subset = self.e1_data.colored_images[:batch_size].reshape(batch_size, -1)
        e2_subset = self.e2_data.colored_images[:batch_size].reshape(batch_size, -1)

        # Initialize kernels with subsets
        self.kernel_e1 = CausalKernel(
            e1_subset.float(),  # Ensure float type
            dataset_e1.labels[:batch_size],
            torch.zeros(batch_size)
        )
        self.kernel_e2 = CausalKernel(
            e2_subset.float(),  # Ensure float type
            dataset_e2.labels[:batch_size],
            torch.ones(batch_size)
        )


class RepresentationTester:
    def __init__(self, batch_size=128, latent_dim=32):
        """Initialize with memory-efficient data handling"""
        # Initialize datasets
        self.train_e1 = ColoredMNIST('e1')
        self.train_e2 = ColoredMNIST('e2')
        self.test_e1 = ColoredMNIST('e1', train=False)
        self.test_e2 = ColoredMNIST('e2', train=False)

        # Create dataloaders with specified batch size
        self.loader_e1 = DataLoader(self.train_e1, batch_size=batch_size, shuffle=True)
        self.loader_e2 = DataLoader(self.train_e2, batch_size=batch_size, shuffle=True)

        # Initialize representations and components
        self._init_representations(latent_dim)
        self.kernel = ColoredMNISTKernel(self.train_e1, self.train_e2)
        self.optimizer = CausalOptimizer(self.phi_L, self.phi_H, self.classifier)

    def _init_representations(self, latent_dim):
        """Initialize neural network components with appropriate architecture"""
        self.phi_L = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )

        self.phi_H = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        self.classifier = nn.Linear(latent_dim, 10)

    def train(self, num_epochs=50):
        """Execute the complete training procedure with evaluation"""
        train_losses = []
        test_results = []

        for epoch in range(num_epochs):
            # Train for one epoch
            epoch_loss = self._train_epoch()
            train_losses.append(epoch_loss)

            # Evaluate every 5 epochs
            if epoch % 5 == 0:
                print(f"\nEpoch {epoch}")
                print(f"Average Loss: {epoch_loss:.4f}")

                # Evaluate on combined data
                with torch.no_grad():
                    x1, y1 = next(iter(self.loader_e1))
                    x2, y2 = next(iter(self.loader_e2))
                    x = torch.cat([x1, x2])
                    y = torch.cat([y1, y2])
                    e = torch.cat([torch.zeros_like(y1), torch.ones_like(y2)])

                    # Get representations
                    Z_L = self.phi_L(x)
                    Z_H = self.phi_H(Z_L)

                    # Evaluate representations
                    results = self.evaluate_representations(Z_L, Z_H, y, e)
                    test_results.append(results)

        return train_losses, test_results

    def _train_epoch(self):
        """Train for one epoch"""
        total_loss = 0
        num_batches = 0

        for (x1, y1), (x2, y2) in zip(self.loader_e1, self.loader_e2):
            # Prepare environment labels
            e1 = torch.zeros_like(y1)
            e2 = torch.ones_like(y2)

            # Combine batch data
            x = torch.cat([x1, x2])
            y = torch.cat([y1, y2])
            e = torch.cat([e1, e2])

            # Forward pass and optimization
            Z_L = self.phi_L(x)
            Z_H = self.phi_H(Z_L)

            # Update parameters
            losses = self.optimizer.train_step(x, y, e, self.kernel)
            total_loss += losses['total_loss']
            num_batches += 1

        return total_loss / num_batches

    def evaluate_representations(self, Z_L, Z_H, Y, E, phase='train'):
        """Evaluate the learned representations"""
        results = {}

        # Environment independence test
        env_independence = self._test_environment_independence(Z_H, Y, E)
        results['env_independence'] = env_independence

        # Low-level invariance test
        low_level_inv = self._test_low_level_invariance(Z_L, Y, E)
        results['low_level_invariance'] = low_level_inv

        # Classification accuracy
        with torch.no_grad():
            logits = self.classifier(Z_H)
            acc = (logits.argmax(1) == Y).float().mean()
            results['accuracy'] = acc.item()

        return results

    def _test_environment_independence(self, Z_H: torch.Tensor, Y: torch.Tensor, E: torch.Tensor,
                                       threshold: float = 0.05) -> float:
        """Test if Z_H ⊥ E|Y using mutual information"""
        mi_score = 0.0
        for y in torch.unique(Y):
            mask = Y == y
            if mask.sum() > 1:
                z_y = Z_H[mask]
                e_y = E[mask]

                # Compute normalized mutual information
                z_flat = z_y[:, 0].detach().numpy()  # Take first dimension for MI computation
                e_flat = e_y.numpy()
                joint = np.histogram2d(z_flat, e_flat)[0]
                mi = mutual_info_score(None, None, contingency=joint)
                mi_score += mi / np.log(len(z_y))

        return mi_score < threshold

    def _test_low_level_invariance(self, Z_L: torch.Tensor, Y: torch.Tensor, E: torch.Tensor,
                                   threshold: float = 0.05) -> float:
        """Test if P(Z_L|Y) is invariant across environments"""
        max_diff = 0.0
        for y in torch.unique(Y):
            mask_y = Y == y
            for e1 in torch.unique(E):
                for e2 in torch.unique(E):
                    if e1 != e2:
                        mask_e1 = E == e1
                        mask_e2 = E == e2

                        # Compute conditional means
                        z1 = Z_L[mask_y & mask_e1]
                        z2 = Z_L[mask_y & mask_e2]

                        if len(z1) > 0 and len(z2) > 0:
                            diff = torch.norm(z1.mean(0) - z2.mean(0)).item()
                            max_diff = max(max_diff, diff)

        return max_diff < threshold

    def visualize_representations(self, Z_L: torch.Tensor, Z_H: torch.Tensor,
                                  Y: torch.Tensor, E: torch.Tensor, epoch: str) -> None:
        """Visualize learned representations through multiple analysis perspectives"""
        plt.figure(figsize=(20, 5))

        # Analyze low-level representation structure
        plt.subplot(141)
        self._plot_tsne(Z_L, Y, 'Low-level (Z_L) by Digit')

        # Analyze high-level representation structure
        plt.subplot(142)
        self._plot_tsne(Z_H, Y, 'High-level (Z_H) by Digit')

        # Analyze environment separation
        plt.subplot(143)
        self._plot_tsne(Z_H, E, 'Z_H by Environment',
                        classes=['Env 1', 'Env 2'])

        # Analyze parity-based clustering
        plt.subplot(144)
        self._plot_tsne(Z_H, Y % 2, 'Z_H by Parity',
                        classes=['Even', 'Odd'])

        plt.suptitle(f'Representation Analysis at Epoch {epoch}')
        plt.tight_layout()
        plt.show()

    def _plot_tsne(self, Z: torch.Tensor, labels: torch.Tensor,
                   title: str, classes: List[str] = None) -> None:
        """Generate dimensionality-reduced visualization using t-SNE"""
        tsne = TSNE(n_components=2, random_state=42)
        Z_2d = tsne.fit_transform(Z.detach().numpy())

        if classes is None:
            classes = [str(i) for i in range(10)]

        for i, label in enumerate(torch.unique(labels)):
            mask = labels == label
            plt.scatter(Z_2d[mask, 0], Z_2d[mask, 1],
                        label=classes[i], alpha=0.6)

        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')



class CausalOptimizer:
    """Implements causal optimization objectives"""
    def __init__(self, phi_L, phi_H, classifier, lambda1=0.1, lambda2=0.1):
        self.phi_L = phi_L
        self.phi_H = phi_H
        self.classifier = classifier
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.optimizer = torch.optim.Adam([
            {'params': phi_L.parameters(), 'lr': 0.0001},
            {'params': phi_H.parameters(), 'lr': 0.0001},
            {'params': classifier.parameters(), 'lr': 0.0001}
        ])

    def train_step(self, x, y, e, kernel):
        """Single optimization step"""
        self.optimizer.zero_grad()

        Z_L = self.phi_L(x)
        Z_H = self.phi_H(Z_L)
        logits = self.classifier(Z_H)

        # Prediction loss
        pred_loss = nn.CrossEntropyLoss()(logits, y)

        # Environment independence loss
        env_loss = self._compute_environment_loss(Z_H, y, e)

        # Low-level invariance loss
        low_level_loss = self._compute_low_level_loss(Z_L, y)

        # Total loss
        total_loss = pred_loss + self.lambda1 * env_loss + self.lambda2 * low_level_loss

        if not torch.isnan(total_loss):
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.phi_L.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.phi_H.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 1.0)
            self.optimizer.step()

        return {
            'total_loss': total_loss.item() if not torch.isnan(total_loss) else 0.0,
            'pred_loss': pred_loss.item(),
            'env_loss': env_loss.item(),
            'low_level_loss': low_level_loss.item()
        }

    def _compute_environment_loss(self, Z_H, y, e):
        """Compute environment independence loss"""
        env_loss = torch.tensor(0., requires_grad=True)
        for digit in torch.unique(y):
            digit_mask = y == digit
            if torch.sum(digit_mask) > 1:
                Z_digit = Z_H[digit_mask]
                e_digit = e[digit_mask]

                for env in [0, 1]:
                    env_mask = e_digit == env
                    if torch.sum(env_mask) > 1:
                        mean = Z_digit[env_mask].mean(0)
                        std = Z_digit[env_mask].std(0) + 1e-6
                        Z_digit[env_mask] = (Z_digit[env_mask] - mean) / std

                env_loss = env_loss + torch.norm(Z_digit, p=2)
        return env_loss

    def _compute_low_level_loss(self, Z_L, y):
        """Compute low-level invariance loss"""
        low_level_loss = torch.tensor(0., requires_grad=True)
        for digit in torch.unique(y):
            digit_mask = y == digit
            if torch.sum(digit_mask) > 1:
                Z_digit = Z_L[digit_mask]
                mean = Z_digit.mean(0, keepdim=True)
                std = Z_digit.std(0, keepdim=True) + 1e-6
                Z_norm = (Z_digit - mean) / std
                pairwise_dist = torch.pdist(Z_norm, p=2)
                low_level_loss = low_level_loss + pairwise_dist.mean()
                return low_level_loss

    def evaluate_representations(self, Z_L, Z_H, Y, E, phase='train'):
        """Evaluate the learned representations"""
        results = {}
        env_independence = self._test_environment_independence(Z_H, Y, E)
        results['env_independence'] = env_independence
        low_level_inv = self._test_low_level_invariance(Z_L, Y, E)
        results['low_level_invariance'] = low_level_inv
        with torch.no_grad():
            logits = self.classifier(Z_H)
            acc = (logits.argmax(1) == Y).float().mean()
            results['accuracy'] = acc.item()
        return results

