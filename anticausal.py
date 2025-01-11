"""
anticausal.py - Implementation of anti-causal kernel calculations with fixed tensor handling
"""

import matplotlib.pyplot as plt
from torchvision import datasets
import torch
from causalkernel import CausalKernel

class ColoredMNIST:
    def __init__(self, env, root='./data'):
        self.env = env
        mnist = datasets.MNIST(root, train=(env == 'e1'), download=True)
        self.images = mnist.data.float() / 255.0
        self.labels = mnist.targets
        self.colored_images = self.color_images()

    def color_images(self):
        n_images = len(self.images)
        colored = torch.zeros((n_images, 3, 28, 28))

        for i, (img, label) in enumerate(zip(self.images, self.labels)):
            p_red = 0.75 if ((label % 2 == 0) == (self.env == 'e1')) else 0.25
            is_red = torch.rand(1) < p_red

            if is_red:
                colored[i, 0] = img  # Red channel
            else:
                colored[i, 1] = img  # Green channel

        return colored

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.colored_images[idx], self.labels[idx]

class ColoredMNISTKernel:
    def __init__(self, dataset_e1: ColoredMNIST, dataset_e2: ColoredMNIST):
        self.e1_data = dataset_e1
        self.e2_data = dataset_e2

        # Convert to batched format
        self.kernel_e1 = CausalKernel(
            sample_space=dataset_e1.colored_images.reshape(len(dataset_e1), -1),  # Flatten images
            Y=dataset_e1.labels,
            E=torch.zeros_like(dataset_e1.labels)
        )
        self.kernel_e2 = CausalKernel(
            sample_space=dataset_e2.colored_images.reshape(len(dataset_e2), -1),  # Flatten images
            Y=dataset_e2.labels,
            E=torch.ones_like(dataset_e2.labels)
        )

    def compute_environment_kernel(self, omega: torch.Tensor, A: torch.Tensor, env: str) -> float:
        """
        Compute K_{e_i}(omega, A) for specific environment
        """
        kernel = self.kernel_e1 if env == 'e1' else self.kernel_e2

        # Reshape omega if needed
        if omega.dim() == 3:  # If input is a single image (3,28,28)
            omega = omega.reshape(-1)  # Flatten to 1D

        # Get color information - now using channel sums properly
        red_sum = omega[:784].sum()  # First 784 elements are red channel
        green_sum = omega[784:1568].sum()  # Next 784 are green channel
        is_red = red_sum > green_sum

        # Find the matching sample in kernel's sample space
        distances = torch.cdist(omega.unsqueeze(0), kernel.sample_space)
        closest_idx = distances.argmin().item()
        label = kernel.Y[closest_idx]

        # Compute p(color|label, env) as per Lemma CMNIST
        p_red = 0.75 if ((label % 2 == 0) == (env == 'e1')) else 0.25

        # Check if A matches the color condition - modified for flattened format
        if A.dim() == 4:  # If A is batch of images
            A = A.reshape(A.size(0), -1)
        A_red_sum = A[:, :784].sum(1)
        A_green_sum = A[:, 784:1568].sum(1)
        A_is_red = A_red_sum > A_green_sum

        return p_red if is_red == A_is_red.any() else (1 - p_red)

    def compute_product_kernel(self, omega: torch.Tensor, A: torch.Tensor) -> float:
        """
        Compute K_s(omega, A) = K_{e1}(omega_{e1}, A_{e1}) ⊗ K_{e2}(omega_{e2}, A_{e2})
        """
        # Ensure omega and A are properly formatted
        if omega.dim() == 3:
            omega = omega.reshape(-1)
        if A.dim() == 4:
            A = A.reshape(A.size(0), -1)

        # Split data for environments (not the tensors)
        mid_point = len(A) // 2
        A_e1, A_e2 = A[:mid_point], A[mid_point:]

        # Compute individual kernels
        k_e1 = self.compute_environment_kernel(omega, A_e1, 'e1')
        k_e2 = self.compute_environment_kernel(omega, A_e2, 'e2')

        return k_e1 * k_e2

    def visualize_kernels(self, n_samples: int = 100):
        """
        Visualize kernel values for both environments and product space
        """
        plt.figure(figsize=(15, 5))

        # Sample points
        idx = torch.randperm(len(self.e1_data))[:n_samples]
        omegas = self.e1_data.colored_images[idx]
        labels = self.e1_data.labels[idx]

        k_e1_values = []
        k_e2_values = []
        k_prod_values = []

        for omega, label in zip(omegas, labels):
            # Create test set A (same label points)
            mask = self.e1_data.labels == label
            A = self.e1_data.colored_images[mask]

            k_e1 = self.compute_environment_kernel(omega, A, 'e1')
            k_e2 = self.compute_environment_kernel(omega, A, 'e2')
            k_prod = self.compute_product_kernel(omega, A)

            k_e1_values.append(k_e1)
            k_e2_values.append(k_e2)
            k_prod_values.append(k_prod)

        # Plot results
        plt.subplot(131)
        plt.hist(k_e1_values, bins=20)
        plt.title('K_{e1} Distribution')

        plt.subplot(132)
        plt.hist(k_e2_values, bins=20)
        plt.title('K_{e2} Distribution')

        plt.subplot(133)
        plt.hist(k_prod_values, bins=20)
        plt.title('K_s Product Distribution')

        plt.tight_layout()
        plt.show()