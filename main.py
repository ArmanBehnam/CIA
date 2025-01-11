"""
main.py - Main execution script for Causal Representation Learning on ColoredMNIST
"""

import torch
from util import ColoredMNIST, ColoredMNISTKernel, RepresentationTester

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Initialize tester with colored MNIST datasets
    tester = RepresentationTester(batch_size=128, latent_dim=32)

    # Train and evaluate representations
    print("Starting training...")
    train_losses, test_results = tester.train(num_epochs=10)

    # Print final results
    print("\nFinal Results:")
    for metric, value in test_results[-1].items():
        print(f"{metric}: {value:.4f}")

    # Verify causal properties
    print("\nVerifying causal properties...")
    # Get test data
    x1, y1 = next(iter(tester.loader_e1))
    x2, y2 = next(iter(tester.loader_e2))

    # Get representations
    with torch.no_grad():
        Z_L1 = tester.phi_L(x1)
        Z_L2 = tester.phi_L(x2)
        Z_H1 = tester.phi_H(Z_L1)
        Z_H2 = tester.phi_H(Z_L2)

    # Combine data
    Z_L = torch.cat([Z_L1, Z_L2])
    Z_H = torch.cat([Z_H1, Z_H2])
    Y = torch.cat([y1, y2])
    E = torch.cat([torch.zeros_like(y1), torch.ones_like(y2)])

    # Final evaluation
    results = tester.evaluate_representations(Z_L, Z_H, Y, E, phase='test')

    # Visualize final representations
    tester.visualize_representations(Z_L, Z_H, Y, E, epoch='final')