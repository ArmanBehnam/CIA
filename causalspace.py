"""
causalspace.py
"""

from typing import List, Tuple, Set
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
from torchvision import datasets

@dataclass
class MeasurableSet:
    """
    Represents a set A in the σ-algebra H_ei (a subset of the sample space Omega_ei by the probability measure P_ei)
        Represents a measurable set in σ-algebra as defined in Definition 1

    """
    data: torch.Tensor  # Indicator function of the set
    name: str  # Description

    def __post_init__(self):
        if not isinstance(self.data, torch.Tensor):
            self.data = torch.tensor(self.data, dtype=torch.bool)

    def __contains__(self, omega: torch.Tensor) -> bool:
        """Check if point omega is in the set"""
        return bool(self.data[tuple(omega.long())])

    def intersection(self, other: 'MeasurableSet') -> 'MeasurableSet':
        """Set intersection operation"""
        return MeasurableSet(
            data=self.data & other.data,
            name=f"({self.name} ∩ {other.name})"
        )

    def union(self, other: 'MeasurableSet') -> 'MeasurableSet':
        """Set union operation"""
        return MeasurableSet(
            data=self.data | other.data,
            name=f"({self.name} ∪ {other.name})"
        )

    def complement(self) -> 'MeasurableSet':
        """Set complement operation"""
        return MeasurableSet(
            data=~self.data,
            name=f"({self.name})ᶜ"
        )


@dataclass
class MeasurableSpace:
    """
    Represents (D_ei, σ_D_ei) for environment e_i as defined in def 1
    """
    # data: List[Tuple[torch.Tensor, torch.Tensor]]  # (x^ei_j, y^ei_j) pairs
    input_space: torch.Tensor  # x^ei_j in D_V_L
    output_space: torch.Tensor  # y^ei_j in D_Y
    index_set: torch.Tensor  # T_ei
    distribution: Callable  # p_ei

    def __post_init__(self):
        self.sigma_algebra = self._generate_sigma_algebra()

    def _generate_sigma_algebra(self) -> List[MeasurableSet]:
        """Generate the σ-algebra for the space"""
        # Start with empty set and full space
        n = len(self.input_space)
        base_sets = [
            MeasurableSet(torch.zeros(n, dtype=torch.bool), "∅"),
            MeasurableSet(torch.ones(n, dtype=torch.bool), "Ω")
        ]
        return base_sets

    def get_empirical_measure(self, A: MeasurableSet) -> float:
        """Compute empirical probability measure of set A"""
        return float(torch.sum(A.data)) / len(A.data)

@dataclass
class CausalSpace:
    """
    Represents causal space (Ω_ei, H_ei, P_ei, K_ei) as per Definition 2
    """
    sample_space: torch.Tensor      # Ω_ei
    sigma_algebra: List[MeasurableSet]  # H_ei
    probability_measure: Callable    # P_ei
    causal_mechanism: Callable      # K_ei

    def evaluate_kernel(self, omega: torch.Tensor, A: MeasurableSet) -> float:
        """Evaluate causal kernel K_ei(ω,A)"""
        return self.causal_mechanism(omega, A)


@dataclass
class ProductCausalSpace:
    """
    Represents product space (Ω, H, P, K) as per Definition 3
    """
    spaces: List[CausalSpace]

    def __init__(self, spaces: List[CausalSpace]):
        self.spaces = spaces
        self.sample_space = self._product_sample_space()
        self.sigma_algebra = self._product_sigma_algebra()
        self.probability = self._product_probability()
        self.kernel = self._product_kernel()

    def __post_init__(self):
        self.sample_space = self._product_sample_space()
        self.sigma_algebra = self._product_sigma_algebra()

    def _product_sample_space(self) -> torch.Tensor:
        """Compute product of sample spaces"""
        return torch.cartesian_prod(*[space.sample_space for space in self.spaces])

    def _product_sigma_algebra(self) -> List[MeasurableSet]:
        """Generate product σ-algebra"""
        # Start with base product sets
        product_sets = []
        for sets in zip(*[space.sigma_algebra for space in self.spaces]):
            indicator = torch.ones(len(self.sample_space), dtype=torch.bool)
            for set_i in sets:
                indicator &= set_i.data
            product_sets.append(MeasurableSet(indicator, "× ".join(s.name for s in sets)))
        return product_sets

    def get_sub_sigma_algebra(self, subset_idx: List[int]) -> List[MeasurableSet]:
        """
        Get sub-σ-algebra H_s for subset s of index set T as per Definition 4
        """
        selected_spaces = [self.spaces[i] for i in subset_idx]
        sub_product = ProductCausalSpace(selected_spaces)
        return sub_product.sigma_algebra