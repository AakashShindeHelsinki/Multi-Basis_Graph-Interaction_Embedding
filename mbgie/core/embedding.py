"""
Multi-Basis Graph Interaction Embedding (MBGIE)

This module implements a quantum embedding method that maps classical data into quantum states
using multi-basis Ising interactions along graph edges. The embedding uses XX, YY, and ZZ
interactions with strengths determined by the input feature vectors.

Copyright (c) 2025 Multi-Basis Graph Interaction Embedding Contributors
SPDX-License-Identifier: MIT

Example:
    >>> import numpy as np
    >>> import pennylane as qml
    >>> from mbgie import MBGIEmbedding
    >>> 
    >>> # Create feature vectors
    >>> features = np.array([[0.1, 0.2, 0.3]])
    >>> wires = [0, 1, 2]
    >>> pattern = [(0, 1), (1, 2)]
    >>> 
    >>> # Create quantum device
    >>> dev = qml.device("default.qubit", wires=len(wires))
    >>> 
    >>> # Define quantum circuit
    >>> @qml.qnode(dev)
    >>> def circuit():
    >>>     MBGIEmbedding(features, wires=wires, pattern=pattern)
    >>>     return qml.state()
    >>> 
    >>> # Execute circuit
    >>> state = circuit()

The embedding maps each 3D feature vector to three Ising interactions (XX, YY, ZZ) between
pairs of qubits specified by the connection pattern. If there are more pairs than features,
the remaining pairs get zero-strength interactions.

References:
    - PennyLane documentation: https://pennylane.ai
    - Ising interaction gates: https://pennylane.ai/qml/demos/tutorial_isingmodel
"""

import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Sequence
import pennylane as qml
from itertools import combinations
from pennylane.operation import AnyWires, Operation

class MBGIEmbedding(Operation):
    """Multi-Basis Graph Interaction Embedding quantum operation.
    
    This class implements a custom quantum operation that embeds classical feature vectors
    into quantum states using multi-basis Ising interactions (XX, YY, ZZ) between pairs
    of qubits defined by a graph structure.

    Args:
        features (np.ndarray): Input feature array of shape (n_samples, 3) where each
            sample is a 3D vector specifying XX, YY, and ZZ interaction strengths.
        wires (Sequence[int]): Qubit wires to use for the embedding.
        pattern (Optional[List[Tuple[int, int]]]): List of qubit pairs specifying the
            interaction graph structure. If None, all possible pairs are used.

    Raises:
        ValueError: If features is not a 2D array with exactly 3 features per sample.
        ValueError: If pattern is provided but not in correct shape (n_edges, 2).

    Attributes:
        features (np.ndarray): The input feature vectors.
        pattern (List[Tuple[int, int]]): The qubit interaction pattern.
        wires (List[int]): The qubit wires used by the operation.

    Note:
        The number of features should match the number of qubit pairs in the pattern.
        If there are more pairs than features, the extra pairs get zero-strength
        interactions. If there are more features than pairs, the extra features are
        ignored.
    """
    
    num_wires = AnyWires
    grad_method = None
    
    def __init__(
        self,
        features: np.ndarray,
        wires: Sequence[int],
        pattern: Optional[List[Tuple[int, int]]] = None
    ):
        """Initialize the Multi-Basis Graph Interaction Embedding."""
        shape = qml.math.shape(features)
        
        if len(shape) != 2 or shape[1] != 3:
            raise ValueError(
                f"Features must be a 2D array with shape (n_samples, 3), got shape {shape}"
            )
        
        # Call parent constructor first
        super().__init__(wires=tuple(wires))
        
        self._features = features
        
        # Generate pattern if not provided
        if pattern is None:
            self._pattern = tuple(combinations(wires, 2))
        else:
            shape_pattern = qml.math.shape(pattern)
            if len(shape_pattern) != 2 or shape_pattern[1] != 2:
                raise ValueError(
                    f"Pattern must be a 2D array with shape (n_edges, 2), got shape {shape_pattern}"
                )
            self._pattern = tuple(map(tuple, pattern))
    
    @property
    def num_params(self):
        """Number of trainable parameters."""
        return 0
    
    @property
    def features(self):
        """Input features for the embedding."""
        return self._features
    
    @property
    def pattern(self):
        """Pattern of wire pairs for the embedding."""
        return self._pattern
        
    def decomposition(self):
        """Decompose the operation into other operations."""
        return self.compute_decomposition(self.features, self.wires, self.pattern)
    
    @staticmethod
    def compute_decomposition(
        features: np.ndarray,
        wires: Sequence[int],
        pattern: List[Tuple[int, int]]
    ) -> List[Operation]:
        """
        Compute the decomposition of the embedding operation.

        Args:
            features (np.ndarray): Input features with shape (n_samples, 3)
            wires (Sequence[int]): Qubit wires to use for the embedding
            pattern (List[Tuple[int, int]]): Pattern of wire pairs for the embedding

        Returns:
            List[Operation]: List of quantum operations representing the embedding
        """
        operations_list = []
        
        for i, (wire1, wire2) in enumerate(pattern):
            if i < len(features):
                # Apply Ising interactions between wire pairs using feature values
                operations_list.append(qml.IsingXX(features[i][0], wires=[wire1, wire2]))
                operations_list.append(qml.IsingYY(features[i][1], wires=[wire1, wire2]))
                operations_list.append(qml.IsingZZ(features[i][2], wires=[wire1, wire2]))
            else:
                # Pad with identity-like operations (strength=0) if more pairs than features
                operations_list.append(qml.IsingXX(0.0, wires=[wire1, wire2]))
                operations_list.append(qml.IsingYY(0.0, wires=[wire1, wire2]))
                operations_list.append(qml.IsingZZ(0.0, wires=[wire1, wire2]))
                
        return operations_list

