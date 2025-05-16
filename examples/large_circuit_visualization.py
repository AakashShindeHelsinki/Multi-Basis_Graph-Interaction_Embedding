"""Example of Multi-Basis Graph Interaction Embedding with 5 qubits and 7 features."""

import numpy as np
import pennylane as qml
from mbgie.core.embedding import MBGIEmbedding
from rich.console import Console
from rich.table import Table

def main():
    # Create feature vectors (7 features × 3 components)
    features = np.array([
        [0.1, 0.2, 0.3],  # Feature 1
        [0.4, 0.5, 0.6],  # Feature 2
        [0.7, 0.8, 0.9],  # Feature 3
        [0.2, 0.3, 0.4],  # Feature 4
        [0.5, 0.6, 0.7],  # Feature 5
        [0.8, 0.9, 1.0],  # Feature 6
        [0.3, 0.4, 0.5]   # Feature 7
    ])
    
    wires = [0, 1, 2, 3, 4]  # 5 qubits
    # Define pattern for nearest-neighbor + some long-range connections
    pattern = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (1, 3), (2, 4)]
    
    # Create quantum device
    dev = qml.device("default.qubit", wires=len(wires))
    
    # Define quantum circuit
    @qml.qnode(dev)
    def circuit():
        MBGIEmbedding(features, wires=wires, pattern=pattern)
        return qml.state()
    
    # Print header
    print("\n=== 5-Qubit Multi-Basis Graph Interaction Embedding ===")
    print("Features: 7 × 3D vectors")
    print("-" * 56)
    
    # Get operations details
    embedding = MBGIEmbedding(features, wires=wires, pattern=pattern)
    ops = embedding.decomposition()
    
    # 1. Show Qubit Connectivity
    print("\n1. Qubit Connectivity Pattern")
    print("-----------------------------")
    print("\nConnections:")
    for w1, w2 in pattern:
        print(f"  {w1} ←→ {w2}")
    
    # 2. Show Operations
    print("\n2. Quantum Operations")
    print("-----------------------------")
    
    # Group operations by wire pairs
    for pair_idx, (w1, w2) in enumerate(pattern):
        base_idx = pair_idx * 3
        xx, yy, zz = ops[base_idx:base_idx + 3]
        print(f"\nPair {w1}-{w2} operations:")
        print(f"  XX({xx.parameters[0]:.3f})")
        print(f"  YY({yy.parameters[0]:.3f})")
        print(f"  ZZ({zz.parameters[0]:.3f})")
    
    # 3. Execute and show state
    print("\n3. Output Quantum State")
    print("-----------------------------")
    state = circuit()
    print("\nFirst 8 basis states (of 32 total):")
    for i, amp in enumerate(state[:8]):
        if abs(amp) > 1e-10:  # Only show non-zero amplitudes
            basis = format(i, f'0{len(wires)}b')
            print(f"|{basis}⟩: {amp.real:+.6f}{amp.imag:+.6f}j")
    
    # 4. Circuit Statistics
    print("\n4. Circuit Metrics")
    print("-----------------------------")
    print(f"Number of qubits: {len(wires)}")
    print(f"Edge pairs: {len(pattern)}")
    print(f"Total gates: {len(ops)}")
    print(f"Circuit depth: {len(ops)//len(pattern)}")
    
    # Calculate connectivity statistics
    max_connections = (len(wires) * (len(wires) - 1)) // 2
    density = len(pattern) / max_connections
    print(f"\nConnectivity Statistics:")
    print(f"Maximum possible connections: {max_connections}")
    print(f"Actual connections: {len(pattern)}")
    print(f"Connectivity density: {density:.1%}")

if __name__ == "__main__":
    main()
