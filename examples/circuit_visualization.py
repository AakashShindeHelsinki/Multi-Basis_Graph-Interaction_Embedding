"""Example of Multi-Basis Graph Interaction Embedding circuit visualization."""

import numpy as np
import pennylane as qml
from mbgie.core.embedding import MBGIEmbedding

def main():
    # Create a simple dataset
    features = np.array([[0.1, 0.2, 0.3]])
    wires = [0, 1, 2]
    pattern = [(0, 1), (1, 2)]  # Define a specific pattern for visualization
    
    # Create quantum device
    dev = qml.device("default.qubit", wires=len(wires))
    
    # Define quantum circuit
    @qml.qnode(dev)
    def circuit():
        MBGIEmbedding(features, wires=wires, pattern=pattern)
        return qml.state()
    
    # Print header
    print("Multi-Basis Graph Interaction Embedding Example")
    print("=" * 50 + "\n")
    
    # Get operations details
    embedding = MBGIEmbedding(features, wires=wires, pattern=pattern)
    ops = embedding.decomposition()
    
    # 1. Print Circuit Structure
    print("1. Circuit Structure")
    print("-" * 30)
    
    # Visual representation of the circuit structure
    print("\nCircuit Layout:")
    for wire in wires:
        wire_ops = [op for op in ops if wire in op.wires]
        print(f"Wire {wire}: ", end="")
        print("─" * 4, end="")
        for op in wire_ops:
            print(f"[{op.name}]─", end="")
        print()
    
    # 2. Print Operation Details
    print("\n2. Quantum Operations")
    print("-" * 30)
    print(f"\nTotal of {len(ops)} quantum operations:")
    for i, op in enumerate(ops, 1):
        params = ", ".join(f"{p:.3f}" for p in op.parameters)
        print(f"{i}. {op.name}({params}) on wires {list(op.wires)}")
    
    # 3. Execute Circuit
    print("\n3. Output Quantum State")
    print("-" * 30)
    state = circuit()
    print("\nFirst 4 state amplitudes:")
    for i, amp in enumerate(state[:4]):
        print(f"|{i}⟩: {amp.real:.6f}{amp.imag:+.6f}j")
    
    # 4. Circuit Metrics
    print("\n4. Circuit Metrics")
    print("-" * 30)
    print(f"Number of qubits: {len(wires)}")
    print(f"Number of edge pairs: {len(pattern)}")
    print(f"Total gates: {len(ops)}")
    print(f"Circuit depth: {len(ops)//len(pattern)}")  # 3 operations per pair (XX, YY, ZZ)
    print(f"Gate types: {sorted(set(op.name for op in ops))}")

if __name__ == "__main__":
    main()
