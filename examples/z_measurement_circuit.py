"""Example of MBGIE circuit with Z-basis measurements."""

import pennylane as qml
from mbgie.core.embedding import MBGIEmbedding
import numpy as np

# Initialize device
dev = qml.device("default.qubit", wires=3)

# Create feature vectors
features = np.array([[0.1, 0.2, 0.3]])
wires = [0, 1, 2]
pattern = [(0, 1), (1, 2)]  # Simple chain pattern

@qml.qnode(dev)
def circuit():
    # Prepare initial superposition state
    for wire in wires:
        qml.Hadamard(wire)
        
    # Apply MBGIE
    MBGIEmbedding(features, wires, pattern)
    
    # Add arbitrary circuit operations
    qml.CNOT(wires=[0, 1])
    qml.RY(0.5, wires=2)
    
    # Return Z-basis measurements
    return [qml.expval(qml.Z(wire)) for wire in wires]

# Draw the circuit
print("Circuit with Z-basis measurements:")
print(qml.draw(circuit)())
