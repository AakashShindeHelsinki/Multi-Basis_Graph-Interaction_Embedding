# Multi-Basis Graph Interaction Embedding (MBGIE)

A quantum embedding method that uses multi-basis Ising interactions along graph edges to encode classical data into quantum states.

## Overview

MBGIE is a quantum embedding technique that:
- Maps classical feature vectors into quantum states
- Uses XX, YY, and ZZ Ising interactions
- Supports arbitrary graph connectivity patterns
- Scales efficiently with the number of qubits

## Installation

```bash
pip install -e .
```

## Usage

Basic usage example:

```python
import numpy as np
import pennylane as qml
from mbgie import MBGIEmbedding

# Create feature vectors
features = np.array([[0.1, 0.2, 0.3]])

# Define qubit wires and connection pattern
wires = [0, 1, 2]
pattern = [(0, 1), (1, 2)]

# Create quantum device
dev = qml.device("default.qubit", wires=len(wires))

# Define quantum circuit
@qml.qnode(dev)
def circuit():
    MBGIEmbedding(features, wires=wires, pattern=pattern)
    return qml.state()

# Execute circuit
state = circuit()
```

## Examples

The repository includes two example scripts:

1. `examples/circuit_visualization.py`: Shows a basic 3-qubit embedding
2. `examples/large_circuit_visualization.py`: Demonstrates a 5-qubit system with 7 features

Run the examples:
```bash
python examples/circuit_visualization.py
python examples/large_circuit_visualization.py
```

## Features

- Flexible graph connectivity patterns
- Support for arbitrary number of qubits
- Multiple feature vector embedding
- Configurable interaction strengths
- Integration with PennyLane quantum framework

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Implementation Details

The embedding consists of:

1. **Feature Mapping**: Each 3D feature vector maps to XX, YY, and ZZ interaction strengths
2. **Graph Structure**: Defines which qubit pairs interact
3. **Quantum Operations**: Applies Ising interactions between connected qubits

### Circuit Structure
For a system with N qubits and M features:
- Number of possible qubit pairs: N(N-1)/2
- Operations per feature: 3 (XX, YY, ZZ)
- Total operations: min(M, number of pairs) × 3
- Circuit depth: 3 (operations can be parallelized across pairs)

## Requirements

- Python 3.8+
- PennyLane >= 0.32.0
- NumPy >= 1.21.0
- pytest (for testing)

## License

Open source under the MIT license.
