# Quick Start Guide

This guide will help you get started with the Multi-Basis Graph Interaction Embedding (MBGIE) package.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Multi-Basis_Graph-Interaction_Embedding
```

2. Install the package:
```bash
pip install -e .
```

## Basic Usage

Here's a simple example of how to use MBGIE:

```python
import numpy as np
import pennylane as qml
from mbgie import MBGIEmbedding

# 1. Prepare your data
features = np.array([
    [0.1, 0.2, 0.3],  # First feature vector
    [0.4, 0.5, 0.6]   # Second feature vector
])

# 2. Define your quantum system
wires = [0, 1, 2]  # Three qubits
pattern = [(0, 1), (1, 2)]  # Connect nearest neighbors

# 3. Create a quantum device
dev = qml.device("default.qubit", wires=len(wires))

# 4. Define your quantum circuit
@qml.qnode(dev)
def circuit():
    MBGIEmbedding(features, wires=wires, pattern=pattern)
    return qml.state()

# 5. Execute the circuit
state = circuit()
print("Output quantum state:", state)
```

## Advanced Usage

### Custom Connection Patterns

You can define custom connection patterns between qubits:

```python
# All-to-all connectivity for 3 qubits
pattern = [(0, 1), (0, 2), (1, 2)]

# Star topology with qubit 0 at center
pattern = [(0, 1), (0, 2), (0, 3)]

# Ring topology
pattern = [(0, 1), (1, 2), (2, 3), (3, 0)]
```

### Large-Scale Systems

For larger systems, you might want to use specific connection patterns:

```python
import numpy as np
from itertools import combinations

# Create 5-qubit system with 7 features
n_qubits = 5
features = np.random.rand(7, 3)  # 7 features, each with 3 components
wires = list(range(n_qubits))

# Generate all possible pairs
all_pairs = list(combinations(wires, 2))

# Select specific pairs for your pattern
pattern = all_pairs[:7]  # Use first 7 pairs
```

## Visualization

Use the provided visualization examples to understand your circuits:

```bash
# View basic 3-qubit example
python examples/circuit_visualization.py

# View larger 5-qubit example
python examples/large_circuit_visualization.py
```

## Testing

Run the test suite to verify everything is working:

```bash
python -m pytest tests/
```

## Next Steps

1. Check out the example scripts in the `examples/` directory
2. Read the full documentation in the `docs/` directory
3. Explore the source code in `mbgie/core/`
4. Try creating your own quantum circuits using MBGIE

## Getting Help

If you encounter any issues:
1. Check the documentation in `docs/`
2. Look at the example scripts
3. Run the tests to verify your installation
4. Create an issue on the repository
