"""Tests for the Multi-Basis Graph Interaction Embedding."""

import numpy as np
import pytest
import pennylane as qml
from mbgie.core.embedding import MBGIEmbedding

def test_embedding_initialization():
    """Test proper initialization of the embedding."""
    features = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    wires = [0, 1, 2]
    
    # Test default pattern
    embedding = MBGIEmbedding(features, wires)
    assert len(embedding.wires) == 3
    assert len(embedding.pattern) == 3  # Number of combinations of 3 wires taken 2 at a time
    assert len(embedding.features) == 2  # Two feature vectors
    
    # Test custom pattern
    pattern = [(0, 1), (1, 2)]
    embedding_custom = MBGIEmbedding(features, wires, pattern)
    assert len(embedding_custom.pattern) == 2
    assert embedding_custom.pattern == ((0, 1), (1, 2))

def test_invalid_features():
    """Test that invalid features raise appropriate errors."""
    with pytest.raises(ValueError, match="Features must be a 2D array with shape"):
        # Wrong shape: should be (n_samples, 3)
        features = np.array([0.1, 0.2, 0.3])
        MBGIEmbedding(features, wires=[0, 1])
        
    with pytest.raises(ValueError, match="Features must be a 2D array with shape"):
        # Wrong number of features per sample
        features = np.array([[0.1, 0.2], [0.3, 0.4]])
        MBGIEmbedding(features, wires=[0, 1])

def test_invalid_pattern():
    """Test that invalid patterns raise appropriate errors."""
    features = np.array([[0.1, 0.2, 0.3]])
    wires = [0, 1, 2]
    
    with pytest.raises(ValueError, match="Pattern must be a 2D array with shape"):
        # Wrong pattern shape
        pattern = [(0,), (1,)]
        MBGIEmbedding(features, wires, pattern)

def test_compute_decomposition():
    """Test the quantum operation decomposition."""
    features = np.array([[0.1, 0.2, 0.3]])
    wires = [0, 1]
    pattern = [(0, 1)]
    
    embedding = MBGIEmbedding(features, wires, pattern)
    ops = embedding.compute_decomposition(features, wires, pattern)
    
    assert len(ops) == 3  # Should have XX, YY, and ZZ operations
    assert isinstance(ops[0], qml.IsingXX)
    assert isinstance(ops[1], qml.IsingYY)
    assert isinstance(ops[2], qml.IsingZZ)
    
    # Check parameter values
    assert ops[0].data[0] == 0.1  # XX strength
    assert ops[1].data[0] == 0.2  # YY strength
    assert ops[2].data[0] == 0.3  # ZZ strength

def test_circuit_execution():
    """Test actual quantum circuit execution."""
    dev = qml.device("default.qubit", wires=3)
    features = np.array([[0.1, 0.2, 0.3]])
    
    @qml.qnode(dev)
    def circuit():
        MBGIEmbedding(features, wires=[0, 1, 2])
        return qml.state()
    
    # Execute circuit
    state = circuit()
    assert state is not None
    assert len(state.shape) == 1
    assert state.shape[0] == 2**3  # 3 qubits = 8 dimensional state vector

def test_large_system_embedding():
    """Test embedding with 5 qubits and 7 features."""
    # Create 7 feature vectors with 3 components each
    features = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7],
        [0.8, 0.9, 1.0],
        [0.3, 0.4, 0.5]
    ])
    
    wires = [0, 1, 2, 3, 4]  # 5 qubits
    # Define specific pattern to test
    pattern = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (1, 3), (2, 4)]
    
    embedding = MBGIEmbedding(features, wires, pattern)
    
    # Test dimensions
    assert len(embedding.wires) == 5
    assert len(embedding.pattern) == 7
    assert len(embedding.features) == 7
    
    # Test decomposition
    ops = embedding.decomposition()
    assert len(ops) == 21  # 7 pairs × 3 operations (XX, YY, ZZ)
    
    # Test operation parameters
    for i, (xx, yy, zz) in enumerate(zip(ops[::3], ops[1::3], ops[2::3])):
        assert xx.name == "IsingXX"
        assert yy.name == "IsingYY"
        assert zz.name == "IsingZZ"
        np.testing.assert_allclose(xx.parameters[0], features[i][0])
        np.testing.assert_allclose(yy.parameters[0], features[i][1])
        np.testing.assert_allclose(zz.parameters[0], features[i][2])

def test_mbgie_with_z_measurements():
    """Test combining MBGIE with arbitrary circuits and Z-basis measurements."""
    # Initialize a 3-qubit device
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
      # Execute circuit and get measurements
    measurements = np.array(circuit())
    
    # Verify output shape and type
    assert len(measurements) == 3
    assert isinstance(measurements, np.ndarray)
    assert all(isinstance(m, float) for m in measurements)
    assert all(-1 <= m <= 1 for m in measurements)  # Z measurements should be in [-1, 1]

    # Test that the circuit is deterministic
    measurements2 = circuit()
    np.testing.assert_allclose(measurements, measurements2)  # Same circuit should give same results
