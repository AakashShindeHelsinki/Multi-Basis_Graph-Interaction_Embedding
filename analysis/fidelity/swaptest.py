import numpy as np
import pennylane as qml
from sklearn.datasets import make_classification
from mbgie.core.embedding import MBGIEmbedding
from sklearn.preprocessing import normalize

def data_prep(data = 'classification',features = 30, n_samples = 10):
    """Prepare data for the swap test."""
    # create a dataset with 2 classes
    if data == 'classification':
        X, y = make_classification(n_samples=n_samples, n_features=features, n_classes=2, class_sep=3.0, n_informative=10, n_redundant=20, n_clusters_per_class=1, random_state=42)
    elif data == 'synthetic':
        raise NotImplementedError("Synthetic dataset not implemented yet.")
        # TODO use equation generated datasets
    elif data == 'image':
        raise NotImplementedError("Image dataset not implemented yet.")
        # TODO check and add image dataset form the previous datset change it to RGB
        
    X = normalize(X)
    
    # Split X into two arrays based on y values (0 and 1)
    X_0 = X[y == 0]
    X_1 = X[y == 1]
    
    # Calculate how many complete groups of 3 features we can make
    n_features = X.shape[1]
    n_complete_groups = n_features // 3
    
    # If features are not divisible by 3, we'll pad with zeros
    if n_features % 3 != 0:
        pad_width = 3 - (n_features % 3)
        X_0 = np.pad(X_0, ((0, 0), (0, pad_width)), mode='constant')
        X_1 = np.pad(X_1, ((0, 0), (0, pad_width)), mode='constant')
        n_complete_groups += 1
    
    # Reshape into 3D arrays where each sample has n_complete_groups sub-arrays of 3 features
    X_0 = X_0.reshape(X_0.shape[0], n_complete_groups, 3)
    X_1 = X_1.reshape(X_1.shape[0], n_complete_groups, 3)
    
    return X_0, X_1

 
def swap_test():
    ens_a, ens_b = data_prep(n_samples=10, features=30, data='classification')
    non_orthogonal = []
    orthogonal = []
    wires = [0, 1, 2, 3, 4] 
    dev = qml.device("default.qubit", wires=wires, shots=1000)
    
    @qml.qnode(dev)
    def circuit(fec_x, fec_c):
        """for wire in wires:
            qml.Hadamard([wire])"""
        MBGIEmbedding(fec_x, wires=[0, 1, 2, 3, 4])
        qml.adjoint(MBGIEmbedding(fec_c, wires=[0, 1, 2, 3, 4]))
        """for wire in wires:
            qml.Hadamard([wire])"""
        return qml.probs(wires=wires)
       
    for fec_x in ens_a:
        for fec_c in ens_a:
            non_orthogonal.append(circuit(fec_x, fec_c)[0])
            
            
    for fec_x in ens_a:
        for fec_c in ens_b:
            orthogonal.append(circuit(fec_x, fec_c)[0])
    
    for i in non_orthogonal:
        print(i)
    
    print("===================================")
        
    for i in orthogonal:
        print(i)
    
swap_test()
    