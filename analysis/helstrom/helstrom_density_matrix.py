import numpy as np
import pennylane as qml
from sklearn.datasets import make_classification
from mbgie.core.embedding import MBGIEmbedding
from sklearn.preprocessing import normalize
from analysis.dataset.synthetic_dataset import synthetic_data_capital1

def data_prep(data = 'classification',features = 30, n_samples = 10):
    """Prepare data for the swap test."""
    # create a dataset with 2 classes
    if data == 'classification':
        X, y = make_classification(n_samples=n_samples, n_features=features, n_classes=2, class_sep=3, n_informative=30, n_repeated= 0 ,n_redundant=0,  n_clusters_per_class=1, random_state=42)
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
    elif data == 'synthetic':
        return synthetic_data_capital1(Sample_size=n_samples, nclass=2, nInfo=30, nFeature=features, nRedun=0, nNusiance=0, pthresh=0.9)
    elif data == 'image':
        raise NotImplementedError("Image dataset not implemented yet.")
        # TODO check and add image dataset form the previous datset change it to RGB
        
    



def helstrom_density_matrix():
    ens_a, ens_b = data_prep(n_samples=20, features=30, data='classification')
    wires = [0, 1, 2, 3, 4] 
    dev = qml.device("default.qubit", wires=wires)
    
    @qml.qnode(dev)
    def circuit(ens_x):
        """for wire in wires:
            qml.Hadamard([wire])
            qml.S([wire])"""
        MBGIEmbedding(ens_x,wires=wires)
        return qml.density_matrix(wires=wires)

    density_matrix_rho = np.zeros(shape=(32,32))#declare empty np array 2d array
        
    density_matrix_sigma = np.zeros(shape=(32,32)) #declare empty np array 2d array
    
    for fec_a in ens_a:
        density_matrix_rho = np.add(density_matrix_rho, circuit(fec_a))
        
    for fec_b in ens_b:
        density_matrix_sigma = np.add(density_matrix_sigma, circuit(fec_b))
        
    Difference = np.subtract(density_matrix_rho, density_matrix_sigma)
    
    # Calculate Eigen Value and respective Eigen Vector
    eigenvalues, eigenvectors = np.linalg.eig(Difference)
    # Sort the eigenvectors into 2 arrays one with eigenvalues greater than 0 and one with eigenvalues less than 0
    eigenvectors_greater_than_0 = np.zeros(shape = (32))
    eigenvectors_less_than_0 = np.zeros(shape = (32))
    for i in range(len(eigenvalues)):
        if eigenvalues[i] > 0:
            eigenvectors_greater_than_0 = np.add(eigenvectors_greater_than_0,eigenvectors[i])
        else:
            eigenvectors_less_than_0 = np.add(eigenvectors_less_than_0,eigenvectors[i]) 
            
    Sig_plus = np.outer(eigenvectors_greater_than_0, eigenvectors_greater_than_0) 
    Sig_minus = np.outer(eigenvectors_less_than_0, eigenvectors_less_than_0)
   
    Sig_diff = np.subtract(Sig_plus, Sig_minus)
    
    ens_a_trace = []
    ens_b_trace = []
    for state_x in ens_a:
        ens_a_trace.append(np.trace(np.dot(circuit(state_x), Sig_diff)))
        
    for state_x in ens_b:
        ens_b_trace.append(np.trace(np.dot(circuit(state_x), Sig_diff)))
    
    print("Trace of ens_a: ", ens_a_trace)
    print("Trace of ens_b: ", ens_b_trace)
    
helstrom_density_matrix()