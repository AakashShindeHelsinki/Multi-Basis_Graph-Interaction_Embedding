from synthetic_data.synthetic_data import make_tabular_data
import numpy as np

def synthetic_data_capital1(Sample_size=10, nclass=2, nInfo=30, nFeature=30, nRedun=0, nNusiance=0, pthresh=0.7):

    expr2 ="x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + x21 + x22 + x23 + x24 + x25 + x26 + x27 + x28 + x29+x30"
    col_map = {"x1": 1, "x2": 1,"x3":1, "x4": 1, "x5": 1, "x6": 1, "x7": 1, "x8": 1, "x9": 1, "x10": 1, "x11": 1, "x12": 1, "x13": 1, "x14": 1, "x15": 1, 
               "x16": 1, "x17": 1, "x18": 1, "x19": 1, "x20": 1, "x21": 1, "x22": 1, "x23": 1, "x24": 1, "x25": 1, "x26": 1, "x27": 1, "x28": 1, "x29": 1,"x30":1}
    cov = np.identity(30)
    X, y_reg, y_prob, y_label = make_tabular_data(n_samples=Sample_size, n_informative=nInfo, n_redundant=nRedun ,n_nuisance=nNusiance, cov=cov, n_classes=nclass, col_map=col_map, expr=expr2, p_thresh=pthresh, seed=4)
    
    y = np.round(y_prob).astype(int)
    
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

"""X0, X1 = synthetic_data_capital1(Sample_size=10, nclass=2, nInfo=30, nFeature=30, nRedun=0, nNusiance=0, pthresh=0.7)
print(X0)
print(X1)"""