import numpy as np

def write_opf_format(X, y, path):
    """
    Writes data matrix on OPF format.
    > X: the features matrix
    > y: the labels matrix. Remember that in OPF labels are 1-indexed
    > path: where the file will be saved with '.txt' extension.
    < None.
    """
    
    with open(path, 'w') as file:
        m, n = X.shape

        # if everything is zero, then format file so OPF doesn't
        # propagate ground truth labels
        C = np.unique(y).size
        if np.sum(y) == 0:
            C = 0
        
        file.write('{} {} {}\n'.format(m, C, n))
        for index, (label, features) in enumerate(zip(y, X)):
            features = ' '.join([str(x) for x in features])
            line = '{} {} {}\n'.format(index, label, features)
            file.write(line)

