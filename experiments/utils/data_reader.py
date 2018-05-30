import csv
import logging
from collections import defaultdict


def read_dataset(path, allowed_classes=None, show_logs=True):
    """Loads columns <clean> and <label> from a tsv file.
    
    :returns X, y
    """
    
    X = list()
    y = list()
    class_freq = defaultdict(int)
    
    with open(path, 'r') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for i, line in enumerate(reader):
            label = line['label']
            
            if allowed_classes and label not in allowed_classes:
                continue
            
            X.append(line['clean'])
            y.append(label)
            class_freq[label] += 1

    if show_logs:
        logging.info('Total of {} samples'.format(len(X)))
        s = sorted(class_freq.items(), key=lambda x:x[1], reverse=True)
        for freq, word in s:
            logging.info('{} - {}'.format(freq, word))
        
    return X, y
