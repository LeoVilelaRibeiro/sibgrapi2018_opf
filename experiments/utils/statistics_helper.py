import csv
from collections import OrderedDict, defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def get_split_statistics(filepath):
    """
    Computes the label frequency, amount of utterances and tokens for a
    preprocessed tsv file.
    """
    
    label_freq = defaultdict(int)
    amount_tokens = 0

    with open(filepath, 'r') as file:
        reader = csv.DictReader(file, delimiter='\t')

        for line in reader:
            label_freq[line['label']] += 1

            text = line['clean']
            amount_tokens += len(text.split(' '))

        amount_samples = sum([e for e in label_freq.values()])

    label_freq = OrderedDict(sorted(label_freq.items(), key=lambda x: x[0]))
    return label_freq, amount_samples, amount_tokens


def get_dataset_statistics(dataset_path, add_fields=True):
    """
    Computes the label frequency, amount of utterances and tokens for
    the entire dataset, considering train/dev/test.
    :param filepath should have the format '.../ds_name' (no split name,
    nor extension).
    """

    n_labels = list()
    n_samples = list()
    n_tokens = list()

    for split in ['train', 'dev', 'test']:
        filepath = '{}_{}.tsv'.format(dataset_path, split)
        label_freq, amount_samples, amount_tokens = get_split_statistics(filepath)
        
        if add_fields:
            label_freq['samples'] = amount_samples
            label_freq['tokens'] = amount_tokens

        n_samples.append(amount_samples)
        n_tokens.append(amount_tokens)
        n_labels.append(label_freq)

    return n_labels, n_samples, n_tokens


def print_all_statistics(filepath):
    """
    Prints the label distribution, amount of utterances and tokens for
    a given dataset.
    """
    
    label_freq, _, _ = get_dataset_statistics(filepath)
    
    print('{:15.15}\t{:11} {:6}  {:7.7}  {:7.7}'.format('-label-', '-train-', '-dev-', '-test-', '-total-'))
    for header in label_freq[0].keys():
        print('{:12.12}\t'.format(header), end='')
        
        # split_no = train, dev, test
        total_labels = 0
        for split_no in range(3):
            amount_in_split = label_freq[split_no].get(header, 0)
            total_labels += amount_in_split
            print('{:7}  '.format(amount_in_split), end='')
        print('{:7}'.format(total_labels))


def plot_all_statistics(filepath, plot_width=15, plot_height=4.5):
    """Plots the label distribution on a given dataset split."""
    # based on https://matplotlib.org/gallery/units/bar_unit_demo.html
    
    def fill_in_zeros(all_labels, freq_table):
        return list([freq_table.get(label, 0) for label in all_labels])
        
    
    label_freq, amount_samples, amount_tokens = get_dataset_statistics(filepath, False)
        
    # we find which of the 3 (train/dev/test) dicts have more labels, just in case some
    # split lacks one class (we hope not)
    labels_len = [len(labels) for labels in label_freq]
    N = max(labels_len)
    Ni = np.argmax(labels_len)
    
    # label_freq is a dict that maps label to frequency. Just the frequency is necessary
    # for plotting, and Python's olist doesn't support slicing, hence they have to be
    # converted explicitly.
    
    all_labels = label_freq[Ni].keys()
    frequencies = list(map(lambda x: fill_in_zeros(all_labels, x), label_freq))
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    ax.grid(linestyle='solid', linewidth=0.5, color='lightgrey')
    matplotlib.rcParams.update({'font.size': 12})

    ind = np.arange(N)    # the x locations for the groups
    width = 0.25          # the width of the bars

    # Plotting
    p1 = ax.bar(ind, frequencies[0], width, color='r', bottom=0)
    p2 = ax.bar(ind + width, frequencies[1], width, color='y', bottom=0)
    p3 = ax.bar(ind + 2*width, frequencies[2], width, color='b', bottom=0)

    ax.legend((p1[0], p2[0], p3[0]), ('Train', 'Dev', 'Test'), loc=0)

    # Formatting x-axis labels
    dataset_name = filepath.split('/')[-1]
    ax.set_title('Label distribution across splits for {} dataset'.format(dataset_name))
    ax.set_xticks(ind + width / 2 + 0.2)
    ax.set_xticklabels(label_freq[0].keys())

    ax.autoscale_view()

    plt.xticks(rotation=90)
    plt.show()