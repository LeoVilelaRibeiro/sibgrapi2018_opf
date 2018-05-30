import logging
from collections import defaultdict

import nltk
import numpy as np
from scipy.stats import mode
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords


def tokenize_stanford(X, tokenizer):
    """Tokenizes a list of sentences using the Stanford Tokenizer.
    
    :return (a,b) where <a> is A list of list of tokens and <b> is
    the word frequency on the dataset.
    """
    
    single_sentence = '____'.join(X)
    all_tokens = tokenizer.tokenize(single_sentence)
    
    word_freq = nltk.FreqDist()
    X_tok = list()
    current_sentence = list()
    
    i = 0
    for token in all_tokens:
        if token == '____':
            i += 1
            X_tok.append(current_sentence)
            current_sentence = list()
        else:
            current_sentence.append(token)
            word_freq[token] += 1
            
    X_tok.append(current_sentence)
    
    return X_tok, word_freq


def tokenize_space_sentences(X):
    """Tokenizes a list of sentences breaking them on spaces.
    
    :return (a,b) where <a> is A list of list of tokens and <b> is
    the word frequency on the dataset.
    """
    
    word_freq = nltk.FreqDist()
    
    X_tok = list()
    for sentence in X:
        words = sentence.split()
        X_tok.append(words)
        for word in words:
            word_freq[word] += 1
        
    return X_tok, word_freq


def compute_weights(model, alpha):
    weight_table = dict()
    total_count = 0

    for word, stats in model.vocab.items():
        total_count += stats.count
        weight_table[word] = stats.count

    weight_table = {word: alpha / (alpha + count / total_count) for (word, count) in weight_table.items()}
    return weight_table


def tok_sentence_to_vec(tok_sentences, vocabulary, model, normalize_sentence=1,
                        normalize_word=True, alpha=1e-4, show_logs=0):
    """
    Maps a list of tokenized sentences to their vectorial representation.
    > tok_sentences: List of tokenized sentences
    > vocabulary: List of words that should not be skipped
    > model: The word vector model
    > normalize_sentence: 0=(nothing), 1=(1/|s|), 2=(1/s + unit_length),
      3=(SIF), 4=(SIF + unit_length)
    > normalize_word: If words should be mapped to unit lenght
    > alpha: used only by SIF smoothing
    > show_logs: 0=(silent), 1=(basic), 2=(verbose)
    
    < X a n-by-m (n=#samples, m=#features) matrix.
    """

    # Precomputing the SIF table
    if normalize_sentence > 2:
        weights_table = compute_weights(model, alpha)
    else:
        weights_table = None
    
    # forcing the model to precompute normalized vectors
    model.most_similar('the');
    
    n = len(tok_sentences)
    m = model.vector_size
    X = np.zeros((n, m))
    
    oov_words = defaultdict(int)
    oov_sentences = 0
    
    for i, sentence in enumerate(tok_sentences):
        sv = np.zeros(m)
        amount_words = 0
        
        for word in sentence:
            # if stopword or unknown word, skip it
            if word not in vocabulary or word not in model.wv.vocab:
                oov_words[word] += 1
                continue
            
            amount_words += 1
            if normalize_sentence > 2:
                # if using SIF smoothing, weight each word individually
                sv += model.wv.word_vec(word, normalize_word) * weights_table[word]
            else:
                sv += model.wv.word_vec(word, normalize_word)
            
        if amount_words > 0:
            # sentence_vec := 1/|sentence| * sentence_vec
            if normalize_sentence > 0:
                sv = sv / amount_words
        else:
            oov_sentences += 1
            if show_logs == 2:
                logging.warning('Sentence <{}> has no representation'.format(sentence))
            
        X[i, :] = sv
        
    # Arora et al. "A simple but tough-to-beat baseline for sentence embeddings"
    if normalize_sentence in [3, 4]:
        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
        svd.fit(X)
        U = svd.components_
        X = X - X.dot(U.transpose()) * U

    if normalize_sentence in [2, 4]:       
        x_norms = np.linalg.norm(X, axis=1)[:, None]  # making this a (m, 1) vector
        x_norms[np.isclose(x_norms, 0)] = 1           # avoiding division by zero
        X = X / x_norms                               # normalizing
    
    if show_logs > 0:
        logging.warning('{} sentences without representation.'.format(oov_sentences))
        logging.warning('{} OOV words'.format(len(oov_words)))
    
    if show_logs == 2:
        s = sorted(oov_words.items(), key=lambda x:x[1], reverse=True)
        for word, freq in s:
            print('{} - {}'.format(word, freq))
        
    return X


def map_cluster_to_label(y, y_hat, show_logs=False, noise_label=-1):
    mapped = np.zeros(y_hat.shape)
    cluster_map = dict()
    
    for i_cluster in set(y_hat):
        # finding all samples on the i-th cluster
        indices = (y_hat == i_cluster)
        clustered_samples = y[indices]
        
        # if the classifier identifies, noise, skip its label mapping
        # otherwise make the most common class conquer the cluster
        if i_cluster == noise_label:
            most_common = noise_label
        else:
            most_common = mode(clustered_samples)[0][0]
           
        # make all samples on this cluster belong to the most common class.
        mapped[indices] = most_common
            
        if show_logs:
            logging.info('Cluster {} most common label {} with {} samples'
                         .format(
                             i_cluster, most_common,
                             len(clustered_samples)
                         ))

        # map cluster_id -> winning class
        cluster_map[i_cluster] = most_common

        
    return cluster_map, mapped


def keep_common_words(word_freq, min_word_freq):
    """
    Prunes the vocabulary to contain just words that appear at
    least <min_word_freq> times.
    > word_freq: Table mapping word to its frequency on the corpus
    > min_word_freq: Words with occurrence smaller than this value are removed.
    < new vocabulary with stopwords removed as well.
    """
    
    vocabulary_size = len(word_freq)
    stop_words = set(stopwords.words('english'))
    stop_words.update(["'t", "'ll", "'re", "'ve", "'m"])

    index_cut = None
    
    # from the least frequent, find the first word that occurs at
    # least <min_word_freq> times.
    for i, u in enumerate(sorted(word_freq.items(), key=lambda x:x[1])):
        if u[1] == min_word_freq:
            index_cut = vocabulary_size - i
            break
    else:
        index_cut = 0

    # getting just the words with the desired min_occurrence
    vocab_to_keep = word_freq.most_common(index_cut)
    vocab_to_keep = set(map(lambda x: x[0], vocab_to_keep))

    vocab_to_keep = vocab_to_keep - stop_words
    
    new_vocab_size  = len(vocab_to_keep)

    logging.info('Vocab size: {:6}'.format(vocabulary_size))
    logging.info('Keeping:    {:6} ({:4.4}%)'.format(new_vocab_size, new_vocab_size*100 / vocabulary_size))
    
    return vocab_to_keep

