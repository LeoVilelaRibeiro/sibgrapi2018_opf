# Unsupervised Dialogue Act Classification with Optimum-Path Forest

This repository contains the code to generate the preprocessed samples and visualize statistics the data used on the paper _"Unsupervised Dialogue Act Classification with Optimum-Path Forest"_.

## Datasets

The following datasets are considered on the paper:

| Name          | Download link                                           |
| ------------- | ------------------------------------------------------- |
| Switchboard   | http://compprag.christopherpotts.net/swda.html#download |
| ICSI          | http://www1.icsi.berkeley.edu/%7Eees/dadb               |
| HCRC Map Task | http://groups.inf.ed.ac.uk/maptask/maptasknxt.html      |
| NPS           | We use the version avaible through the `nltk` package   |

It is possible to obtain each dataset used on the experiments by running the script `download.sh` inside each folder on the directory `datasets/`. To clean samples according to the procedure described on the paper, please execute the corresponding dataset notebook `clean_[dataset_name].ipynb`. This will create appropriate entries on the folder `clean/` using the naming convention `[dataset_name]_[partition].tsv` following a stardard format. For the NPS dataset, running `clean_nps.ipynb` will also download it.

**NOTE:** To parse the SwDA dataset the helper functions developed by [Christopher Potts](http://compprag.christopherpotts.net/swda.html#downloa) are used.

## Setting up the environment

Before generating the data splits, it is necessary to have Stanford GloVe and CoreNLP placed on the required folders and to have the proper version of `LibOPF` installed. This latter step is not necessary if you just want to obtain a copy of the data used to run the experiments.

GloVe can be obtained by running the script `vsms/get_glove.sh`. Please notice this will download a 2.1Gb file and then convert it to the Gensim binary format, which may take some space and time. Next, CoreNLP (we use version 3.9.0) can be downloaded through `get_corenlp.sh`.

Instead of downloading the canonical version of [LibOPF](https://github.com/jppbsi/LibOPF), we recommend using the code from the branch  `deep` from this [alternative repository](https://github.com/lzfelix/LibOPF), which contains the M-OPF implementation. Simple instructions on how to install the OPF framework are provided on the [original repository wiki](https://github.com/jppbsi/LibOPF/wiki/Installation). Following, you might want to consider creating an environment variable `$OPF_PATH` pointing to  LibOPF's `bin/` directory.

## Performing the feature extraction procedure

Within the `experiments/` folder, each dataset has a corresponding feature extraction notebook and a storage folder named `compute_[dataest_name]_features.ipynb` and `[dataset_name]_opf/`, respectivelly. By running the notebook, its related folder will be populated with an OPF-formatted text file that describes the dataset. The first line of this file contains three numbers that correspond to the amount of samples `m`, classes `C` and size of the feature vectors `n`. Each of the next `m` lines contain the corresponding sample ID, its ground-truth label and the `n`-dimensional sentence vector.

## Generating the folds

The folds reported in the paper were generated using [opfy_split](https://github.com/lzfelix/opfy_split) `opfy_opf` and  `opfy_numpy` utilities. The following example snippet shows how to generate the folds for the Map Task dataset:

```bash
$OPF_PATH/../tools/txt2opf maptask_samples.txt maptask_samples.dat
opfy_opf maptask_samples.dat -k=15
opfy_numpy .
```

## References

To be added in the future.