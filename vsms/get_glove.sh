echo "Downloading and extracting GloVe"
wget -O glove.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.zip
rm glove.zip

echo "Generating gensim representation"
pip install gensim
python -m gensim.scripts.glove2word2vec -i glove.840B.300d.txt -o wglove.840B.300d.txt

echo "================================"
echo "Generating gensim representation"
echo "================================"

python binarize_glove.py
rm wglove.840B.300d.txt

echo "Done"

