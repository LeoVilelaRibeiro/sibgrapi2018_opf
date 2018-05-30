from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('./wglove.840B.300d.txt')
model.save('wglove.840B.300d.bin')

