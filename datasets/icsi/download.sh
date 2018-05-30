# downloading
wget -O icsi.tar.gz http://www.icsi.berkeley.edu/~ees/dadb/icsi_mrda+hs_corpus_050512.tar.gz
tar -zxf icsi.tar.gz

mv icsi_mrda+hs_corpus_050512/* .
rm -rf icsi_mrda+hs_corpus_050512/
rm icsi.tar.gz

