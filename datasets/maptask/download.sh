# downloading
wget -O maptask.zip http://groups.inf.ed.ac.uk/maptask/hcrcmaptask.nxtformatv2-1.zip
wget -O nite.zip https://sourceforge.net/projects/nite/files/nite/nxt_1.4.4/nxt_1.4.4.zip/download

unzip maptask.zip
unzip nite -d nite/

# compiling the transcriptions
NXT="$(pwd)/nite"
export CLASSPATH=".:$NXT:$NXT/lib:$NXT/lib/nxt.jar:$NXT/lib/jdom.jar:$NXT/lib/JMF/lib/jmf.jar:$NXT/lib/pnuts.jar:$NXT/lib/resolver.ja
r:$NXT/lib/xalan.jar:$NXT/lib/xercesImpl.jar:$NXT/lib/xml-apis.jar:$NXT/lib/jmanual.jar:$NXT/lib/jh.jar:$NXT/lib/helpset.jar:$NXT/lib
/poi.jar:$NXT/lib/eclipseicons.jar:$NXT/lib/icons.jar:$NXT/lib/forms.jar:$NXT/lib/looks.jar:$NXT/lib/necoderHelp.jar:$NXT/lib/videola
belerHelp.jar:$NXT/lib/dacoderHelp.jar:$NXT/lib/testcoderHelp.jar:$NXT/lib/prefuse.jar"

cd nite/lib
java SortedOutput -corpus ../../maptaskv2-1/Data/maptask.xml -q '($m move):' -t -atts who label > ../../all_transcripts.txt

# cleaning up
cd ../../
rm -rf maptaskv2-1/
rm -rf nite/
rm maptask.zip
rm nite.zip

