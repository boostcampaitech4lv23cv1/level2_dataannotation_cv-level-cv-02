#ICDAR17_English data
cd ../input/data
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Awc9LIrhid_SU0l9b8B757pyf3vVvmmT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Awc9LIrhid_SU0l9b8B757pyf3vVvmmT" \
-O dataset.tar && rm -rf ~/cookies.txt
tar -xvf dataset.tar
