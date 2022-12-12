cd ~/input/data

# full17 다운로드 코드
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=10199HFlZyTcq5jn9UoNYyangXaGp13YV' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10199HFlZyTcq5jn9UoNYyangXaGp13YV" \
-O dataset.tar && rm -rf ~/cookies.txt
tar -xvf dataset.tar