# AIHUB_real_final_easy(500개 다운로드) :: 기존 aihub_easy 는 제거!

cd /opt/ml/input/data
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1jkyzJ0_U7dBjNXGjx5VOaPwVGBYDzLUm' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1jkyzJ0_U7dBjNXGjx5VOaPwVGBYDzLUm" \
-O dataset.tar && rm -rf ~/cookies.txt
tar -xvf dataset.tar