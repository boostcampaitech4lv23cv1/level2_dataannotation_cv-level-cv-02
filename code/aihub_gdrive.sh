# AIHUB_real_final_easy(500개 다운로드) :: 기존 aihub_easy 는 제거!
cd /opt/ml/input/data
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1jkyzJ0_U7dBjNXGjx5VOaPwVGBYDzLUm' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1jkyzJ0_U7dBjNXGjx5VOaPwVGBYDzLUm" \
-O dataset.tar && rm -rf ~/cookies.txt
tar -xvf dataset.tar

# # AIHub 세로간판 다운로드(Test 100개) 코드
# cd /opt/ml/input/data
# wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Wxjml_UkErscD60KdfO5FOksAeYiX4sQ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Wxjml_UkErscD60KdfO5FOksAeYiX4sQ" \
# -O dataset.tar && rm -rf ~/cookies.txt
# tar -xvf dataset.tar

# # AIHub 세로간판 다운로드(Train 1000개) 코드 :: fine-tuning 용
# cd /opt/ml/input/data
# wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hFZdXTckX95ZmQy2nHgO6ym7aIsLfikE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hFZdXTckX95ZmQy2nHgO6ym7aIsLfikE" \
# -O dataset.tar && rm -rf ~/cookies.txt
# tar -xvf dataset.tar

# # AIHub Full ver(Train 3000개) 코드 :: full-training 용
# cd /opt/ml/input/data
# wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1YSUwcfDLqJa9bfFn5eFXgrFJ9kyr32p3' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1YSUwcfDLqJa9bfFn5eFXgrFJ9kyr32p3" \
# -O dataset.tar && rm -rf ~/cookies.txt
# tar -xvf dataset.tar