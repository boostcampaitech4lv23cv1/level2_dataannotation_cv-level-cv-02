#!/usr/bin/env bash

echo "하단의 주석친 부분은 원격 서버에서 ICDAR19를 받아오는 코드입니다. 구글 드라이브를 이용할 시 이 파일은 필요 없습니다.."

mkdir -p "../input/data/ICDAR19"
cd "../input/data/ICDAR19"
echo "https://datasets.cvc.uab.es/rrc/ImagesPart1.zip" >> download.txt
echo "https://datasets.cvc.uab.es/rrc/ImagesPart2.zip" >> download.txt

for url in $(cat download.txt | tr -d '\r')
do
    wget $url --no-check-certificate
done
for i in *.zip
do
  unzip $i -d "/opt/ml/input/data/ICDAR19/raw_images"
done
wget https://datasets.cvc.uab.es/rrc/train_gt_t13.zip --no-check-certificate
unzip train_gt_t13.zip -d "/opt/ml/input/data/ICDAR19/gt"

mv /opt/ml/input/data/ICDAR19/raw_images/ImagesPart1/* /opt/ml/input/data/ICDAR19/raw_images
mv /opt/ml/input/data/ICDAR19/raw_images/ImagesPart2/* /opt/ml/input/data/ICDAR19/raw_images

echo  "추후 파이썬 파일 실행을 위하여 디렉토리를 code로 바꿔줍니다.\n"
cd "/opt/ml/code"

echo  "Converting ICDAR19 with only korean\n"
python convert_mlt_19.py --only_korean "true"

echo  "Converting ICDAR19 with full data. only korean 지정 안해야 돌아갑니다.\n"
python convert_mlt_19.py 

echo  "ICDAR19 데이터셋의 무결성을 검사합니다. 데이터를 고치기 원한다면 --mode fix를 인자로 넣어주세요.\n"
python integrity_check.py --data_name ko19 #--mode fix
python integrity_check.py --data_name full19 #--mode fix

echo  "이제, 수정된 데이터셋을 train/random으로 나누겠습니다. {data_name}/ufo/random_split에 train.json, val.json이 저장됩니다. \n"
python split_dataset.py --data_name ko19
python split_dataset.py --data_name full19

echo "잔여 파일들을 삭제하겠습니다."
cd "../input/data/ICDAR19"
rm -rf gt
rm -rf raw_images

echo "다운로드 시간이 오래 걸리므로, 혹시 몰라 .zip 파일을 남겨놓습니다. 이를 제거하고 싶을시 rm *.zip을 해주세요"
#rm *.zip
