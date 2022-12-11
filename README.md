# Data_Annotation_competition

## METHOD
- ISSUE 기반의 협업

## Commit message format

```
feat: (brief-comment)
#Commit message number
```

# integrity_check.py

## 사용할 토큰은 base.py의 TOKENS_TO_PATH 참고
```
TOKEN_TO_PATH = {
    "ko17" : "ICDAR17_Korean" ,
    "ko19" : "ICDAR19_Korean" ,
    "full17": "ICDAR17" , 
    "full19" : "ICDAR19",
    "camper" : "boostcamp" , 
    "aihub"  : "aihub"}

DATASETS_TO_USE = ["ko17", "full19"]
```

### 데이터셋 정합성 검사
- 아직 empty_dataset 코드가 정확하지 않습니다. 확인 부탁드립니다.

>> python integrity_check.py --data_name ko19

### 데이터셋 다운로드
- 구글 드라이브에서 파일을 받아옵니다.(이 방식을 권장합니다.)
- 드라이브 최대 요청 수 증가 시, 오류가 날 것입니다.
- 이 스크립트를 실행시, gdrive에서 파일을 받아옵니다.
>> sh icdar19_gdrive.sh 

- 다음 스크립트는, 원격 ICDAR19 서버에서 데이터를 받아옵니다.
>> sh ICDAR19_download.sh 

### 데이터셋 나누기
- ratio를 지정하여 나눌 수 있습니다.
- 일례로, 다음 코드는 해당 데이터셋을 90%, 10%의 비율로 분할합니다. 

>> python split_dataset.py --data_name ko19 --split_ratio 0.9 