#base.py에서는 여러 py에서 쓰이는 공통적인 모듈 상수들을 정의해놓는다.

TOKEN_TO_PATH = {
    "ko17" : "ICDAR17_Korean" ,
    "ko19" : "ICDAR19_Korean" ,
    "full17": "ICDAR17" , 
    "full19" : "ICDAR19",
    "camper" : "boostcamp" , 
    "aihub"  : "aihub"}

DATASETS_TO_USE = ["ko17", "full19"]