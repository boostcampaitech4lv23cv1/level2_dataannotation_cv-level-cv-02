#base.py에서는 여러 py에서 쓰이는 공통적인 모듈 상수들을 정의해놓는다.

TOKEN_TO_PATH = {
    "ko17" : "ICDAR17_Korean" ,
    "ko19" : "ICDAR19_Korean" ,
    "full17": "ICDAR17" , 
    "full19" : "ICDAR19",
    "camper" : "boostcamp" , 
    "aihub"  : "aihub",
    "aihub_small" : "aihub_small" , 
    "aihub_test" : "aihub_test_100",
    "eng17" : "ICDAR17_English",
    "aihub_easy" : "aihub_final_easy"
    }

DATASETS_TO_USE = ["ko17", "ko19", "eng17", "aihub_easy"]


# Test Experiment시 하단 식으로 세팅을 해주시기 바랍니다.
# DATASETS_TO_USE = ["ko17", "ko19", "eng17", "aihub_test"]
