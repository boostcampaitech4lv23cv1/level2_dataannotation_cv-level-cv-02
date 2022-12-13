# # ko19 다운로드 코드
cd ~/input/data
# ko19 다운로드 코드
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wi66oXnSPEyo5weSGHz1MpTjq3hChtIN' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wi66oXnSPEyo5weSGHz1MpTjq3hChtIN" \
-O dataset.tar && rm -rf ~/cookies.txt
tar -xvf dataset.tar
rm dataset.tar

# full19 다운로드 코드
cd ~/input/data
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1tNm1wjNTHLYqtyUX9hsxnQy6HhuVDyrm' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1tNm1wjNTHLYqtyUX9hsxnQy6HhuVDyrm" \
-O dataset.tar && rm -rf ~/cookies.txt
tar -xvf dataset.tar
rm dataset.tar

#aihub 세로간판
cd ~/input/data
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1p_ZUb_UcGScnSOjA5VphKweqSqFpf6ux' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1p_ZUb_UcGScnSOjA5VphKweqSqFpf6ux" \
-O dataset.tar && rm -rf ~/cookies.txt
tar -xvf dataset.tar
rm dataset.tar