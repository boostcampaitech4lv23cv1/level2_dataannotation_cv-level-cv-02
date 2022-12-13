import os

path = '/opt/ml/input/data/aihub/images/'
f = open("/opt/ml/error_images.txt", 'r')
files = f.readlines()
for file in files:
    file = file.strip()
    file_path = os.path.join(path, file)
    os.remove(file_path)
    print(f'removed {file_path}')
f.close()