import os
import json
from PIL import Image
from PIL import ImageDraw

#https://yunwoong.tistory.com/148 참고하여 api key 발급
os.environ['GOOGLE_APPLICATION_CREDENTIALS']="/opt/ml/ocr_api.json"
ufo = {}
images = {}

input_path = '/opt/ml/aihub_test_100/tmp/'
bbox_path = '/opt/ml/aihub_test_100/bbox/'
output_path = '/opt/ml/aihub_test_100/train.json'
f = open("/opt/ml/error_images.txt", 'w')

def make_img_json(vertices, desc):
    word = {}
    v_list = []
    for v in vertices:
        tmp = [v.x, v.y]
        v_list.append(tmp)

    word["transcription"] = desc
    word["points"] = v_list
    word["orientation"] = "Horizontal"
    word["language"] = ["en"]
    word["tags"] = []
    word["confidence"] = None
    word["illegibility"] = False 
    return word


def detect_text(dir_path, file_name):
    """Detects text in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()
    path = os.path.join(dir_path, file_name)

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    pages = response.full_text_annotation.pages
    

    words = {}
    w=h=-1
    for a in pages:
        w = a.width
        h = a.height
    
    
    for i, text in enumerate(texts):
        if i==0 : continue 
        
        padding = str(i).zfill(4)
        words[padding] = make_img_json(text.bounding_poly.vertices, text.description)
        
        vertices = text.bounding_poly.vertices
        v_list = []
        for v in vertices:
            v_list.append((v.x, v.y))
        v_list.append((vertices[0].x, vertices[0].y))

        draw.line(v_list, fill="red", width=4)
    
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    return words, w, h

try:
    for i, file in enumerate(os.listdir(input_path)):
        err_file = file
        img = Image.open(os.path.join(input_path, file))
        draw = ImageDraw.Draw(img)
        words, w, h = detect_text(input_path, file)
        if w==-1:
            f.write(file)
        else:
            images[file] = {"words":words, "img_w":w, "img_h":h}
        img.save(os.path.join(bbox_path, "bbox_"+file))
        if(i%10==1): print(i, file)
    ufo["images"] = images
except Exception as e:
    print(err_file+ " :::: ", e)
    ufo["images"] = images
    with open(output_path, 'w') as outfile:
        json.dump(ufo, outfile)

with open(output_path, 'w') as outfile:
    json.dump(ufo, outfile)

f.close()