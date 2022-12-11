import os
import json

#https://yunwoong.tistory.com/148 참고하여 api key 발급
os.environ['GOOGLE_APPLICATION_CREDENTIALS']="/opt/ml/ocr_api.json"
ufo = {}

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
    
    for a in pages:
        size = (a.width, a.height) 
    
    for i, text in enumerate(texts):
        if i>0 : 
            padding = str(i).zfill(4)
            words[padding] = make_img_json(text.bounding_poly.vertices, text.description)
            
        v_list = []
        for v in text.bounding_poly.vertices:
            v_list.append((v.x, v.y))
        v_list.append((text.bounding_poly.vertices[0].x,text.bounding_poly.vertices[0].y))

    
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    return words, size


dir_path = '/opt/ml/input/data/boostcamp/tmp/'

images = {}
try:
    for i, file in enumerate(os.listdir(dir_path)):
        words, size = detect_text(dir_path, file)
        images[file] = {"words":words, "img_w":size[0], "img_h":size[1]}
        if(i%10==1): print(file)
    ufo["images"] = images
except Exception as e:
    print(e)
    with open("/opt/ml/output.json", 'w') as outfile:
        json.dump(ufo, outfile)

with open("/opt/ml/output.json", 'w') as outfile:
    json.dump(ufo, outfile)