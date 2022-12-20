from shapely.geometry import Polygon
import json
import numpy as np
import unicodedata
import copy

aihub_path = '/opt/ml/vertical.json'
google_path = '/opt/ml/input/data/aihub/ufo/train.json'
out_path = '/opt/ml/combine1.json'

a = open(aihub_path, 'r')
g = open(google_path, 'r')

a_json = json.load(a)['images']
g_json = json.load(g)['images']

ufo={}
out_json = copy.deepcopy(g_json)

for file in g_json:
    g_words = g_json[file]['words']
    a_file = unicodedata.normalize('NFC', file)
    del_flag = False
    for g_word in g_words:
        p1 = Polygon(np.array(g_words[g_word]['points']))
        g_area = p1.area
        try:
            a_words = a_json[a_file]['words']
        except:
            del(out_json[file])
            del_flag=True
            break
        if del_flag: break
        for idx, a_word in enumerate(a_words, start=5001):
            if a_words[a_word]['transcription'] !='xxx':
                out_json[file]['words'][str(idx)]=a_words[a_word]

        for a_word in a_words:
            if a_words[a_word]['transcription'] =='xxx': continue
            p2 = Polygon(np.array(a_words[a_word]['points']))
            a_area = p2.area
            inter_area = p1.intersection(p2).area

            if g_area!=0.0 and inter_area/g_area >= 0.8:
                del(out_json[file]['words'][g_word])
                break

ufo['images'] = out_json
with open(out_path, 'w') as outfile:
    json.dump(ufo, outfile)








a.close()
g.close()