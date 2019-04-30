import json
import os

def create_indexing(split):
    file_path = {}
    for subdir in os.listdir('./nlvr/{}/images'.format(split)):
        if os.path.isdir(os.path.join('./nlvr/{}/images'.format(split))):
            path = os.path.join('./nlvr/{}/images'.format(split), subdir)
            # For each image in the subdirectory,
            for fn in os.listdir(path):
                if not fn.endswith('.png'): continue
                file_path[fn] = os.path.join('./nlvr/{}/images'.format(split), subdir,fn)
    with open('./nlvr/{}/{}_indexing.txt'.format(split,split), 'w') as outfile:  
        json.dump(file_path, outfile)

if __name__ =='__main__':
    create_indexing('train')
    create_indexing('dev')
    