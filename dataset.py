import os
import pickle

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import h5py
import json
from transforms import Scale
from torch.utils.data import DataLoader


transform = transforms.Compose([
    
    transforms.ToTensor(),
    
])


class NLVR(Dataset):
    def __init__(self, root, split='train', transform= transform, perm=str(0)):
        with open(f'data/{split}.pkl', 'rb') as f:
            self.questions = pickle.load(f)
        self.perm = perm
        self.transform = transform
        self.root = root
        self.split = split
        #self.h = h5py.File('nlvr/{}/{}_features.h5'.format(split,split), 'r')
        #self.img = self.h['features']
        # loading indexing file
        with open(os.path.join(self.root, self.split,'{}_indexing.txt'.format(self.split)), 'r+') as f:
            self.indexing = json.loads(f.read())
        print('indexing length',len(self.indexing))
        
              
                    

    def close(self):
        self.h.close()

    def __getitem__(self, index):
        identifier, question, answer = self.questions[index]

        #img = self.transform(img)
        perm = self.perm
        img_name = self.split + '-'+ identifier + '-'+ perm + '.png'
        img_path =  self.indexing[img_name]
        
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, question, len(question), answer

    def __len__(self):
        return len(self.questions)

class CLEVR(Dataset):
    def __init__(self, root, split='train', transform=None):
        with open(f'data/{split}.pkl', 'rb') as f:
            self.data = pickle.load(f)

        # self.transform = transform
        self.root = root
        self.split = split

        self.h = h5py.File('data/{}_features.hdf5'.format(split), 'r')
        self.img = self.h['data']

    def close(self):
        self.h.close()

    def __getitem__(self, index):
        imgfile, question, answer, family = self.data[index]
        # img = Image.open(os.path.join(self.root, 'images',
        #                            self.split, imgfile)).convert('RGB')

        
        id = int(imgfile.rsplit('_', 1)[1][:-4])
        img = Image.open(self.img[id]).convert('RGB')
        img = self.transform(img)
        

        return img, question, len(question), answer, family

    def __len__(self):
        return len(self.data)



def collate_data(batch):
    images, lengths, answers  = [], [], []
    batch_size = len(batch)

    max_len = max(map(lambda x: len(x[1]), batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

    for i, b in enumerate(sort_by_len):
        image, question, length, answer= b
        images.append(image)
        length = len(question)
        questions[i, :length] = question
        lengths.append(length)
        answers.append(answer)
        
    return torch.stack(images), torch.from_numpy(questions), \
        lengths, torch.LongTensor(answers)



### TEST

# if __name__ =='__main__':
#     for subdir in os.listdir('./nlvr/{}/images'.format(self.split)):
#             if os.path.isdir(os.path.join('./nlvr/{}/images'.format(self.split))):
#                 path = os.path.join('./nlvr/{}/images'.format(self.split), subdir)
#                 # For each image in the subdirectory,
#                 for fn in os.listdir(path):
#                     if not fn.endswith('.png'): continue
#                     self.file_path[fn] = os.join('./nlvr/{}/images'.format(split), subdir,fn)
#         with open('/nlvr/{}/indexing.txt'.format(self.split), 'w') as outfile:  
#             json.dump(self.file_path, outfile)
    
if __name__ =='__main__':
    nlvr = NLVR('./nlvr')
    train_set = DataLoader(
        nlvr, batch_size=3, num_workers=4, collate_fn=collate_data
    )
    tmp = 0
    
    for i, (img, question, lenquestion, answer) in enumerate(train_set):
        if i>=10:
            break
        print(img.type() ,lenquestion)

