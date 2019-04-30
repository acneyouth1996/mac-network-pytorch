import sys
import pickle
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CLEVR, collate_data, transform, NLVR
from model import MACNetwork
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', required = True)
args = parser.parse_args()

batch_size = 2
n_epoch = 3

dim = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def train(epoch):
    moving_loss = 0
    for i in range(6):
        nlvr = NLVR(args.root, transform=transform, perm = str(i))
        
        train_set = DataLoader(
            nlvr, batch_size=batch_size, num_workers=4, collate_fn=collate_data
        )

        dataset = iter(train_set)
        pbar = tqdm(dataset)
        

        net.train(True)
        for image, question, q_len, answer in pbar:
            image, question, answer = (
                image.to(device),
                question.to(device),
                answer.to(device),
            )

            net.zero_grad()
            output = net(image, question, q_len)
            loss = criterion(output, answer)
            loss.backward()
            optimizer.step()
            correct = output.detach().argmax(1) == answer
            correct = torch.tensor(correct, dtype=torch.float32).sum() / batch_size

            if moving_loss == 0:
                moving_loss = correct

            else:
                moving_loss = moving_loss * 0.99 + correct * 0.01

            pbar.set_description(
                'Epoch: {}; Loss: {:.5f}; Acc: {:.5f}'.format(
                    epoch + 1, loss.item(), moving_loss
                )
            )

            accumulate(net_running, net)

    


def valid(epoch):
    correct_num = 0
    total = 0
    for i in range(6):
        nlvr = NLVR(args.root, 'dev', transform=transform, perm = str(i))
        valid_set = DataLoader(
            nlvr, batch_size=batch_size, num_workers=4, collate_fn=collate_data
        )
        dataset = iter(valid_set)

        net_running.train(False)
        
        with torch.no_grad():
            for image, question, q_len, answer in tqdm(dataset):
                image, question = image.to(device), question.to(device)

                output = net_running(image, question, q_len)
                correct = output.detach().argmax(1) == answer.to(device)
                
                for c in correct:
                    if c:
                        correct_num += 1
                    total += 1

        with open('log/log_{}.txt'.format(str(epoch + 1).zfill(2)), 'w') as w:
            w.write('{:.5f}\n'.format(correct_num/total))

        print(
            'Avg Acc: {:.5f}'.format(
                correct_num/total
            )
        )

    

if __name__ == '__main__':
    with open('data/dic.pkl', 'rb') as f:
        dic = pickle.load(f)

    n_words = len(dic['word_dic']) + 1
    n_answers = len(dic['answer_dic'])

    net = MACNetwork(n_words, dim).to(device)
    net_running = MACNetwork(n_words, dim).to(device)
    accumulate(net_running, net, 0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    for epoch in range(n_epoch):
        valid(epoch)
        train(epoch)
        

        with open(
            'checkpoint/checkpoint_{}.model'.format(str(epoch + 1).zfill(2)), 'wb'
        ) as f:
            torch.save(net_running.state_dict(), f)
