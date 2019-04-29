import os
import sys
import json
import pickle

import nltk
import tqdm
from torchvision import transforms
from PIL import Image
from transforms import Scale
nltk.download('punkt')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_questions', required = True)


def process_question(root, split, word_dic=None, answer_dic=None):
    if word_dic is None:
        word_dic = {}

    if answer_dic is None:
        answer_dic = {}

    path = os.path.join(root, split,
                        '{}.json'.format(split))
    data = [json.loads(line) for line in open(path).readlines()]

    result = []
    word_index = 1
    answer_index = 0

    for item in tqdm.tqdm(data):
        question = item['sentence']
        words = nltk.word_tokenize(question)
        question_token = []

        for word in words:
            try:
                question_token.append(word_dic[word])

            except:
                question_token.append(word_index)
                word_dic[word] = word_index
                word_index += 1

        answer_word = item['label']

        try:
            answer = answer_dic[answer_word]

        except:
            answer = answer_index
            answer_dic[answer_word] = answer_index
            answer_index += 1

        result.append((item['identifier'], question_token, answer))

    with open('data/{}.pkl'.format(split), 'wb') as f:
        pickle.dump(result, f)

    return word_dic, answer_dic

if __name__ == '__main__':
    args = parser.parse_args()
    root = args.input_questions

    word_dic, answer_dic = process_question(root, 'train')
    process_question(root, 'val', word_dic, answer_dic)

    with open('data/dic.pkl', 'wb') as f:
        pickle.dump({'word_dic': word_dic, 'answer_dic': answer_dic}, f)