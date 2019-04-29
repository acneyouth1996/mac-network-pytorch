
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse, os, json
import h5py
import numpy as np
from scipy.misc import imread, imresize
from tqdm import tqdm

import torch
import torchvision


parser = argparse.ArgumentParser()
parser.add_argument('--input_image_dir', required=True)
parser.add_argument('--max_images', default=None, type=int)
parser.add_argument('--output_h5_file', required=True)
parser.add_argument('--output_indexing_file', required=True)

parser.add_argument('--image_height', default=224, type=int)
parser.add_argument('--image_width', default=224, type=int)

parser.add_argument('--model', default='None')
parser.add_argument('--model_stage', default=3, type=int)
parser.add_argument('--batch_size', default=128, type=int)


def run_batch(cur_batch, model):
  image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
  return image_batch / 255.  # Scale pixel values to [0, 1]

def main(args):
  input_paths = []
  idx_set = set()

  # For each image subdirectory,
  for subdir in os.listdir(args.input_image_dir):
    if os.path.isdir(os.path.join(args.input_image_dir, subdir)):
      path = os.path.join(args.input_image_dir, subdir)
      # For each image in the subdirectory,
      for fn in os.listdir(path):
        if not fn.endswith('.png'): continue
        idx = fn.split(".")[0]
        idx = fn.split("-")[0:]
        idx = "".join(idx)
        # Save the original path paired with the index, which is just the numbers in the ID + .png
        input_paths.append((os.path.join(path, fn), idx))
        idx_set.add(idx)
  # Sort the paths alphabetically by their recomputed ID
  input_paths.sort(key=lambda x: x[1])
  print(len(idx_set))
  print(len(input_paths))
  assert len(idx_set) == len(input_paths)

  # Cut if off if only processing a certain amount of images
  if args.max_images is not None:
    print('Processing only ' + str(args.max_images) + ' images')
    input_paths = input_paths[:args.max_images]

  print('Saving a total of ' + str(len(input_paths)) + ' images')

  # Image size: what to resize the images to
  img_size = (args.image_height, args.image_width)
  indexing_file = {}
  with h5py.File(args.output_h5_file, 'w') as f:
    feat_dset = None
    i0 = 0
    cur_batch = []

    # For each image (still sorted by alphabitical recomputed ID)
    for i, (path, idx) in tqdm(enumerate(input_paths)):
      # Write the index to the indexing file
      indexing_file[str(idx)] = i
      #indexing_file.write(str(i) + "\t" + str(idx) + "\n")

      # Read in the image and resize it 
      img = imread(path, mode='RGB')
      img = imresize(img, img_size, interp='bicubic')
      img = img.transpose(2, 0, 1)[None]

      # If you're at a batch size to save, then run the batch (in this case, just normalize the pixel values between zero and one)
      # and save it to the h5
      cur_batch.append(img)
      if len(cur_batch) == args.batch_size:
        feats = run_batch(cur_batch, None)

        # If at the beginning, create the dataset and fill it with what was processed
        if feat_dset is None:
          N = len(input_paths)
          _, C, W, H = feats.shape
          feat_dset = f.create_dataset('features', (N, C, W,H),
                                       dtype=np.float32)
        i1 = i0 + len(cur_batch)
        feat_dset[i0:i1] = feats
        i0 = i1
        cur_batch = []

    # Save the last batch if there is one
    if len(cur_batch) > 0:
      feats = run_batch(cur_batch, None)
      i1 = i0 + len(cur_batch)
      feat_dset[i0:i1] = feats

  with open(args.output_indexing_file, 'w') as file:
      json.dump(indexing_file, file)
  return


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
