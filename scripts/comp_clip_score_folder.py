# -*- coding: utf-8 -*-
import os
import argparse
import torch
import clip
from PIL import Image

# save a list to a txt
def text_save(content, filename, mode = 'a'):
    # try to save a list variable in txt file.
    # Use the following command if Chinese characters are written (i.e., text in the file will be encoded in utf-8)
    # file = open(filename, mode, encoding='utf-8')
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

# read a txt expect EOF
def text_readlines(filename, mode = 'r'):
    # try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        # Use the following command if there is Chinese characters are read
        # file = open(filename, mode, encoding='utf-8')
        file = open(filename, mode)
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i]) - 1]
    file.close()
    return content

# read a folder, return the complete path of all files
def get_files(path):
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

# multi-layer folder creation
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def define_formal_text_list(text_list, prefix = "a photo of"):
    output_list = []
    for i in range(len(text_list)):
        output_list.append(prefix + text_list[i])
    return output_list

def inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image_list = get_files(args.datapath)
    text_list = define_formal_text_list(text_readlines(args.filename))
    text = clip.tokenize(text_list).to(device)

    with torch.no_grad():
        # only inference once for texts
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim = -1, keep_dims = True)

        # loop images
        for i, image_name in enumerate(image_list):
            image = preprocess(image_name).unsqueeze(0).to(device)
            image_features = model.encode_image(image)
            image_features = image_features / image_features.norm(dim = -1, keep_dims = True)
        
            clip_scores = torch.matmul(image_features, text_features.T)
            clip_scores = clip_scores.cpu().numpy()[0]

            print(image_name + ':' + clip_scores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='./data')
    parser.add_argument('--filename', type=str, default='filename.txt', help='word list')
    args = parser.parse_args()

    inference(args)