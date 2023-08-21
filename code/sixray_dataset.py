import csv
import os
import pickle
import random
from collections import defaultdict

import mlflow
import numpy as np
import pandas as pd
from mlflow.types import Schema, TensorSpec
from nltk import RegexpTokenizer
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import get_imgs
from logger import logger
from miscc.config import cfg
from mlflow_utils import stop_tracking


class SixrayDataset(Dataset):
    def __init__(self, data_dir, split='train', base_size=64, transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE
        
        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2
        
        self.data = []
        self.data_dir = data_dir
        
        self.split_dir = os.path.join(data_dir, split)
        
        self.filenames, self.captions, self.ixtoword, self.wordtoix, self.n_words = self.load_text_data(data_dir, split)
        
        self.class_id = self.load_class_id(self.data_dir, ["gun", "knife", "gun-knife"])
        self.number_example = len(self.filenames)
    
    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            
            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox
    
    def load_captions(self, data_dir, split, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/%s/captions/%s.txt' % (data_dir, split, filenames[i])
            with open(cap_path, "r", encoding='utf8') as f:
                captions = [line for line in f.read().split('\n') if line.strip() != ""]
                cnt = 0
                if len(captions) < 5:
                    # if number of captions are less than 5, copy the first caption
                    captions.append(captions[0])
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue
                    
                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    logger.info('ERROR: the captions for %s less than %d' % (filenames[i], cnt))
        return all_captions
    
    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1
        
        vocab = [w for w in word_counts if word_counts[w] >= 0]
        
        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1
        
        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)
        
        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)
        
        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]
    
    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, 'train', train_names)
            test_captions = self.load_captions(data_dir, 'test', test_names)
            
            train_captions, test_captions, ixtoword, wordtoix, n_words = self.build_dictionary(
                train_captions, test_captions
            )
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        return filenames, captions, ixtoword, wordtoix, n_words
    
    def load_class_id(self, data_dir, classes):
        if os.path.isfile(data_dir + '/labels.csv'):
            # a csv file contains filename and class-name
            with open(data_dir + '/labels.csv', 'r') as f:
                reader = csv.reader(f, delimiter=",")
                class_id = {row[0].split(".")[0]: classes.index(row[1].strip()) for row in reader if
                            row[1].strip() in classes}
        return class_id
    
    def load_filenames(self, data_dir, split):
        caption_dir = os.path.join(data_dir, split, "JPEGImages")
        filenames = [cap_file.split(".")[0]
                     for cap_file in os.listdir(caption_dir) if cap_file.endswith('.jpg')]
        return filenames
    
    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            logger.info('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len
    
    def __getitem__(self, index):
        #
        filename_only = self.filenames[index]
        cls_id = self.class_id[filename_only]
        data_dir = self.split_dir
        #
        img_name = '%s/JPEGImages/%s.jpg' % (data_dir, filename_only)
        imgs = get_imgs(img_name, self.imsize,
                        None, self.transform, normalize=self.norm)
        # random select a sentence
        sent_ix = random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix
        try:
            caps, cap_len = self.get_caption(new_sent_ix)
        except IndexError as e:
            logger.debug(
                "INDEX=%d, SENT_IX=%d, IDX=%d, FILENAMES=%d, CAPTIONS:%d" %
                (index, sent_ix, new_sent_ix, len(self.filenames), len(self.captions)))
            logger.exception(e)
            stop_tracking(exit_message=str(e))
            raise e
        return imgs, caps, cap_len, cls_id, filename_only
    
    def __len__(self):
        return len(self.filenames)
