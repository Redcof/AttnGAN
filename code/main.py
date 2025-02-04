from __future__ import print_function
from dotenv import load_dotenv
load_dotenv('.env')  # take environment variables from .env.
import logging

from dotenv import load_dotenv
from torch.utils.data import DataLoader

from logger import logger, attach_file_to_logger
from miscc.utils import mkdir_p
from mlflow_utils import except_hook, AspectResize, stop_tracking, start_tracking

load_dotenv('..env')  # take environment variables from .env.
from miscc.config import cfg, cfg_from_file
from datasets import TextDataset
from trainer import condGANTrainer as trainer
from sixray_dataset import SixrayDataset
import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird_attn2.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def gen_example(wordtoix, algo):
    '''generate images from example sentences'''
    from nltk.tokenize import RegexpTokenizer
    filepath = '%s/example_filenames.txt' % (cfg.DATA_DIR)
    data_dic = {}
    with open(filepath, "r") as f:
        filenames = f.read().decode('utf8').split('\n')
        for name in filenames:
            if len(name) == 0:
                continue
            filepath = '%s/%s.txt' % (cfg.DATA_DIR, name)
            with open(filepath, "r") as f:
                logger.info('Load from: %s' % name)
                sentences = f.read().decode('utf8').split('\n')
                # a list of indices for a sentence
                captions = []
                cap_lens = []
                for sent in sentences:
                    if len(sent) == 0:
                        continue
                    sent = sent.replace("\ufffd\ufffd", " ")
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(sent.lower())
                    if len(tokens) == 0:
                        logger.info('sent %s' % sent)
                        continue
                    
                    rev = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0 and t in wordtoix:
                            rev.append(wordtoix[t])
                    captions.append(rev)
                    cap_lens.append(len(rev))
            max_len = np.max(cap_lens)
            
            sorted_indices = np.argsort(cap_lens)[::-1]
            cap_lens = np.asarray(cap_lens)
            cap_lens = cap_lens[sorted_indices]
            cap_array = np.zeros((len(captions), max_len), dtype='int64')
            for i in range(len(captions)):
                idx = sorted_indices[i]
                cap = captions[idx]
                c_len = len(cap)
                cap_array[i, :c_len] = cap
            key = name[(name.rfind('/') + 1):]
            data_dic[key] = [cap_array, cap_lens, sorted_indices]
    algo.gen_example(data_dic)


if __name__ == "__main__":
    sys.excepthook = except_hook
    logger.setLevel(logging.DEBUG)  # logger
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False
    
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    logger.info('Using config: %s' % cfg)
    
    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = 47
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    # ############################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = 'output/ATTGAN_%s_%s_%s' % \
                 (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
    cfg.OUTPUT_DIR = output_dir
    cfg.log_file = os.path.join(output_dir, "log.txt")
    mkdir_p(output_dir)
    attach_file_to_logger(cfg.log_file)
    # ###########################################################
    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        # b-shuffle = False
        split_dir = 'test'
    
    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    batch_size = cfg.TRAIN.BATCH_SIZE
    if cfg.DATASET_NAME == "birds":
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])
        dataset = TextDataset(cfg.DATA_DIR, split_dir,
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
        model_io_signature = None
        assert dataset
    elif cfg.DATASET_NAME == "sixray":
        image_transform = transforms.Compose([
            AspectResize(int(imsize * 76 / 64)),
            # transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])
        dataset = SixrayDataset(cfg.DATA_DIR, split_dir,
                                base_size=cfg.TREE.BASE_SIZE,
                                transform=image_transform)
        model_io_signature = None
        assert dataset
    
    else:
        logger.error("NotImplementedError(cfg.DATASET_NAME = '%s')" % cfg.DATASET_NAME)
        raise NotImplementedError("cfg.DATASET_NAME = '%s'" % cfg.DATASET_NAME)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True,
                            shuffle=True, num_workers=int(cfg.WORKERS))
    try:
        start_tracking()
        # Define models and go to train/evaluate
        algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword)
        
        start_t = time.time()
        if cfg.TRAIN.FLAG:
            # model training
            algo.train()
        else:
            '''Generate images from pre-extracted embeddings'''
            if cfg.B_VALIDATION:
                algo.sampling(split_dir)  # generate images for the whole valid dataset
            else:
                gen_example(dataset.wordtoix, algo)  # generate images for customized captions
        end_t = time.time()
        logger.info('Total time for training: %d' % (end_t - start_t))
        stop_tracking()
    except Exception as e:
        logger.info('Exiting from training with exception')
        logger.exception(e)
        stop_tracking("Exception: %s" % (str(e)))
    logger.info("Output saved at: '%s'" % cfg.OUTPUT_DIR)
