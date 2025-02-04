from __future__ import print_function
from dotenv import load_dotenv
load_dotenv('.env')  # take environment variables from .env.

import logging

import mlflow
from logger import logger, attach_file_to_logger
from mlflow_utils import start_tracking, stop_tracking, log_model, AspectResize, except_hook, can_i_log_model, \
    is_early_stop



from torch.utils.data import DataLoader

from miscc.utils import mkdir_p
from miscc.utils import build_super_images
from miscc.losses import sent_loss, words_loss
from miscc.config import cfg, cfg_from_file

from datasets import TextDataset
from datasets import prepare_data

from model import RNN_ENCODER, CNN_ENCODER

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from sixray_dataset import SixrayDataset

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

ARTIFACT_INTERVAL_GLOBAL_STEP = cfg.TRAIN.ARTIFACT_INTERVAL_GLOBAL_STEP


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/DAMSM/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def train(dataloader, cnn_model, rnn_model, batch_size,
          labels, optimizer, epoch_idx, ixtoword, image_dir):
    cnn_model.train()
    rnn_model.train()
    s_loss0 = 0
    s_loss1 = 0
    w_loss0 = 0
    w_loss1 = 0
    s_total_loss = 0
    w_total_loss = 0
    global_step = (epoch_idx + 1) * len(dataloader)
    start_time = time.time()
    batch_id = 0
    for batch_id, data in enumerate(dataloader, 0):
        global_step = epoch_idx * len(dataloader) + batch_id
        # print('batch_id', batch_id)
        rnn_model.zero_grad()
        cnn_model.zero_grad()
        
        imgs, captions, cap_lens, class_ids, keys = prepare_data(data)
        
        # words_features: batch_size x nef x 17 x 17
        # sent_code: batch_size x nef
        words_features, sent_code = cnn_model(imgs[-1])
        # --> batch_size x nef x 17*17
        nef, att_sze = words_features.size(1), words_features.size(2)
        # words_features = words_features.view(batch_size, nef, -1)
        
        hidden = rnn_model.init_hidden(batch_size)
        # words_emb: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)
        
        # word loss
        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels,
                                                 cap_lens, class_ids, batch_size)
        loss = w_loss0 + w_loss1
        w_total_loss += loss.data
        
        # sentence loss
        s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        loss += s_loss0 + s_loss1
        s_total_loss += loss.data
        #
        loss.backward()
        elapsed = time.time() - start_time
        try:
            mlflow.log_metrics(dict(
                train_batch_ms=elapsed,
                train_batch_s_loss0=s_loss0.item(),
                train_batch_s_loss1=s_loss1.item(),
                train_batch_w_loss0=w_loss0.item(),
                train_batch_w_loss1=w_loss1.item(),
                train_batch_s_loss=(s_loss0 + s_loss1).item(),
                train_batch_w_loss=(w_loss0 + w_loss1).item(),
            ), step=global_step)
        except Exception as e:
            logger.exception(e)
        logger.info('| Training epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} '
                    's_loss {:5.2f} {:5.2f} | '
                    'w_loss {:5.2f} {:5.2f}'
                    .format(epoch_idx, batch_id, len(dataloader),
                            elapsed,
                            s_loss0, s_loss1,
                            w_loss0, w_loss1))
        #
        # `clip_grad_norm` helps prevent
        # the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(rnn_model.parameters(),
                                      cfg.TRAIN.RNN_GRAD_CLIP)
        optimizer.step()
        
        start_time = time.time()
        if global_step % ARTIFACT_INTERVAL_GLOBAL_STEP == 0:
            # attention Maps
            img_set, _ = build_super_images(imgs[-1].cpu(), captions,
                                            ixtoword, attn_maps, att_sze)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/attention_maps_epoch_%d_batch_%d.png' % (image_dir, epoch_idx, batch_id)
                im.save(fullpath)
                mlflow.log_artifact(fullpath, "output/images")
        if is_early_stop(epoch_idx):
            break
        # #### BATCH LOOP ############
    s_cur_loss = s_total_loss.item() / (batch_id + 1)
    w_cur_loss = w_total_loss.item() / (batch_id + 1)
    try:
        mlflow.log_metrics(dict(
            train_epoch_s_loss0=s_loss0.item(),
            train_epoch_s_loss1=s_loss1.item(),
            train_epoch_w_loss0=w_loss0.item(),
            train_epoch_w_loss1=w_loss1.item(),
            train_epoch_s_loss=s_cur_loss,
            train_epoch_w_loss=w_cur_loss,
        ), step=epoch_idx)
    except Exception as e:
        logger.exception(e)
    
    return global_step


def evaluate(dataloader, cnn_model, rnn_model, batch_size):
    cnn_model.eval()
    rnn_model.eval()
    s_loss0 = 0
    s_loss1 = 0
    w_loss0 = 0
    w_loss1 = 0
    s_total_loss = 0
    w_total_loss = 0
    batch_id = 0
    for batch_id, data in enumerate(dataloader, 0):
        global_step = epoch_idx * len(dataloader) + batch_id
        real_imgs, captions, cap_lens, class_ids, keys = prepare_data(data)
        
        words_features, sent_code = cnn_model(real_imgs[-1])
        # nef = words_features.size(1)
        # words_features = words_features.view(batch_size, nef, -1)
        
        hidden = rnn_model.init_hidden(batch_size)
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)
        
        w_loss0, w_loss1, attn = words_loss(words_features, words_emb, labels,
                                            cap_lens, class_ids, batch_size)
        w_total_loss += (w_loss0 + w_loss1).data
        
        s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        s_total_loss += (s_loss0 + s_loss1).data
        
        mlflow.log_metrics(dict(
            test_batch_s_loss0=s_loss0.item(),
            test_batch_s_loss1=s_loss1.item(),
            test_batch_w_loss0=w_loss0.item(),
            test_batch_w_loss1=w_loss1.item(),
            test_batch_s_loss=(s_loss0 + s_loss1).item(),
            test_batch_w_loss=(w_loss0 + w_loss1).item(),
        ), step=global_step)
        
        logger.info('| Evaluating epoch {:3d} | {:5d} '
                    's_loss {:5.2f} {:5.2f} | '
                    'w_loss {:5.2f} {:5.2f}'
                    .format(epoch_idx, batch_id, len(dataloader),
                            s_loss0, s_loss1,
                            w_loss0, w_loss1))
    
    s_cur_loss = s_total_loss.item() / (batch_id + 1)
    w_cur_loss = w_total_loss.item() / (batch_id + 1)
    
    mlflow.log_metrics(dict(
        test_epoch_s_loss0=s_loss0.item(),
        test_epoch_s_loss1=s_loss1.item(),
        test_epoch_w_loss0=w_loss0.item(),
        test_epoch_w_loss1=w_loss1.item(),
        test_epoch_s_loss=s_cur_loss,
        test_epoch_w_loss=w_cur_loss,
    ), step=epoch_idx)
    
    return s_cur_loss, w_cur_loss


def build_models():
    # build model ############################################################
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    labels = Variable(torch.LongTensor(range(batch_size)))
    start_epoch = 0
    if cfg.TRAIN.NET_E != '':
        state_dict = torch.load(cfg.TRAIN.NET_E)
        text_encoder.load_state_dict(state_dict)
        logger.info('Load %s' % cfg.TRAIN.NET_E)
        #
        name = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)
        image_encoder.load_state_dict(state_dict)
        logger.info('Load %s' % name)
        
        istart = cfg.TRAIN.NET_E.rfind('_') + 8
        iend = cfg.TRAIN.NET_E.rfind('.')
        start_epoch = cfg.TRAIN.NET_E[istart:iend]
        start_epoch = int(start_epoch) + 1
        logger.info('start_epoch=%d' % start_epoch)
    if cfg.CUDA:
        text_encoder = text_encoder.cuda()
        image_encoder = image_encoder.cuda()
        labels = labels.cuda()
    
    return text_encoder, image_encoder, labels, start_epoch


if __name__ == "__main__":
    # Set logging levels
    sys.excepthook = except_hook
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = 'output/DAMSM_%s_%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
    cfg.OUTPUT_DIR = output_dir
    
    ##########################################################################
    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    mkdir_p(model_dir)
    mkdir_p(image_dir)
    ##########################################################################
    cfg.log_file = os.path.join(output_dir, "log.txt")
    attach_file_to_logger(cfg.log_file)
    logger.setLevel(logging.DEBUG)  # logger
    ##########################################################################
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    
    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id
    
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    logger.info('Using config:%s' % cfg)
    
    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = 100
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True
    
    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    batch_size = cfg.TRAIN.BATCH_SIZE
    if cfg.DATASET_NAME == "birds":
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])
        dataset = TextDataset(cfg.DATA_DIR, 'train',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
        model_io_signature = None
        assert dataset
        
        # # validation data #
        dataset_val = TextDataset(cfg.DATA_DIR, 'test',
                                  base_size=cfg.TREE.BASE_SIZE,
                                  transform=image_transform)
    elif cfg.DATASET_NAME == "sixray":
        image_transform = transforms.Compose([
            AspectResize(int(imsize * 76 / 64)),
            # transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])
        dataset = SixrayDataset(cfg.DATA_DIR, 'train',
                                base_size=cfg.TREE.BASE_SIZE,
                                transform=image_transform)
        model_io_signature = None
        assert dataset
        
        # # validation data #
        dataset_val = SixrayDataset(cfg.DATA_DIR, 'test',
                                    base_size=cfg.TREE.BASE_SIZE,
                                    transform=image_transform)
    else:
        logger.error("NotImplementedError(cfg.DATASET_NAME = '%s')" % cfg.DATASET_NAME)
        raise NotImplementedError("cfg.DATASET_NAME = '%s'" % cfg.DATASET_NAME)
    logger.info("dataset.n_words=%d, dataset.embeddings_num=%d" % (dataset.n_words, dataset.embeddings_num))
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True,
                            shuffle=True, num_workers=int(cfg.WORKERS))
    dataloader_val = DataLoader(
        dataset_val, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))
    
    # Train ##############################################################
    text_encoder, image_encoder, labels, start_epoch = build_models()
    para = list(text_encoder.parameters())
    for v in image_encoder.parameters():
        if v.requires_grad:
            para.append(v)
    # optimizer = optim.Adam(para, lr=cfg.TRAIN.ENCODER_LR, betas=(0.5, 0.999))
    # At any point you can hit Ctrl + C to break out of training early.
    start_tracking()
    epoch_idx = start_epoch
    try:
        lr = cfg.TRAIN.ENCODER_LR
        # ### ##### TRAIN LOOP STARTS ##########
        while epoch_idx <= cfg.TRAIN.MAX_EPOCH:
            optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))
            epoch_start_time = time.time()
            count = train(dataloader, image_encoder, text_encoder,
                          batch_size, labels, optimizer, epoch_idx,
                          dataset.ixtoword, image_dir)
            logger.info('-' * 89)
            if len(dataloader_val) > 0:
                s_loss, w_loss = evaluate(dataloader_val, image_encoder,
                                          text_encoder, batch_size)
                logger.info('| end epoch {:3d} | valid loss '
                            '{:5.2f} {:5.2f} | lr {:.5f}|'
                            .format(epoch_idx, s_loss, w_loss, lr))
                # model logging routine
                loss = w_loss if cfg.TRAIN.save_schedule.key == "w_loss" else s_loss
                if can_i_log_model(epoch_idx) or can_i_log_model(epoch_idx, loss):
                    log_model(cfg.OUTPUT_DIR, "image_encoder_epoch_%d" % epoch_idx, image_encoder, model_io_signature)
                    log_model(cfg.OUTPUT_DIR, "text_encoder_epoch_%d" % epoch_idx, text_encoder, model_io_signature)
            logger.info('-' * 89)
            if lr > cfg.TRAIN.ENCODER_LR / 10.:
                lr *= 0.98
            if can_i_log_model(epoch_idx):
                log_model(cfg.OUTPUT_DIR, "image_encoder_epoch_%d" % epoch_idx, image_encoder, model_io_signature)
                log_model(cfg.OUTPUT_DIR, "text_encoder_epoch_%d" % epoch_idx, text_encoder, model_io_signature)
            if is_early_stop(epoch_idx):
                break
            epoch_idx += 1
        
        # ######### TRAIN LOOP ENDS ########
        stop_tracking()
    except Exception as e:
        logger.info('-' * 89)
        logger.info('Exiting from training with exception')
        logger.exception(e)
        log_model(cfg.OUTPUT_DIR, "image_encoder_epoch_%d" % epoch_idx, image_encoder, model_io_signature)
        log_model(cfg.OUTPUT_DIR, "text_encoder_epoch_%d" % epoch_idx, text_encoder, model_io_signature)
        stop_tracking("Exception: %s" % (str(e)))
    logger.info("Output saved at: '%s'" % cfg.OUTPUT_DIR)
