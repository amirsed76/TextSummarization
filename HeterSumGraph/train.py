import datetime
import os
import random
import numpy as np
import torch
from config import pars_args
from HiGraph import HSumGraph
from module.embedding import Word_Embedding
from module.vocabulary import Vocab
from tools.logger import *
from runner.train import setup_training
from data_manager import data_loaders


def set_device(hps):
    if hps.cuda and hps.gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info("[INFO] Use cuda")
    else:
        device = torch.device("cpu")
        logger.info("[INFO] Use CPU")
    hps.device = device
    return hps


def initial_seed(hps):
    random.seed(hps.seed)
    np.random.seed(hps.seed)
    torch.manual_seed(hps.seed)


def get_files(hps):
    # train_file = os.path.join(hps.data_dir, "train.label.jsonl")
    train_file = os.path.join(hps.data_dir, "val.label.jsonl")
    valid_file = os.path.join(hps.data_dir, "val.label.jsonl")
    vocal_file = os.path.join(hps.cache_dir, "vocab")
    filter_word = os.path.join(hps.cache_dir, "filter_word.txt")
    train_w2s_path = os.path.join(hps.cache_dir, "val.w2s.tfidf.jsonl")
    # train_w2s_path = os.path.join(hps.cache_dir, "train.w2s.tfidf.jsonl")
    val_w2s_path = os.path.join(hps.cache_dir, "val.w2s.tfidf.jsonl")
    log_path = hps.log_root

    return train_file, valid_file, vocal_file, filter_word, train_w2s_path, val_w2s_path, log_path


def main():
    args = pars_args()
    hps = args
    hps = set_device(hps=hps)
    os.environ['CUDA_VISIBLE_DEVICES'] = hps.gpu
    torch.set_printoptions(threshold=50000)
    train_file, valid_file, vocal_file, filter_word, train_w2s_path, val_w2s_path, log_path = get_files(hps=hps)

    # train_log setting
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_path, "train_" + now_time)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Pytorch %s", torch.__version__)
    logger.info("[INFO] Create Vocab, vocab path is %s", vocal_file)
    vocab = Vocab(vocal_file, hps.vocab_size)
    embed = torch.nn.Embedding(vocab.size(), hps.word_emb_dim, padding_idx=0)
    if hps.word_embedding:
        embed_loader = Word_Embedding(hps.embedding_path, vocab)
        vectors = embed_loader.load_my_vecs(hps.word_emb_dim)
        pretrained_weight = embed_loader.add_unknown_words_by_avg(vectors, hps.word_emb_dim)
        embed.weight.data.copy_(torch.Tensor(pretrained_weight))
        embed.weight.requires_grad = hps.embed_train

    logger.info(hps)

    if hps.model == "HSG":
        model = HSumGraph(hps, embed)
        logger.info("[MODEL] HeterSumGraph ")
        train_loader = data_loaders.make_dataloader(data_file=train_file, vocab=vocab, hps=hps, filter_word=filter_word,
                                                    w2s_path=train_w2s_path,max_instance=100)
        valid_loader = data_loaders.make_dataloader(data_file=valid_file, vocab=vocab, hps=hps, filter_word=filter_word,
                                                    w2s_path=val_w2s_path,max_instance=100)

    # CAN use HDSG
    else:
        logger.error("[ERROR] Invalid Model Type!")
        raise NotImplementedError("Model Type has not been implemented")

    setup_training(model, train_loader, valid_loader, valid_loader.dataset, hps)


if __name__ == '__main__':
    main()
