import os
import pickle
import logging


logger = logging.getLogger(__name__)


def txt2pkl(path):
    label_corpus = set()
    chars_file, words_file, labels_file = [], [], []
    with open(path) as f:
        logger.info(f"Reading data from {path}...")
        lines = f.readlines()
    
    for line in lines:
        chars_line, words_line, labels_line = [], [], []
        line = line.strip().split(' ')
        for chars_label in line:
            chars_label = chars_label.split('/')
            chars, label = chars_label[0], chars_label[1]
            chars_line.extend([c for c in chars])
            if label == 'o':
                label_corpus.add('o')
                labels_line.extend(['o' for i in range(len(chars))])
            else:
                label_corpus.add('B_' + label)
                label_corpus.add('I_' + label)
                label_corpus.add('S_' + label)
                label_corpus.add('E_' + label)

                if len(chars) == 1:
                    labels_line.append('S_' + label)
                else:
                    labels_line.append('B_' + label)
                    labels_line.extend(['I_' + label for i in range(len(chars) - 2)])
                    labels_line.append('E_' + label)

        chars_file.append(chars_line)
        labels_file.append(labels_line)
    
    data = {"chars": chars_file, "words": words_file, "labels": labels_file}
    return data, label_corpus
            


def preprocess(cfg):
    train_txt_path = os.path.join(cfg.cwd, f"data/origin/{cfg.dataset}/train.txt")
    test_txt_path = os.path.join(cfg.cwd, f"data/origin/{cfg.dataset}/test.txt")

    train_pkl_path = os.path.join(cfg.cwd, f"data/out/{cfg.dataset}/train.pkl")
    test_pkl_path = os.path.join(cfg.cwd, f"data/out/{cfg.dataset}/test.pkl")

    train_pkl_file = open(train_pkl_path, "wb")
    test_pkl_file = open(test_pkl_path, "wb")

    train_data, corpus = txt2pkl(train_txt_path)
    test_data, _ = txt2pkl(test_txt_path)

    logger.info(f"Saving train data to {train_pkl_path}...")
    pickle.dump(train_data, train_pkl_file)
    logger.info(f"Saving trest data to {test_pkl_path}...")
    pickle.dump(test_data, test_pkl_file)

    return corpus
    