import torch
import numpy as np


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        """
        Class for generating digits dataset.
        :param cfg: dataset config
        """
        self.cfg = cfg
        self.get_vocab()

    def get_vocab(self):
        vocab = ['<BOS>', '<EOS>', '<PAD>'] + list(np.arange(0, 10))
        self.word_to_index = {word: idx for idx, word in enumerate(vocab)}
        self.index_to_word = {idx: word for idx, word in enumerate(vocab)}

    def __len__(self):
        """
        Gets dataset length.
        :return: dataset length
        """
        return int(1e8)

    def __getitem__(self, idx):
        """
        Gets dataset item (encoded text and corresponding label)
        :param idx: index for getting data
        :return: sentence
        """
        sentence_len = np.random.randint(self.cfg.sen_len_range[0], self.cfg.sen_len_range[1] + 1)
        sentence = []
        for i in range(sentence_len):
            digit = np.random.randint(0, 10)
            sentence.append(digit)
        out = [self.word_to_index[s] for s in sentence]
        return out
