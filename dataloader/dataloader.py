import torch
import numpy as np

from datasets.synthetic_dataset import SyntheticDataset
from configs.dataset_config import cfg as dataset_cfg
from configs.train_config import cfg as train_cfg


def get_dataloader():
    """
    Initializes dataset and gets dataloader.
    :return: dataloader
    """
    dataset = SyntheticDataset(cfg=dataset_cfg)
    dl = torch.utils.data.DataLoader(dataset, batch_size=train_cfg.batch_size, collate_fn=collate_fn)
    return dl


def collate_fn(data):
    lens = np.asarray([len(d) for d in data])
    max_len = max(lens) + 1
    padded_inputs, padded_target_dec_input, padded_target_dec_output = [], [], []
    for d in data:
        #  adding indexes of tokens:
        #  '<BOS>' -> 0 (to padded_target_dec_input),
        #  '<EOS>' -> 1 (to padded_target_dec_output),
        #  '<PAD>' - 2 (for padding)
        padded_inputs.append(list(np.concatenate([d, np.ones(max_len - len(d)).astype(int) * 2])))
        padded_target_dec_input.append([0] + list(d) + list(np.ones(max_len - len(d) - 1).astype(int) * 2))
        padded_target_dec_output.append(list(d) + [1] + list(np.ones(max_len - len(d) - 1).astype(int) * 2))
    return torch.tensor(padded_inputs), torch.tensor(padded_target_dec_input), torch.tensor(padded_target_dec_output)
