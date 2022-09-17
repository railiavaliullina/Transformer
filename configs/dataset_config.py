from easydict import EasyDict


cfg = EasyDict()

cfg.dataset_path = '../data/synthetic_dataset.pickle'
cfg.load_saved_data = False

cfg.sentences_num = 10000
cfg.sen_len_range = (5, 20)
cfg.train_set_part = 0.8
