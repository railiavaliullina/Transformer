from easydict import EasyDict


cfg = EasyDict()

cfg.batch_size = 32
cfg.lr = 0
cfg.betas = (0.9, 0.98)
cfg.optimizer_eps = 1e-9
cfg.layer_norm_eps = 1e-6
cfg.warmup_steps = 4000
cfg.p_drop = 0.1
cfg.d_model = 512
cfg.h = 8
cfg.dim_feedforward = 2048
cfg.dropout = 0.1
cfg.num_encoder_layers = 6
cfg.num_decoder_layers = 6
cfg.pos_enc_max_size = 500

cfg.log_metrics = False
cfg.experiment_name = 'transformer_digits_last_exp'

cfg.run_inference_before_training = False
cfg.load_saved_model = False
cfg.checkpoints_dir = f'../saved_files/checkpoints/{cfg.experiment_name}'
cfg.step_to_load = 1101
cfg.save_model = True
cfg.inference_pred_max_len = 100
