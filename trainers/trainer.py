import torch
import numpy as np
import os
import time
from torch.nn.functional import softmax, cross_entropy
import Levenshtein

from dataloader.dataloader import get_dataloader
from models.Transformer import get_model
from utils.logging import Logger


class Trainer(object):
    def __init__(self, cfg):
        """
        Class for initializing and performing training procedure.
        :param cfg: train config
        """
        self.cfg = cfg
        self.dl = get_dataloader()
        self.dataset = self.dl.dataset
        self.cfg.vocab_size = len(self.dataset.word_to_index)
        self.index_to_word = self.dataset.index_to_word
        self.model = get_model(self.cfg)
        self.optimizer = self.get_optimizer()
        self.logger = Logger(self.cfg)

    def get_optimizer(self):
        """
        Gets optimizer.
        :return: optimizer
        """
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.cfg.lr,
                                     betas=self.cfg.betas,
                                     eps=self.cfg.optimizer_eps)
        return optimizer

    def restore_model(self):
        """
        Restores saved model.
        """
        if self.cfg.load_saved_model:
            print(f'Trying to load checkpoint from epoch {self.cfg.step_to_load}...')
            try:
                checkpoint = torch.load(self.cfg.checkpoints_dir + f'/checkpoint_{self.cfg.step_to_load}.pth')
                load_state_dict = checkpoint['model']
                self.model.load_state_dict(load_state_dict)
                self.global_step = checkpoint['global_step'] + 1
                self.optimizer.load_state_dict(checkpoint['opt'])
                print(f'Loaded checkpoint from epoch {self.cfg.step_to_load}.')
            except FileNotFoundError:
                print('Checkpoint not found')

    def save_model(self):
        """
        Saves model.
        """
        if self.cfg.save_model:
            print('Saving current model...')
            state = {
                'model': self.model.state_dict(),
                'epoch': self.epoch,
                'global_step': self.global_step,
                'opt': self.optimizer.state_dict()
            }
            if not os.path.exists(self.cfg.checkpoints_dir):
                os.makedirs(self.cfg.checkpoints_dir)

            path_to_save = os.path.join(self.cfg.checkpoints_dir, f'checkpoint_{self.global_step}.pth')
            torch.save(state, path_to_save)
            print(f'Saved model to {path_to_save}.')

    def inference(self):
        """
        Runs inference and calculates Levenshtein.distance between prediction ans target.
        """
        # '<BOS>' -> 0, '<EOS>' -> 1
        self.model.eval()
        predictions, reals, l_distances = [], [], []
        step = 0

        for iter_, batch in enumerate(self.dl):
            input_, target_dec_input, target_dec_output = batch

            for sentence in input_:
                encoder_out = self.model.encoder(sentence.unsqueeze(0), (sentence != 2).unsqueeze(-2))
                prediction = torch.zeros(self.cfg.inference_pred_max_len, dtype=torch.int32)

                for digit_idx in range(1, self.cfg.inference_pred_max_len):
                    target_mask = torch.autograd.Variable(torch.tensor(
                        np.triu(np.ones((1, digit_idx, digit_idx)), k=1).astype('uint8')) == 0)

                    out = self.model.out(self.model.decoder(prediction[:digit_idx].unsqueeze(0), encoder_out,
                                                            (sentence != 2).unsqueeze(-2), target_mask))
                    _, top_k_ids = softmax(out, dim=-1)[:, -1].data.topk(1)
                    prediction[digit_idx] = top_k_ids[0][0]
                    if top_k_ids[0][0] == 1:
                        break

                pred = np.asarray([self.index_to_word[ix.item()] for ix in prediction[:digit_idx] if
                                   not isinstance(self.index_to_word[ix.item()], str)])
                target = np.asarray([self.index_to_word[ix.item()] for ix in sentence
                                     if not isinstance(self.index_to_word[ix.item()], str)])

                levenshtein_dist = sum(
                    [1.0 - (Levenshtein.distance(predicted_word, word) / max(len(word), len(predicted_word)))
                     for word, predicted_word in
                     zip(''.join([str(t) for t in target]), ''.join([str(t) for t in pred]))]) / len(target)

                self.logger.log_metrics(names=['train/levenshtein_dist', 'train/levenshtein_dist_mean'],
                                        metrics=[levenshtein_dist, np.mean(l_distances)], step=step)
                l_distances.append(levenshtein_dist)
                predictions.append(pred)
                reals.append(target)
                print(f'\nStep: {step}\nreal: {reals[-1]}\npredicted: {predictions[-1]}'
                      f'\nlevenshtein_dist: {np.mean(l_distances)}\n')
                step += 1

    def make_training_step(self, batch):
        """
        Makes single training step.
        :param batch: current batch containing input and targets
        :return: loss on current batch
        """
        input_, target_dec_input, target_dec_output = batch
        b_size, seq_len = input_.size()
        padding_start_idx = [torch.argmin(seq) for seq in input_]
        input_padding_mask = (input_ != 2).unsqueeze(1)
        target_padding_mask = (target_dec_input != 2).unsqueeze(1)
        target_mask = target_padding_mask * torch.autograd.Variable(
            torch.tensor(np.triu(np.ones((1, seq_len, seq_len)), k=1).astype('uint8')) == 0)

        out = self.model(input_, target_dec_input, input_padding_mask, target_mask)
        pred = torch.argmax(softmax(out, dim=-1), -1)
        acc = np.mean([np.mean(inp[:pad_idx] == pr[:pad_idx])
                       for pad_idx, inp, pr in zip(padding_start_idx, input_.numpy(), pred.numpy())])

        loss = cross_entropy(out.view(b_size * seq_len, -1), target_dec_output.view(-1), ignore_index=2)
        assert not torch.isnan(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), acc * 100

    def train(self):
        """
        Runs training procedure.
        """
        total_training_start_time = time.time()
        self.epoch, self.global_step = -1, 0

        # restore model if necessary
        self.restore_model()

        if self.cfg.run_inference_before_training:
            self.inference()

        # start training
        print(f'Starting training...')
        iter_num = len(self.dl)

        losses, accs = [], []
        for iter_, batch in enumerate(self.dl):

            loss, acc = self.make_training_step(batch)
            self.logger.log_metrics(names=['train/loss', 'train/acc'], metrics=[loss, acc], step=self.global_step)

            losses.append(loss)
            accs.append(acc)
            self.global_step += 1

            for param_groups in self.optimizer.param_groups:
                param_groups['lr'] = self.cfg.d_model ** (-0.5) * min(self.global_step ** (-0.5), self.global_step *
                                                                      self.cfg.warmup_steps ** (-1.5))

            if iter_ % 1 == 0:
                mean_loss = np.mean(losses[-50:]) if len(losses) > 50 else np.mean(losses)
                mean_acc = np.mean(accs)
                print(f'iter: {iter_}/{iter_num}, loss: {mean_loss}, mean acc: {mean_acc}, acc: {acc}')

            # save model
            if iter_ % 100 == 0:
                self.save_model()

        print(f'Training time: {round((time.time() - total_training_start_time) / 60, 3)} min')
