# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
from io import open
from tensorboardX import SummaryWriter

import torch


logger = logging.getLogger(__name__)


class tbLogger(object):
    def __init__(self, log_dir, txt_dir, task_names, task_ids, task_num_iters,
                 gradient_accumulation_steps, save_logger=True, txt_name="out.txt"):
        logger.info("logging file at: " + log_dir)

        self.save_logger = save_logger
        self.log_dir = log_dir
        self.txt_dir = txt_dir
        if self.save_logger:
            self.logger = SummaryWriter(log_dir=log_dir)

        self.txt_f = open(txt_dir + "/" + txt_name, "w")
        self.task_id2name = {ids: name.replace("+", "plus") for ids, name in zip(task_ids, task_names)}
        self.task_ids = task_ids
        self.task_loss = {task_id: 0 for task_id in task_ids}
        self.task_loss_tmp = {task_id: 0 for task_id in task_ids}
        self.task_score_tmp = {task_id: 0 for task_id in task_ids}
        self.task_norm_tmp = {task_id: 0 for task_id in task_ids}
        self.task_step = {task_id: 0 for task_id in task_ids}
        self.task_step_tmp = {task_id: 0 for task_id in task_ids}
        self.task_num_iters = task_num_iters
        self.epochId = 0
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.task_loss_val = {task_id: 0 for task_id in task_ids}
        self.task_score_val = {task_id: 0 for task_id in task_ids}
        self.task_step_val = {task_id: 0 for task_id in task_ids}
        self.task_iter_val = {task_id: 0 for task_id in task_ids}
        self.task_datasize_val = {task_id: 0 for task_id in task_ids}

        self.masked_t_loss = {task_id: 0 for task_id in task_ids}
        self.masked_v_loss = {task_id: 0 for task_id in task_ids}
        self.next_sentense_loss = {task_id: 0 for task_id in task_ids}

        self.masked_t_loss_val = {task_id: 0 for task_id in task_ids}
        self.masked_v_loss_val = {task_id: 0 for task_id in task_ids}
        self.next_sentense_loss_val = {task_id: 0 for task_id in task_ids}

    def __getstate__(self):
        d = dict(self.__dict__)
        del d["logger"]
        del d["txt_f"]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        if self.save_logger:
            self.logger = SummaryWriter(log_dir=self.log_dir)

        self.txt_f = open(self.txt_dir + "/" + "out.txt", "a")

    def txt_close(self):
        self.txt_f.close()

    def linePlot(self, step, val, split, key, xlabel="None"):
        if self.save_logger:
            self.logger.add_scalar(split + "/" + key, val, step)

    def step_train(self, epochId, stepId, loss, score, norm, task_id, split):
        self.task_loss[task_id] += loss
        self.task_loss_tmp[task_id] += loss
        self.task_score_tmp[task_id] += score
        self.task_norm_tmp[task_id] += norm
        self.task_step[task_id] += self.gradient_accumulation_steps
        self.task_step_tmp[task_id] += self.gradient_accumulation_steps
        self.epochId = epochId

        # plot on tensorboard.
        self.linePlot(stepId, loss, split, self.task_id2name[task_id] + "_loss")
        self.linePlot(stepId, score, split, self.task_id2name[task_id] + "_score")
        self.linePlot(stepId, norm, split, self.task_id2name[task_id] + "_norm")

    def step_train_CC(self, epochId, stepId, masked_loss_t, masked_loss_v, next_sentence_loss, norm, task_id, split):
        self.masked_t_loss[task_id] += masked_loss_t
        self.masked_v_loss[task_id] += masked_loss_v
        self.next_sentense_loss[task_id] += next_sentence_loss
        self.task_norm_tmp[task_id] += norm

        self.task_step[task_id] += self.gradient_accumulation_steps
        self.task_step_tmp[task_id] += self.gradient_accumulation_steps
        self.epochId = epochId

        # plot on tensorboard.
        self.linePlot(stepId, masked_loss_t, split, self.task_id2name[task_id] + "_masked_loss_t")
        self.linePlot(stepId, masked_loss_v, split, self.task_id2name[task_id] + "_masked_loss_v")
        self.linePlot(stepId, next_sentence_loss, split, self.task_id2name[task_id] + "_next_sentence_loss")

    def step_val(self, epochId, loss, score, task_id, batch_size, split):
        self.task_loss_val[task_id] += loss * batch_size
        self.task_score_val[task_id] += score
        self.task_step_val[task_id] += self.gradient_accumulation_steps
        self.task_datasize_val[task_id] += batch_size

    def step_val_CC(self, iter_id, masked_loss_t, masked_loss_v, next_sentence_loss, task_id, batch_size, split):
        self.masked_t_loss_val[task_id] += masked_loss_t
        self.masked_v_loss_val[task_id] += masked_loss_v
        self.next_sentense_loss_val[task_id] += next_sentence_loss

        self.task_step_val[task_id] += self.gradient_accumulation_steps
        self.task_datasize_val[task_id] += batch_size
        
        # plot on tensorboard.
        self.linePlot(iter_id, masked_loss_t, split, self.task_id2name[task_id] + "_masked_loss_t")
        self.linePlot(iter_id, masked_loss_v, split, self.task_id2name[task_id] + "_masked_loss_v")
        self.linePlot(iter_id, next_sentence_loss, split, self.task_id2name[task_id] + "_next_sentence_loss")

    def showLossValAll(self):
        progressInfo = "Eval Ep: %d " % self.epochId
        lossInfo = "Validation "
        val_scores = {}
        ave_loss = 0
        for task_id in self.task_ids:
            loss = self.task_loss_val[task_id] / float(self.task_step_val[task_id])
            score = self.task_score_val[task_id] / float(self.task_datasize_val[task_id])
            val_scores[task_id] = score
            ave_loss += loss
            lossInfo += "[%s]: loss %.3f score %.3f " % (self.task_id2name[task_id], loss, score * 100.0)

            self.linePlot(self.epochId, loss, "val", self.task_id2name[task_id] + "_loss")
            self.linePlot(self.epochId, score, "val", self.task_id2name[task_id] + "_score")

        self.task_loss_val = {task_id: 0 for task_id in self.task_loss_val}
        self.task_score_val = {task_id: 0 for task_id in self.task_score_val}
        self.task_datasize_val = {task_id: 0 for task_id in self.task_datasize_val}
        self.task_step_val = {task_id: 0 for task_id in self.task_ids}

        logger.info(progressInfo)
        logger.info(lossInfo)
        print(lossInfo, file=self.txt_f)
        return val_scores

    def getValScore(self, task_id):
        return self.task_score_val[task_id] / float(self.task_datasize_val[task_id])

    def showLossVal(self, task_id, task_stop_controller=None):
        progressInfo = "Eval task %s on iteration %d " % (task_id, self.task_step[task_id])
        lossInfo = "Validation "
        ave_loss = 0
        loss = self.task_loss_val[task_id] / float(self.task_datasize_val[task_id])
        score = self.task_score_val[task_id] / float(self.task_datasize_val[task_id])
        ave_loss += loss
        lossInfo += "[%s]: loss %.3f score %.3f " % (self.task_id2name[task_id], loss, score * 100.0)

        self.linePlot(self.task_step[task_id], loss, "val", self.task_id2name[task_id] + "_loss")
        self.linePlot(self.task_step[task_id], score, "val", self.task_id2name[task_id] + "_score")
        if task_stop_controller is not None:
            self.linePlot(self.task_step[task_id], task_stop_controller[task_id].in_stop,
                          "val", self.task_id2name[task_id] + "_early_stop")

        self.task_loss_val[task_id] = 0
        self.task_score_val[task_id] = 0
        self.task_datasize_val[task_id] = 0
        self.task_step_val[task_id] = 0
        logger.info(progressInfo)
        logger.info(lossInfo)
        print(lossInfo, file=self.txt_f)
        return score

    def showLossTrain(self):
        # show the current loss, once showed, reset the loss.
        lossInfo = ""
        for task_id in self.task_ids:
            if self.task_num_iters[task_id] > 0:
                if self.task_step_tmp[task_id]:
                    lossInfo += (
                        "[%s]: iter %d Ep: %.2f loss %.3f score %.3f lr %.6g "
                        % (
                            self.task_id2name[task_id], self.task_step[task_id],
                            self.task_step[task_id] / float(self.task_num_iters[task_id]),
                            self.task_loss_tmp[task_id] / float(self.task_step_tmp[task_id]),
                            self.task_score_tmp[task_id] / float(self.task_step_tmp[task_id]),
                            self.task_norm_tmp[task_id] / float(self.task_step_tmp[task_id]),
                        )
                    )

        logger.info(lossInfo)
        print(lossInfo, file=self.txt_f)

        self.task_step_tmp = {task_id: 0 for task_id in self.task_ids}
        self.task_loss_tmp = {task_id: 0 for task_id in self.task_ids}
        self.task_score_tmp = {task_id: 0 for task_id in self.task_ids}
        self.task_norm_tmp = {task_id: 0 for task_id in self.task_ids}

    def showLossValCC(self):
        lossInfo = "Validation "
        for task_id in self.task_ids:
            masked_t_loss_val = self.masked_t_loss_val[task_id] / float(self.task_step_val[task_id])
            masked_v_loss_val = self.masked_v_loss_val[task_id] / float(self.task_step_val[task_id])
            next_sentense_loss_val = self.next_sentense_loss_val[task_id] / float(self.task_step_val[task_id])

            lossInfo += "[%s]: masked_t %.3f masked_v %.3f NSP %.3f" % (
                self.task_id2name[task_id],
                masked_t_loss_val,
                masked_v_loss_val,
                next_sentense_loss_val,
            )

            self.linePlot(self.epochId, masked_t_loss_val, "val", self.task_id2name[task_id] + "_mask_t")
            self.linePlot(self.epochId, masked_v_loss_val, "val", self.task_id2name[task_id] + "_maks_v")
            self.linePlot(self.epochId, next_sentense_loss_val, "val", self.task_id2name[task_id] + "_nsp")

        self.masked_t_loss_val = {task_id: 0 for task_id in self.masked_t_loss_val}
        self.masked_v_loss_val = {task_id: 0 for task_id in self.masked_v_loss_val}
        self.next_sentense_loss_val = {task_id: 0 for task_id in self.next_sentense_loss_val}
        self.task_datasize_val = {task_id: 0 for task_id in self.task_datasize_val}
        self.task_step_val = {task_id: 0 for task_id in self.task_ids}

        logger.info(lossInfo)
        print(lossInfo, file=self.txt_f)

    def showLossTrainCC(self):
        # show the current loss, once showed, reset the loss.
        lossInfo = ""
        for task_id in self.task_ids:
            if self.task_num_iters[task_id] > 0:
                if self.task_step_tmp[task_id]:
                    lossInfo += (
                        "[%s]: iter %d Ep: %.2f masked_t %.3f masked_v %.3f NSP %.3f lr %.6g"
                        % (
                            self.task_id2name[task_id], self.task_step[task_id],
                            self.task_step[task_id] / float(self.task_num_iters[task_id]),
                            self.masked_t_loss[task_id] / float(self.task_step_tmp[task_id]),
                            self.masked_v_loss[task_id] / float(self.task_step_tmp[task_id]),
                            self.next_sentense_loss[task_id] / float(self.task_step_tmp[task_id]),
                            self.task_norm_tmp[task_id] / float(self.task_step_tmp[task_id]),
                        )
                    )

        logger.info(lossInfo)
        print(lossInfo, file=self.txt_f)

        self.task_step_tmp = {task_id: 0 for task_id in self.task_ids}
        self.masked_t_loss = {task_id: 0 for task_id in self.task_ids}
        self.masked_v_loss = {task_id: 0 for task_id in self.task_ids}
        self.next_sentense_loss = {task_id: 0 for task_id in self.task_ids}
        self.task_norm_tmp = {task_id: 0 for task_id in self.task_ids}


def freeze_layers(model):
    fixed_layers = set(model.config.fixed_layers)  # e.g. "bert.embeddings", "bert.v_embeddings.LayerNorm", "bert.encoder.layer.15.output.v_dense"
    for key, value in dict(model.named_parameters()).items():
        for name in fixed_layers:
            if (key + '.').startswith(name + '.'):
                value.requires_grad = False


def print_and_log(string, logger=None):
    if logger is None:
        print(string)
    else:
        logger.info(string)


def summary_parameters(model, logger=None):
    """
    Summary Parameters of Model
    :param model: torch.nn.module_name
    :param logger: logger
    :return: None
    """

    print_and_log('>> Parameters:', logger)
    parameters = [(str(n), str(v.dtype), str(tuple(v.shape)), str(v.numel()), str(v.requires_grad))
                           for n, v in model.named_parameters()]
    max_lens = [max([len(item) + 4 for item in col]) for col in zip(*parameters)]
    raw_format = '|' + '|'.join(['{{:{}s}}'.format(max_len) for max_len in max_lens]) + '|'
    raw_split = '-' * (sum(max_lens) + len(max_lens) + 1)
    print_and_log(raw_split, logger)
    print_and_log(raw_format.format('Name', 'Dtype', 'Shape', '#Params', 'Trainable'), logger)
    print_and_log(raw_split, logger)

    for name, dtype, shape, number, grad in parameters:
        print_and_log(raw_format.format(name, dtype, shape, number, grad), logger)
        print_and_log(raw_split, logger)

    num_trainable_params = sum([v.numel() for v in model.parameters() if v.requires_grad])
    total_params = sum([v.numel() for v in model.parameters()])
    non_trainable_params = total_params - num_trainable_params
    print_and_log('>> {:25s}\t{:.2f}\tM'.format('# TrainableParams:', num_trainable_params / (1.0 * 10 ** 6)), logger)
    print_and_log('>> {:25s}\t{:.2f}\tM'.format('# NonTrainableParams:', non_trainable_params / (1.0 * 10 ** 6)), logger)
    print_and_log('>> {:25s}\t{:.2f}\tM'.format('# TotalParams:', total_params / (1.0 * 10 ** 6)), logger)


def save(path, logger, epoch_id, model, optimizer, scheduler, global_step, tb_logger, default_gpu, score, is_best=False):
    if default_gpu:
        # Save a trained model
        logger.info("** ** * Saving model * ** ** ")
        model_to_save = model.module if hasattr(model, "module") else model  # Only save the model it-self
        output_model_file = os.path.join(path, "pytorch_model_" + str(epoch_id) + ".bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        if is_best:
            output_model_file = os.path.join(path, "pytorch_model_best.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
        output_checkpoint = os.path.join(path, "pytorch_ckpt_latest.tar")
        torch.save(
            {"model_state_dict": model_to_save.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(),
             "scheduler_state_dict": scheduler.state_dict(),
             "global_step": global_step,
             "epoch_id": epoch_id,
             "tb_logger": tb_logger,
             "score": score,
             },
            output_checkpoint,
        )


def resume(path, model, optimizer, scheduler, tb_logger):
    start_iter_id = 0
    global_step = 0
    start_epoch = 0
    best_score = float("-inf")
    if path != "" and os.path.exists(path):
        checkpoint = torch.load(path, map_location="cpu")
        new_dict = {}
        for attr in checkpoint["model_state_dict"]:
            if attr.startswith("module."):
                new_dict[attr.replace("module.", "", 1)] = checkpoint["model_state_dict"][attr]
            else:
                new_dict[attr] = checkpoint["model_state_dict"][attr]
        model.load_state_dict(new_dict)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])  # OLD: checkpoint["warmup_scheduler_state_dict"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = checkpoint["global_step"]
        start_epoch = int(checkpoint["epoch_id"]) + 1
        tb_logger = checkpoint["tb_logger"]
        best_score = checkpoint.get("score", float("-inf"))
        del checkpoint
    return start_iter_id, global_step, start_epoch, tb_logger, best_score
