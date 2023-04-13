import os
import shutil
import time
import numpy as np
import torch
import dgl
from config import _DEBUG_FLAG_
from data_manager import data_loaders
from tools.logger import *
from tools.utils import save_model
from runner.evaluation import run_eval


def setup_training(model, hps, data_variables):
    train_dir = os.path.join(hps.save_root, "train")
    if os.path.exists(train_dir) and hps.restore_model != 'None':
        logger.info("[INFO] Restoring %s for training...", hps.restore_model)
        best_model_file = os.path.join(train_dir, hps.restore_model)
        model.load_state_dict(torch.load(best_model_file))
        hps.save_root = hps.save_root + "_reload"
    else:
        logger.info("[INFO] Create new model for training...")
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        os.makedirs(train_dir)

    try:
        run_training(model, hps, data_variables=data_variables)
    except KeyboardInterrupt:
        logger.error("[Error] Caught keyboard interrupt on worker. Stopping supervisor...")
        save_model(model, os.path.join(train_dir, "earlystop"))


class Trainer:
    def __init__(self, model, hps):
        self.model = model
        self.hps = hps
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=hps.lr)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.best_train_loss = None
        self.best_loss = None
        self.best_F = None
        self.non_descent_cnt = 0
        self.saveNo = 0
        self.epoch = 1
        self.epoch_avg_loss = 0

    def run_epoch(self, train_loader):
        epoch_start_time = time.time()

        batch_time_sum = 0
        train_loss = 0.0
        epoch_loss = 0.0

        for i, (G, index) in enumerate(train_loader):
            iter_start_time = time.time()
            loss = self.train_batch(G=G)
            train_loss += float(loss.data)
            epoch_loss += float(loss.data)
            batch_time_sum += time.time() - iter_start_time
            if i % 100 == 99:
                if _DEBUG_FLAG_:
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            logger.debug(name)
                            logger.debug(param.grad.data.sum())
                logger.info(
                    '| end of iter {:3d} | time: {:5.2f}s | train loss {:5.4f} | '.format(i, (batch_time_sum / 100),
                                                                                          float(train_loss / 100)))
                train_loss = 0.0
                batch_time_sum = 0

        epoch_avg_loss = epoch_loss / len(train_loader)
        logger.info('   | end of epoch {:3d} | time: {:5.2f}s | epoch train loss {:5.4f} | '
                    .format(self.epoch, (time.time() - epoch_start_time), float(epoch_avg_loss)))
        return epoch_loss

    def train_batch(self, G):
        G = G.to(self.hps.device)
        outputs = self.model.forward(G)  # [n_snodes, 2]
        snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        label = G.ndata["label"][snode_id].sum(-1)  # [n_nodes]
        G.nodes[snode_id].data["loss"] = self.criterion(outputs, label.to(self.hps.device)).unsqueeze(
            -1)  # [n_nodes, 1]
        loss = dgl.sum_nodes(G, "loss")  # [batch_size, 1]
        loss = loss.mean()
        if not (np.isfinite(loss.data.cpu())).numpy():
            logger.error("train Loss is not finite. Stopping.")
            logger.info(loss)
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    logger.info(name)
                    # logger.info(param.grad.data.sum())
            raise Exception("train Loss is not finite. Stopping.")
        self.optimizer.zero_grad()
        loss.backward()
        if self.hps.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hps.max_grad_norm)

        self.optimizer.step()
        return loss

    def change_learning_rate(self):
        if self.hps.lr_descent:
            new_lr = max(5e-6, self.hps.lr / (self.epoch + 1))
            for param_group in list(self.optimizer.param_groups):
                param_group['lr'] = new_lr
            logger.info("[INFO] The learning rate now is %f", new_lr)

    def save_model(self, train_dir):
        if not self.best_train_loss or self.epoch_avg_loss < self.best_train_loss:
            save_file = os.path.join(train_dir, "bestmodel")
            logger.info('[INFO] Found new best model with %.3f running_train_loss. Saving to %s',
                        float(self.epoch_avg_loss),
                        save_file)
            save_model(self.model, save_file)
            best_train_loss = self.epoch_avg_loss
        elif self.epoch_avg_loss >= self.best_train_loss:
            logger.error("[Error] training loss does not descent. Stopping supervisor...")
            save_model(self.model, os.path.join(train_dir, "earlystop"))
            sys.exit(1)


def run_training(model, hps, data_variables):
    trainer = Trainer(model=model, hps=hps)

    for epoch in range(1, hps.n_epochs + 1):
        train_loader = data_loaders.make_dataloader(data_file=data_variables["train_file"],
                                                    vocab=data_variables["vocab"], hps=hps,
                                                    filter_word=data_variables["filter_word"],
                                                    w2s_path=data_variables["train_w2s_path"],
                                                    max_instance=hps.max_instances)
        trainer.epoch = epoch
        model.train()
        trainer.run_epoch(train_loader=train_loader)
        del train_loader

        valid_loader = data_loaders.make_dataloader(data_file=data_variables["valid_file"],
                                                    vocab=data_variables["vocab"], hps=hps,
                                                    filter_word=data_variables["filter_word"],
                                                    w2s_path=data_variables["val_w2s_path"],
                                                    max_instance=hps.max_instances)

        best_loss, best_F, non_descent_cnt, saveNo = run_eval(model, valid_loader, valid_loader.dataset, hps,
                                                              trainer.best_loss,
                                                              trainer.best_F, trainer.non_descent_cnt, trainer.saveNo)

        del valid_loader

        if non_descent_cnt >= 3:
            logger.error("[Error] val loss does not descent for three times. Stopping supervisor...")
            save_model(model, os.path.join(data_variables["train_dir"], "earlystop"))
            return
