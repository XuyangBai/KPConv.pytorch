import time, os
import numpy as np
from tensorboardX import SummaryWriter
import torch
from utils.timer import Timer, AverageMeter
from utils.metrics import calculate_acc, calculate_iou


class Trainer(object):
    def __init__(self, args):
        self.config = args.config
        # parameters
        self.start_epoch = args.start_epoch
        self.epoch = args.epoch
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.device = args.device
        self.verbose = args.verbose
        self.best_iou = 0
        self.best_acc = 0
        self.best_loss = 10000000

        self.model = args.model.to(self.device)
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.scheduler_interval = args.scheduler_interval
        self.snapshot_interval = args.snapshot_interval
        self.evaluate_interval = args.evaluate_interval
        self.evaluate_metric = args.evaluate_metric
        self.writer = SummaryWriter(log_dir=args.tboard_dir)
        # self.writer = SummaryWriter(logdir=args.tboard_dir)

        if args.resume != None:
            self._load_pretrain(args.resume)

        self.train_loader = args.train_loader
        self.test_loader = args.test_loader

    def train(self):
        self.train_hist = {
            'iou': [],
            'loss': [],
            'accuracy': [],
            'per_epoch_time': [],
            'total_time': []
        }
        print('training start!!')
        start_time = time.time()

        self.model.train()
        res = self.evaluate(self.start_epoch)
        if self.writer:
            self.writer.add_scalar('val/IOU', res['iou'], 0)
            self.writer.add_scalar('val/Loss', res['loss'], 0)
            self.writer.add_scalar('val/Accuracy', res['accuracy'], 0)
        print(f'Evaluation: Epoch 0: Loss {res["loss"]}, Accuracy {res["accuracy"]}, IOU {res["iou"]}')
        for epoch in range(self.start_epoch, self.epoch):
            self.train_epoch(epoch)

            if (epoch + 1) % self.evaluate_interval == 0 or epoch == 0:
                res = self.evaluate(epoch + 1)
                print(f'Evaluation: Epoch {epoch + 1}: Loss {res["loss"]}, Accuracy {res["accuracy"]}, IOU {res["iou"]}')
                if res['loss'] < self.best_loss:
                    self.best_loss = res['loss']
                    self._snapshot(epoch + 1, 'best_loss')
                if res['iou'] > self.best_iou:
                    self.best_iou = res['iou']
                    self._snapshot(epoch + 1, 'best_iou')

                if self.writer:
                    self.writer.add_scalar('val/IOU', res['iou'], epoch + 1)
                    self.writer.add_scalar('val/Loss', res['loss'], epoch + 1)
                    self.writer.add_scalar('val/Accuracy', res['accuracy'], epoch + 1)

            if (epoch + 1) % self.scheduler_interval == 0:
                self.scheduler.step()

            if (epoch + 1) % self.snapshot_interval == 0:
                self._snapshot(epoch + 1)

            if self.writer:
                self.writer.add_scalar('train/Learning Rate', self._get_lr(), epoch + 1)
                self.writer.add_scalar('train/IOU', self.train_hist['iou'][-1], epoch + 1)
                self.writer.add_scalar('train/Loss', self.train_hist['loss'][-1], epoch + 1)
                self.writer.add_scalar('train/Accuracy', self.train_hist['accuracy'][-1], epoch + 1)

                # finish all epoch
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

    def train_epoch(self, epoch):
        data_timer, model_timer = Timer(), Timer()
        loss_meter, acc_meter, iou_meter = AverageMeter(), AverageMeter(), AverageMeter()
        num_iter = int(len(self.train_loader.dataset) // self.train_loader.batch_size)
        train_loader_iter = self.train_loader.__iter__()
        # for iter, inputs in enumerate(self.train_loader):
        for iter in range(num_iter):
            data_timer.tic()
            inputs = train_loader_iter.next()
            for k, v in inputs.items():  # load inputs to device.
                if type(v) == list:
                    inputs[k] = [item.to(self.device) for item in v]
                else:
                    inputs[k] = v.to(self.device)
            data_timer.toc()

            model_timer.tic()
            # forward
            self.optimizer.zero_grad()
            predict = self.model(inputs)
            labels = inputs['labels'].long().squeeze()
            loss = self.evaluate_metric(predict, labels)
            # acc = torch.sum(torch.max(predict, dim=1)[1].int() == labels.int()) * 100 / predict.shape[0]
            acc = calculate_acc(predict, labels)
            part_iou = calculate_iou(predict, labels, stack_lengths=inputs['stack_lengths'][0], n_parts=self.config.num_classes)
            iou = np.mean(part_iou)

            # backward
            loss.backward()
            self.optimizer.step()
            model_timer.toc()
            loss_meter.update(float(loss))
            iou_meter.update(float(iou))
            acc_meter.update(float(acc))

            if (iter + 1) % 1 == 0 and self.verbose:
                print(f"Epoch: {epoch+1} [{iter+1:4d}/{num_iter}] "
                      f"loss: {loss_meter.avg:.2f} "
                      f"iou: {iou_meter.avg:.2f} "
                      f"data time: {data_timer.avg:.2f}s "
                      f"model time: {model_timer.avg:.2f}s")
        # finish one epoch
        epoch_time = model_timer.total_time + data_timer.total_time
        self.train_hist['per_epoch_time'].append(epoch_time)
        self.train_hist['loss'].append(loss_meter.avg)
        self.train_hist['accuracy'].append(acc_meter.avg)
        self.train_hist['iou'].append(iou_meter.avg)
        print(f'Epoch {epoch+1}: Loss : {loss_meter.avg:.2f}, Accuracy: {acc_meter.avg:.2f}, IOU: {iou_meter.avg:.2f}, time {epoch_time:.2f}s')

    def evaluate(self, epoch):
        data_timer, model_timer = Timer(), Timer()
        loss_meter, acc_meter, iou_meter = AverageMeter(), AverageMeter(), AverageMeter()
        num_iter = int(len(self.test_loader.dataset) // self.train_loader.batch_size)
        test_loader_iter = self.train_loader.__iter__()
        for iter in range(num_iter):
            data_timer.tic()
            inputs = test_loader_iter.next()
            for k, v in inputs.items():  # load inputs to device.
                if type(v) == list:
                    inputs[k] = [item.to(self.device) for item in v]
                else:
                    inputs[k] = v.to(self.device)
            data_timer.toc()

            model_timer.tic()
            predict = self.model(inputs)
            labels = inputs['labels'].long().squeeze()
            loss = self.evaluate_metric(predict, labels)
            # acc = torch.sum(torch.max(predict, dim=1)[1].int() == labels.int()) * 100 / predict.shape[0]
            acc = calculate_acc(predict, labels)
            part_iou = calculate_iou(predict, labels, stack_lengths=inputs['stack_lengths'][0], n_parts=self.config.num_classes)
            iou = np.mean(part_iou)

            model_timer.toc()
            loss_meter.update(float(loss))
            iou_meter.update(float(iou))
            acc_meter.update(float(acc))

            if (iter + 1) % 1 == 0 and self.verbose:
                print(f"Eval epoch: {epoch+1} [{iter+1:4d}/{num_iter}] "
                      f"loss: {loss_meter.avg:.2f} "
                      f"iou: {iou_meter.avg:.2f} "
                      f"data time: {data_timer.avg:.2f}s "
                      f"model time: {model_timer.avg:.2f}s")
        self.model.train()
        res = {
            'iou': iou_meter.avg,
            'loss': loss_meter.avg,
            'accuracy': acc_meter.avg
        }
        return res

    def _snapshot(self, epoch, name=None):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_iou': self.best_iou,
        }
        if name is None:
            filename = os.path.join(self.save_dir, f'model_{epoch}.pth')
        else:
            filename = os.path.join(self.save_dir, f'model_{epoch}_{name}.pth')
        print(f"Save model to {filename}")
        torch.save(state, filename)

    def _load_pretrain(self, resume):
        if os.path.isfile(resume):
            print(f"=> loading checkpoint {resume}")
            state = torch.load(resume)
            self.start_epoch = state['epoch']
            self.model.load_state_dict(state['state_dict'])
            self.scheduler.load_state_dict(state['scheduler'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.best_loss = state['best_loss']
            self.best_acc = state['best_acc']
        else:
            raise ValueError(f"=> no checkpoint found at '{resume}'")

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']
