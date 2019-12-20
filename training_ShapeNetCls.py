import os
import time
import shutil
from datasets.ShapeNet import ShapeNetDataset
from utils.config import Config
from trainer_cls import Trainer
from models.KPCNN import KPCNN
from datasets.dataloader import get_dataloader
from torch import optim
from torch import nn
import torch


class ShapeNetPartConfig(Config):
    # dataset
    dataset = 'ShapeNetCls'
    num_classes = 16
    first_subsampling_dl = 0.02
    in_features_dim = 4
    data_train_dir = "./data/shapenetcore_partanno_segmentation_benchmark_v0"
    data_test_dir = "./data/shapenetcore_partanno_segmentation_benchmark_v0"
    train_batch_size = 16
    test_batch_size = 16

    # model
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'global_average'
                    ]
    dropout = 0.5
    resume = None
    use_batch_norm = True
    batch_norm_momentum = 0.02
    # https://github.com/pytorch/examples/issues/289 pytorch bn momentum 0.02 == tensorflow bn momentum 0.98

    # kernel point convolution
    KP_influence = 'linear'
    KP_extent = 1.0
    convolution_mode = 'sum'
    modulated = False

    # training
    max_epoch = 200
    learning_rate = 1e-2
    momentum = 0.98
    exp_gamma = 0.1 ** (1 / 80)
    exp_interval = 1


class Args(object):
    def __init__(self, config):
        is_test = False
        if is_test:
            self.experiment_id = "KPConvNet" + time.strftime('%m%d%H%M') + 'Test'
        else:
            self.experiment_id = "KPConvNet" + time.strftime('%m%d%H%M')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = True

        # snapshot
        self.snapshot_interval = 5
        snapshot_root = f'snapshot/{config.dataset}_{self.experiment_id}'
        tensorboard_root = f'tensorboard/{config.dataset}_{self.experiment_id}'
        os.makedirs(snapshot_root, exist_ok=True)
        os.makedirs(tensorboard_root, exist_ok=True)
        shutil.copy2(os.path.join('.', 'training_ShapeNetCls.py'), os.path.join(snapshot_root, 'train.py'))
        shutil.copy2(os.path.join('datasets', 'ShapeNet.py'), os.path.join(snapshot_root, 'dataset.py'))
        shutil.copy2(os.path.join('datasets', 'dataloader.py'), os.path.join(snapshot_root, 'dataloader.py'))
        self.save_dir = os.path.join(snapshot_root, 'models/')
        self.result_dir = os.path.join(snapshot_root, 'results/')
        self.tboard_dir = tensorboard_root

        # dataset & dataloader
        self.train_set = ShapeNetDataset(root=config.data_train_dir,
                                         split='train',
                                         first_subsampling_dl=config.first_subsampling_dl,
                                         classification=True,
                                         config=config,
                                         )
        self.test_set = ShapeNetDataset(root=config.data_test_dir,
                                        split='test',
                                        first_subsampling_dl=config.first_subsampling_dl,
                                        classification=True,
                                        config=config,
                                        )
        self.train_loader = get_dataloader(dataset=self.train_set,
                                           batch_size=config.train_batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           )
        self.test_loader = get_dataloader(dataset=self.test_set,
                                          batch_size=config.test_batch_size,
                                          shuffle=False,
                                          num_workers=4,
                                          )
        print("Training set size:", self.train_loader.dataset.__len__())
        print("Test set size:", self.test_loader.dataset.__len__())

        # model
        self.model = KPCNN(config)
        self.resume = config.resume
        # optimizer 
        self.start_epoch = 0
        self.epoch = config.max_epoch
        self.optimizer = optim.SGD(self.model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config.exp_gamma)
        self.scheduler_interval = config.exp_interval

        # evaluate
        self.evaluate_interval = 1
        self.evaluate_metric = nn.CrossEntropyLoss(reduction='mean')

        self.check_args()

    def check_args(self):
        """checking arguments"""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.exists(self.tboard_dir):
            os.makedirs(self.tboard_dir)
        return self


if __name__ == '__main__':
    config = ShapeNetPartConfig()
    args = Args(config)
    trainer = Trainer(args)
    trainer.train()
