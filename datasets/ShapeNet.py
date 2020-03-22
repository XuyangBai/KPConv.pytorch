import os
import os.path
import numpy as np
import json
import open3d
from utils.pointcloud import make_point_cloud
from datasets.common import grid_subsampling
import torch.utils.data as data


class ShapeNetDataset(data.Dataset):
    # Borrow from https://github.com/fxia22/pointnet.pytorch
    def __init__(self,
                 root,
                 split='train',
                 first_subsampling_dl=0.03,
                 config=None,
                 classification=False,
                 class_choice=None,
                 data_augmentation=True):
        self.config = config
        self.first_subsampling_dl = first_subsampling_dl
        self.root = root
        self.split = split
        self.cat2id = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}

        # parse category file.
        with open(os.path.join(self.root, 'synsetoffset2category.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat2id[ls[0]] = ls[1]

        # parse segment num file.
        with open('misc/num_seg_classes.txt', 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])

        # if a subset of classes is specified.
        if class_choice is not None:
            self.cat2id = {k: v for k, v in self.cat2id.items() if k in class_choice}
        self.id2cat = {v: k for k, v in self.cat2id.items()}

        self.datapath = []
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        filelist = json.load(open(splitfile, 'r'))
        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat2id.values():
                self.datapath.append([
                    self.id2cat[category],
                    os.path.join(self.root, category, 'points', uuid + '.pts'),
                    os.path.join(self.root, category, 'points_label', uuid + '.seg')
                ])
        # if split == 'train':
        #     self.datapath = self.datapath[0:5000]
        # else:
        #     self.datapath = self.datapath[0:500]
        self.classes = dict(zip(sorted(self.cat2id), range(len(self.cat2id))))
        # print("classes:", self.classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        # print("Origin Point size:", len(point_set))
        seg = np.loadtxt(fn[2]).astype(np.int32)

        point_set, seg = grid_subsampling(point_set, labels=seg, sampleDl=self.first_subsampling_dl)

        # Center and rescale point for 1m radius
        pmin = np.min(point_set, axis=0)
        pmax = np.max(point_set, axis=0)
        point_set -= (pmin + pmax) / 2
        scale = np.max(np.linalg.norm(point_set, axis=1))
        point_set *= 1.0 / scale

        if self.data_augmentation and self.split == 'train':
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            # TODO: why only rotate the x and z axis??
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.001, size=point_set.shape)  # random jitter

        pcd = make_point_cloud(point_set)
        open3d.estimate_normals(pcd)
        normals = np.array(pcd.normals)

        if self.config.in_features_dim == 1:
            features = np.ones([point_set.shape[0], 1])
        elif self.config.in_features_dim == 4:
            features = np.ones([point_set.shape[0], 1])
            features = np.concatenate([features, point_set], axis=1)
        elif self.config.in_features_dim == 7:
            features = np.ones([point_set.shape[0], 1])
            features = np.concatenate([features, point_set, normals], axis=1)

        if self.classification:
            # manually convert numpy array to Tensor.
            # cls = torch.from_numpy(cls) - 1  # change to 0-based labels
            # cls = torch.from_numpy(np.array([cls]))
            # dict_inputs = segmentation_inputs(point_set, features, cls, self.config)
            # return dict_inputs
            return point_set, features, cls
        else:
            # manually convert numpy array to Tensor.
            # seg = torch.from_numpy(seg) - 1  # change to 0-based labels
            # dict_inputs = segmentation_inputs(point_set, features, seg, self.config)
            # return dict_inputs
            seg = seg - 1
            return point_set, features, seg

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    datapath = "./data/shapenetcore_partanno_segmentation_benchmark_v0"
    from training_ShapeNetCls import ShapeNetPartConfig

    config = ShapeNetPartConfig()

    print("Segmentation task:")
    dset = ShapeNetDataset(root=datapath, config=config, first_subsampling_dl=0.01, classification=False)
    input = dset[0]

    from datasets.dataloader import get_dataloader

    dataloader = get_dataloader(dset, batch_size=2)
    for iter, input in enumerate(dataloader):
        print(input)
        break
