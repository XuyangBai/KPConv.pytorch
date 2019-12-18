import os
import os.path
import numpy as np
import time
import pickle
from datasets.common import grid_subsampling
import torch.utils.data as data


class ModelNetDataset(data.Dataset):
    classification = True

    def __init__(self,
                 root,
                 split='train',
                 first_subsampling_dl=0.03,
                 config=None,
                 data_augmentation=True):
        self.config = config
        self.first_subsampling_dl = first_subsampling_dl
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.points, self.normals, self.labels = [], [], []

        # Dict from labels to names
        self.label_to_names = {0: 'airplane',
                               1: 'bathtub',
                               2: 'bed',
                               3: 'bench',
                               4: 'bookshelf',
                               5: 'bottle',
                               6: 'bowl',
                               7: 'car',
                               8: 'chair',
                               9: 'cone',
                               10: 'cup',
                               11: 'curtain',
                               12: 'desk',
                               13: 'door',
                               14: 'dresser',
                               15: 'flower_pot',
                               16: 'glass_box',
                               17: 'guitar',
                               18: 'keyboard',
                               19: 'lamp',
                               20: 'laptop',
                               21: 'mantel',
                               22: 'monitor',
                               23: 'night_stand',
                               24: 'person',
                               25: 'piano',
                               26: 'plant',
                               27: 'radio',
                               28: 'range_hood',
                               29: 'sink',
                               30: 'sofa',
                               31: 'stairs',
                               32: 'stool',
                               33: 'table',
                               34: 'tent',
                               35: 'toilet',
                               36: 'tv_stand',
                               37: 'vase',
                               38: 'wardrobe',
                               39: 'xbox'}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

        t0 = time.time()
        # Load wanted points if possible
        print(f'\nLoading {split} points')
        filename = os.path.join(self.root, f'{split}_{first_subsampling_dl:.3f}_record.pkl')
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                self.points, self.normals, self.labels = pickle.load(file)
        else:
            # Collect training file names
            names = np.loadtxt(os.path.join(self.root, f'modelnet40_{split}.txt'), dtype=np.str)

            # Collect point clouds
            for i, cloud_name in enumerate(names):

                # Read points
                class_folder = '_'.join(cloud_name.split('_')[:-1])
                txt_file = os.path.join(self.root, class_folder, cloud_name) + '.txt'
                data = np.loadtxt(txt_file, delimiter=',', dtype=np.float32)

                # Subsample them
                if first_subsampling_dl > 0:
                    points, normals = grid_subsampling(data[:, :3],
                                                       features=data[:, 3:],
                                                       sampleDl=first_subsampling_dl)
                else:
                    points = data[:, :3]
                    normals = data[:, 3:]

                # Add to list
                self.points += [points]
                self.normals += [normals]

            # Get labels
            label_names = ['_'.join(name.split('_')[:-1]) for name in names]
            self.labels = np.array([self.name_to_label[name] for name in label_names])

            # Save for later use
            with open(filename, 'wb') as file:
                pickle.dump((self.points, self.normals, self.labels), file)

        lengths = [p.shape[0] for p in self.points]
        sizes = [l * 4 * 6 for l in lengths]
        print('{:.1f} MB loaded in {:.1f}s'.format(np.sum(sizes) * 1e-6, time.time() - t0))

    def __getitem__(self, index):
        points, normals, labels = self.points[index], self.normals[index], self.labels[index]

        if self.data_augmentation and self.split == 'train':
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            # TODO: why only rotate the x and z axis??
            points[:, [0, 2]] = points[:, [0, 2]].dot(rotation_matrix)  # random rotation
            points += np.random.normal(0, 0.001, size=points.shape)  # random jitter

        if self.config.in_features_dim == 1:
            features = np.ones([points.shape[0], 1])
        elif self.config.in_features_dim == 4:
            features = np.ones([points.shape[0], 1])
            features = np.concatenate([features, points], axis=1)

        return points, features, labels

    def __len__(self):
        return len(self.points)


if __name__ == '__main__':
    datapath = "./data/modelnet40_normal_resampled/"
    from training_ModelNet import ModelNetConfig

    config = ModelNetConfig()

    print("Segmentation task:")
    dset = ModelNetDataset(root=datapath, config=config, first_subsampling_dl=0.01)
    input = dset[0]

    from datasets.dataloader import get_dataloader

    dataloader = get_dataloader(dset, batch_size=2)
    for iter, input in enumerate(dataloader):
        print(input)
        break
