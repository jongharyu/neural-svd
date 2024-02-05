# -*- coding: utf-8 -*-
import argparse
import os
import random
from collections import defaultdict
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


class SketchyVGGDataLoader:
    def __init__(self, batch_size, shuffle=True, drop_last=False,
                 root_path='..', split=1, train_or_test='train'):
        assert train_or_test in ['train', 'test', 'valid']
        self.path_sketchy = os.path.join(root_path, 'data', 'Sketchy')
        self.path_sketch = os.path.join(self.path_sketchy, 'sketch', 'tx_000000000000')  # 75,471 sketches
        self.path_photo = os.path.join(self.path_sketchy, 'extended_photo')  # 12,500 + 60,502 = 73,002 photos

        assert not drop_last
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.root_path = root_path
        self.split = split
        self.train_or_test = train_or_test

        self.sketch_features, self.sketch_classes, self.sketch_paths, self.sketch_idx_per_class = \
            self.load_sketchy_features(root_path=root_path, split=split, train_or_test=train_or_test, data_type='sketch')
        self.photo_features, self.photo_classes, self.photo_paths, self.photo_idx_per_class = \
            self.load_sketchy_features(root_path=root_path, split=split, train_or_test=train_or_test, data_type='photo')

        self.sketch_features_mean = self.sketch_features.mean(axis=0, keepdims=True)
        self.sketch_features_stdev = self.sketch_features.std(axis=0, keepdims=True)
        self.photo_features_mean = self.photo_features.mean(axis=0, keepdims=True)
        self.photo_features_stdev = self.photo_features.std(axis=0, keepdims=True)

        self.dataset = self.sketch_features  # for compatibility
        self.max_steps = np.ceil(self.sketch_features.shape[0] / batch_size).astype(int)

        assert set(self.sketch_classes.tolist()) == set(self.photo_classes.tolist())
        self.classes = sorted(list(set(self.sketch_classes.tolist())))
        self.cls_to_num = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        self._step = 0
        return self

    def __next__(self):
        if self._step < self.max_steps:
            self._step += 1
            train_sketch_idx, train_photo_idx, train_cls = self._pick_random_pairs()
            batch_sketch = np.take(self.sketch_features, train_sketch_idx, axis=0)
            batch_photo = np.take(self.photo_features, train_photo_idx, axis=0)

            batch_sketch = torch.Tensor(batch_sketch)
            batch_photo = torch.Tensor(batch_photo)
            batch_cls = torch.Tensor(train_cls)

            return batch_sketch, batch_photo, batch_cls
        else:
            raise StopIteration

    def _pick_random_pairs(self):
        sketch_idx_list = []
        photo_idx_list = []
        cls_list = []
        random.shuffle(self.classes)
        for i, cls in enumerate(cycle(self.classes)):
            if i >= self.batch_size:
                break

            sketch_idx = random.choice(self.sketch_idx_per_class[cls])
            photo_idx = random.choice(self.photo_idx_per_class[cls])
            assert cls == self.sketch_classes[sketch_idx] == self.photo_classes[photo_idx]

            sketch_idx_list.append(sketch_idx)
            photo_idx_list.append(photo_idx)
            cls_list.append(self.cls_to_num[cls])

        return np.array(sketch_idx_list), np.array(photo_idx_list), np.array(cls_list)

    @staticmethod
    def load_sketchy_features(root_path, split, train_or_test, data_type):
        # Note: 75,479 sketches, 73,002 (12,500 + 60,502) photos
        #   Split 1 (100/25; a.k.a. SEM-PCYC split)
        #   Split 2 (104/21; a.k.a. ECCV 2018 split)
        #   Split {i}_{seed} (valid classes are generated using {seed} from Split {i})
        assert train_or_test in ['train', 'test', 'valid']
        assert data_type in ['sketch', 'photo']
        path_sketchy_features = os.path.join(root_path, 'data', 'SketchyVGG', f'split{split}')

        loaded_vars = np.load(os.path.join(path_sketchy_features, f'{train_or_test}_{data_type}.npz'))
        features = loaded_vars['features']
        paths = loaded_vars['paths']
        classes = loaded_vars['classes']
        indices_per_class = defaultdict(list)
        for i, path in enumerate(paths.tolist()):
            cls = path.split('/')[-2]
            indices_per_class[cls].append(i)

        print(f"No. {data_type} images ({train_or_test}): {features.shape[0]}")

        return features, classes, paths, indices_per_class


class VGGNetFeats(nn.Module):
    def __init__(self, pretrained=True, finetune=True):
        super(VGGNetFeats, self).__init__()
        model = models.vgg16(pretrained=pretrained)
        for param in model.parameters():
            param.requires_grad = finetune
        self.features = model.features
        self.classifier = nn.Sequential(
            *list(model.classifier.children())[:-1],
            nn.Linear(4096, 512)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class InvertImage:
    def __init__(self):
        pass

    def __call__(self, x):
        return 1 - x


def main(config):
    # cuda
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    path_sketchy = os.path.join(config.root, 'data', 'Sketchy')

    path_sketch_model = os.path.join(path_sketchy, 'pretrained', 'vgg16_sketch.pth')
    path_photo_model = os.path.join(path_sketchy, 'pretrained', 'vgg16_photo.pth')

    # Sketch model: pre-trained on ImageNet
    sketch_model = VGGNetFeats(pretrained=False, finetune=False).to(device)
    sketch_model.load_state_dict(torch.load(path_sketch_model, map_location=device)['state_dict_sketch'])

    # Photo model: pre-trained on ImageNet
    photo_model = VGGNetFeats(pretrained=False, finetune=False).to(device)
    photo_model.load_state_dict(torch.load(path_photo_model, map_location=device)['state_dict_image'])

    transform_sketch = transforms.Compose([transforms.Resize((config.image_size, config.image_size)),
                                           transforms.ToTensor(),
                                           InvertImage()])
    transform_photo = transforms.Compose([transforms.Resize((config.image_size, config.image_size)),
                                          transforms.ToTensor()])

    sketch_dataset = ImageFolder(os.path.join(path_sketchy, 'sketch', 'tx_000000000000'), transform=transform_sketch)
    photo_dataset = ImageFolder(os.path.join(path_sketchy, 'extended_photo'), transform=transform_photo)

    # all the unique classes
    assert set(sketch_dataset.classes) == set(photo_dataset.classes)
    classes = sorted(sketch_dataset.classes)

    # divide the classes
    if config.split.startswith('1'):
        # According to Shen et al., "Zero-Shot Sketch-Image Hashing", CVPR 2018.
        np.random.seed(0)
        train_classes = np.random.choice(classes, int(0.8 * len(classes)), replace=False)
        test_classes = np.setdiff1d(classes, train_classes)
        if len(config.split) == 1:
            valid_classes = []
        else:
            valid_seed = int(config.split.split('_')[-1])
            np.random.seed(valid_seed)
            valid_classes = np.random.choice(train_classes, int(0.1 * len(train_classes)), replace=False)
            train_classes = np.setdiff1d(classes, valid_classes)
    elif config.split.startswith('2'):
        # According to Yelamarthi et al., "A Zero-Shot Framework for Sketch Based Image Retrieval", ECCV 2018.
        with open(os.path.join(path_sketchy, "test_split_eccv2018.txt")) as fp:
            test_classes = fp.read().splitlines()
            train_classes = np.setdiff1d(classes, test_classes)
        if len(config.split) == 1:
            valid_classes = []
        else:
            valid_seed = int(config.split.split('_')[-1])
            np.random.seed(valid_seed)
            valid_classes = np.random.choice(train_classes, int(0.1 * len(train_classes)), replace=False)
            train_classes = np.setdiff1d(classes, valid_classes)
    else:
        raise NotImplementedError
    set_of_classes = dict()
    set_of_classes['train'] = train_classes
    set_of_classes['test'] = test_classes
    set_of_classes['valid'] = valid_classes

    path_features = os.path.join(config.root, 'data', 'SketchyVGG', f'split{config.split}')
    if not os.path.exists(path_features):
        os.makedirs(path_features)

    # 1) Compute and save sketch features
    sketch_loader = DataLoader(sketch_dataset,
                               batch_size=config.batch_size,
                               num_workers=config.num_workers,
                               shuffle=False, drop_last=False,
                               pin_memory=True)

    for train_or_test in ['train', 'test', 'valid']:
        sketch_features, sketch_classes, sketch_paths = get_features(sketch_model, sketch_loader,
                                                                     set_of_classes[train_or_test], device)
        np.savez_compressed(os.path.join(path_features, f'{train_or_test}_sketch.npz'),
                            features=sketch_features,
                            classes=sketch_classes,
                            paths=sketch_paths)

    # 2) Compute and save photo features
    photo_loader = DataLoader(photo_dataset,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              shuffle=False, drop_last=False,
                              pin_memory=True)

    for train_or_test in ['train', 'test', 'valid']:
        photo_features, photo_classes, photo_paths = get_features(photo_model, photo_loader,
                                                                  set_of_classes[train_or_test], device)
        np.savez_compressed(os.path.join(path_features, f'{train_or_test}_photo.npz'),
                            features=photo_features,
                            classes=photo_classes,
                            paths=photo_paths)


def get_features(model, dataloader, set_of_classes, device):
    dataset = dataloader.dataset
    batch_size = dataloader.batch_size
    all_classes = np.array(dataset.classes)

    features = []
    paths = []
    classes = []

    for i, batch in tqdm(enumerate(dataloader)):
        imgs, cls_idxs = batch
        fts = model(imgs.to(device))
        rel_pths = np.array([os.path.join(*path.split('/')[-2:])
                             for (path, _) in dataset.imgs[i * batch_size: (i + 1) * batch_size]])
        clss = all_classes[cls_idxs.numpy()]
        idxs = [i for (i, cls) in enumerate(clss) if cls in set_of_classes]

        features.append(fts[idxs])
        paths.append(rel_pths[idxs])
        classes.append(clss[idxs])

    features = torch.cat(features, dim=0).detach().cpu().numpy()
    classes = np.concatenate(classes, axis=0)
    paths = np.concatenate(paths, axis=0)

    return features, classes, paths


if __name__ == '__main__':
    # Parse options for processing
    parser = argparse.ArgumentParser(description='Extracting pretrained VGG16 features of Sketchy dataset')
    parser.add_argument('--no_cuda', action='store_true', help='disable CUDA use')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size for data (default: 64)')
    parser.add_argument('--root', type=str, default="..",
                        help='main path where datasets live and loggings are saved')
    parser.add_argument('--split', type=str, default='1',
                        help='split1=(SEM-PCYC); split2=(ECCV 2018)')
    parser.add_argument('--image_size', default=224, type=int, help='image size for VGG16 input')
    config = parser.parse_args()

    main(config)
