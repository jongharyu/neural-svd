import os
import random
from collections import Counter
from shutil import copyfile

import faiss
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def inner_product_distances(x, y):
    return - x.T @ y


class SketchyRetrieval:
    def __init__(self, test_loader,
                 n_images_to_save=10, n_retrievals=100,
                 metric='inner_product',
                 run_path=None,
                 device=None):
        self.test_loader = test_loader
        self.batch_size = test_loader.batch_size
        self.n_images_to_save = n_images_to_save
        self.n_retrievals = n_retrievals
        self.metric = metric
        self.run_path = run_path
        self.device = device

        self.n_classes = Counter(self.test_loader.sketch_classes)
        tmp = []
        for i in range(len(self.test_loader.sketch_features)):
            tmp.append(self.n_classes[self.test_loader.sketch_classes[i]])
        self.n_classes_items = np.array(tmp)

    @staticmethod
    def parse_class(path):
        return path.split('/')[-2]

    def evaluate(self, model_x, model_y, epoch,
                 save_retrieved_images=False, ap_ver=1, tag='',
                 return_map_all=False):
        sketch_features = self.test_loader.sketch_features
        photo_features = self.test_loader.photo_features
        sketch_classes = self.test_loader.sketch_classes
        photo_classes = self.test_loader.photo_classes

        # Step 1: Convert all test sketches (X) to shared representation.
        zxs = []
        for step in tqdm(range(np.ceil(sketch_features.shape[0] / self.batch_size).astype(int))):
            sketch_batch = torch.Tensor(sketch_features[step * self.batch_size: (step + 1) * self.batch_size]).to(self.device)
            zx = model_x(sketch_batch)
            zxs.append(zx)
        zxs = torch.cat(zxs, dim=0).detach().cpu().numpy()

        # Step 2: Convert all test photos (Y) to shared representation.
        zys = []
        for step in tqdm(range(np.ceil(photo_features.shape[0] / self.batch_size).astype(int))):
            photo_batch = torch.Tensor(photo_features[step * self.batch_size: (step + 1) * self.batch_size]).to(self.device)
            zy = model_y(photo_batch)
            zys.append(zy)
        zys = torch.cat(zys, dim=0).detach().cpu().numpy()

        # Step 3: compute Precision@K
        relevances_K, _ = self.get_retrievals(zxs, zys,
                                              xclss=sketch_classes, yclss=photo_classes,
                                              K=self.n_retrievals,
                                              metric=self.metric,
                                              package='faiss' if self.metric == 'inner_product' else 'sklearn')
        precision_Ks = self.compute_precisions_at_k(relevances_K)  # (n_queries, )
        precision_K = precision_Ks.mean()
        print(f'{tag}\tP@{self.n_retrievals} ({self.metric})\t{precision_K:.4f}')

        # Step 4: compute mAP@all
        average_precisions = np.array(0.)
        if return_map_all or save_retrieved_images:
            # Note: for computing mAP@K, see https://stackoverflow.com/questions/54966320/mapk-computation
            relevances, retrieved_zys_idxs = self.get_retrievals(zxs, zys,
                                                                 xclss=sketch_classes, yclss=photo_classes,
                                                                 metric=self.metric,
                                                                 package='faiss' if self.metric == 'inner_product' else 'sklearn')
            average_precisions = self.compute_average_precisions(relevances, self.n_classes_items, ver=ap_ver)  # (n_queries, )
            mAP = average_precisions.mean()
            print(f'{tag}\tmAP ({self.metric})\t{mAP:.4f}')

            # Step 5: (optional) save retrieved images
            if save_retrieved_images:
                self.save_retrieved_images(retrieved_zys_idxs, epoch, tag=tag)

        return precision_Ks, average_precisions

    @staticmethod
    def get_retrievals(zxs, zys, xclss, yclss, K=None,
                       package='faiss', metric='euclidean'):
        assert package in ['faiss', 'sklearn']
        if K is None:
            # if K is not specified, evaluate with respect to the entire retrievals
            K = zys.shape[0]

        # find nearest neighbors of zxs (query sketches) with respect to zys (photos)
        # Note: (n_queries, K) matrix 'relevances' contains every information for computing
        #       "precision@K" and "average precision" for each query
        #       relevances[i, j] = (j-th retrieval is relevant for query i)
        if package == 'sklearn':
            nbrs = NearestNeighbors(n_neighbors=K, metric=metric, algorithm='auto').fit(zys)
            _, retrieved_zys_idxs = nbrs.kneighbors(zxs)
        elif package == 'faiss':
            if metric == 'euclidean':
                index = faiss.IndexFlatL2(zxs.shape[1])
            elif metric == 'inner_product':
                index = faiss.IndexFlatIP(zxs.shape[1])
            else:
                raise NotImplementedError

            index.add(zys)
            _, retrieved_zys_idxs = index.search(zxs, K)  # actual search
        else:
            raise NotImplementedError

        retrieved_yclss = yclss[retrieved_zys_idxs]
        relevances = (retrieved_yclss == xclss[:, np.newaxis])  # (n_queries, K)

        return relevances, retrieved_zys_idxs

    def save_retrieved_images(self, retrieved_zys_idxs, epoch, tag=''):
        path_retrievals = os.path.join(self.run_path, 'retrievals', f'e{epoch:03d}')

        sketch_paths = self.test_loader.sketch_paths
        photo_paths = self.test_loader.photo_paths

        sketch_idx_per_class = self.test_loader.sketch_idx_per_class
        sketch_classes = self.test_loader.sketch_classes

        path_sketch = self.test_loader.path_sketch
        path_photo = self.test_loader.path_photo

        retrieved_paths = photo_paths[retrieved_zys_idxs[..., :self.n_images_to_save]]
        for cls in tqdm(sorted(set(sketch_classes.tolist())),
                        desc='Saving retrieved images for sketch sample queries...'):
            # create folder
            path_retrievals_per_class = os.path.join(path_retrievals, f'{tag}_{cls}')
            os.makedirs(path_retrievals_per_class, exist_ok=True)

            # given a class, pick one query sketch image at random
            sketch_idx = random.choice(sketch_idx_per_class[cls])  # alternative: sketch_idx_per_class[label][0]

            # find the path of the query sketch image
            query_path = os.path.join(path_sketch, sketch_paths[sketch_idx])
            assert os.path.exists(query_path)  # sanity check
            copyfile(query_path, os.path.join(path_retrievals_per_class, 'query.jpg'))  # save the query image

            # for the selected query sketch image
            for rank, retrieved_path in enumerate(retrieved_paths[sketch_idx]):
                abs_retrieved_path = os.path.join(path_photo, retrieved_path)
                assert os.path.exists(abs_retrieved_path), 'Error: retrieved photo does not exist!'
                match_suffix = '' if int(self.parse_class(retrieved_path) == cls) else '_f'
                copyfile(abs_retrieved_path,
                         os.path.join(path_retrievals_per_class, f'{rank}{match_suffix}.jpg'))
        else:
            print('done!')

    @staticmethod
    def compute_precisions_at_k(relevances):
        # relevances: (n_queries, K)
        # P@K = (correct retrievals) / K
        return relevances.mean(axis=1)  # (n_queries, )

    @staticmethod
    def compute_average_precisions(relevances, n_relevant_items, ver=1):
        # compute average precision @ K (AP@K) for each query
        # relevances: (n_queries, K)
        # n_relevant_items: (n_queries, )
        #                   for each query, how many items are there in the dataset that are relevant to the query?
        if ver == 1:
            # ver1
            n_queries = relevances.shape[0]
            # Note: precs[i, K] = Precision@K's for query i
            precs = relevances.cumsum(axis=1) / np.ones_like(relevances).cumsum(axis=1)  # (n_queries, K)
            # perform "optimistic interpolation", a convention in information retrieval
            max_precs = np.maximum.accumulate(precs[..., ::-1], axis=1)[..., ::-1]
            avg_precs = np.zeros(precs.shape[0])
            for i in range(n_queries):
                avg_precs[i] = max_precs[i][relevances[i] == 1].sum() / relevances[i].sum()
            return avg_precs  # (n_queries, )

        elif ver == 2:
            # ver2: https://stackoverflow.com/questions/54966320/mapk-computation
            K = relevances.shape[1]
            # Note: precs[i, K] = Precision@K's for query i
            precs = relevances.cumsum(axis=1) / np.ones_like(relevances).cumsum(axis=1)  # (n_queries, K)
            avg_precs = (precs * relevances).sum(-1) / np.minimum(K, n_relevant_items)
            return avg_precs  # (n_queries, )

        elif ver == 3:
            # ver3: from IIAE / CVAE (ECCV 2018)
            mAP_K_term = 1.0 / np.stack([np.arange(1, relevances.shape[1] + 1) for _ in range(relevances.shape[0])], axis=0)
            mAP_K = np.sum(map_change(relevances) * mAP_K_term, axis=1)
            gt_cnts = relevances.sum(axis=-1)
            assert gt_cnts.shape == mAP_K.shape
            return mAP_K / gt_cnts  # (n_queries, )


def map_change(arr):
    dup = np.copy(arr)
    for idx in range(arr.shape[1]):
        if idx != 0:
            dup[:, idx] = dup[:, idx - 1] + dup[:, idx]
    return np.multiply(dup, arr)
