#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import random
import threading
import numpy as np
import queue as Queue
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from glob import glob
from pdb import set_trace as st
from torch.utils.data import Dataset, DataLoader

from cad_transformer.Config.image_net import IMAGENET_MEAN, IMAGENET_STD


class CADDataset(Dataset):

    def __init__(self, split='train', do_norm=True, cfg=None, max_prim=12000):
        self.set_random_seed(123)
        self.root = cfg.root
        self.split = split
        self.max_prim = max_prim
        if cfg is not None:
            self.clus_num_per_batch = cfg.clus_num_per_batch
            self.nn = cfg.clus_nn
            self.size = cfg.img_size
            self.filter_num = cfg.filter_num
            self.aug_ratio = cfg.aug_ratio
            # FIXME: why rgb_dim is 32?
            self.rgb_dim = cfg.rgb_dim
            #  self.rgb_dim = 0
        else:
            self.clus_num_per_batch = 16
            self.nn = 64
            self.size = 700
            self.filter_num = 64
            self.aug_ratio = 0.5
            self.rgb_dim = 0
        # transformations
        transform = [T.ToTensor()]
        if do_norm:
            transform.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        self.transform = T.Compose(transform)

        # pre-loading
        self.image_path_list = glob(
            os.path.join(self.root, "png", split, "*.png"))
        self.anno_path_list = glob(
            os.path.join(self.root, "npy", split, "*.npy"))
        self.image_path_list = sorted(self.image_path_list)
        self.anno_path_list = sorted(self.anno_path_list)

        # data augmentation
        self.train_len = len(self.anno_path_list)
        if ("train" in split) and (self.aug_ratio >= 1e-4):
            print(f" > before aug training: {len(self.anno_path_list)}")
            self.aug_training()
            print(f" > after aug training: {len(self.anno_path_list)}")

        assert len(self.image_path_list) == len(self.anno_path_list)
        self.length = len(self.image_path_list)

        print(" > before filter_smallset:", len(self.anno_path_list))
        self.image_path_list, self.anno_path_list = self.filter_smallset()

        self.length = len(self.image_path_list)
        print(" > after filter_smallset:", len(self.anno_path_list))
        return

    def filter_smallset_test(self):
        '''
        for test gpu memory only
        '''
        anno_path_list_new = []
        image_path_list_new = []
        for idx, ann_path in tqdm(enumerate(self.anno_path_list),
                                  total=len(self.anno_path_list)):
            adj_node_classes = np.load(ann_path, \
                                    allow_pickle=True).item()
            target = adj_node_classes["cat"]

            if int(self.max_prim * 1.1) >= len(target) >= self.max_prim:
                anno_path_list_new.append(self.anno_path_list[idx])
                image_path_list_new.append(self.image_path_list[idx])
                break
        return image_path_list_new, anno_path_list_new

    def filter_smallset(self):
        #  return self.filter_smallset_test()

        anno_path_list_new = []
        image_path_list_new = []
        for idx, ann_path in tqdm(enumerate(self.anno_path_list),
                                  total=len(self.anno_path_list)):
            if len(anno_path_list_new) > 10:
                break
            adj_node_classes = np.load(ann_path, \
                                    allow_pickle=True).item()
            target = adj_node_classes["cat"]

            if self.max_prim >= len(target) >= self.filter_num:
                anno_path_list_new.append(self.anno_path_list[idx])
                image_path_list_new.append(self.image_path_list[idx])
        return image_path_list_new, anno_path_list_new

    def __len__(self):
        return self.length

    def _get_item(self, index):
        img_path = self.image_path_list[index]
        ann_path = self.anno_path_list[index]
        assert os.path.basename(img_path).split(".")[0] == \
            os.path.basename(ann_path).split(".")[0]

        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.size, self.size))
        image = self.transform(image).cuda()

        adj_node_classes = np.load(ann_path, \
                            allow_pickle=True).item()
        target = adj_node_classes["cat"]
        target = torch.from_numpy(np.array(target, dtype=np.int64)).cuda()

        center = adj_node_classes["ct_norm"]
        xy = torch.from_numpy(np.array(center, dtype=np.float32)).cuda()

        if self.rgb_dim > 0:
            #  rgb_npy_path = ann_path.replace('/npy/', '/npy_rgb/')
            #  rgb_info = np.load(rgb_npy_path, allow_pickle=True).item()['rgbs']
            rgb_info = adj_node_classes['ct_norm']
            rgb_info = torch.from_numpy(np.array(rgb_info,
                                                 dtype=np.int64)).cuda()
        else:
            rgb_info = xy

        nns = adj_node_classes["nns"]
        nns = torch.from_numpy(np.array(nns, dtype=np.int64)).cuda()

        instance = adj_node_classes["inst"]
        instance_center = self.get_instance_center_tensor(instance,
                                                          center,
                                                          semantic=target,
                                                          img_path=img_path)
        instance = torch.from_numpy(np.array(instance,
                                             dtype=np.float32)).cuda()
        offset = xy - instance_center

        indexes = torch.Tensor([1]).cuda()
        basename = os.path.basename(img_path)

        return image, xy, target, rgb_info, nns, offset, instance, indexes, basename

    def __getitem__(self, index):
        return self._get_item(index)

    def random_sample(self, image, xy, target, rgb_info, nns, offset, instance,
                      indexes, basename):
        length = xy.shape[0]
        rand_idx = random.sample(range(length), self.max_prim)
        rand_idx = sorted(rand_idx)
        xy = xy[rand_idx]
        target = target[rand_idx]
        rgb_info = rgb_info[rand_idx]
        nns = nns[rand_idx]
        offset = offset[rand_idx]
        instance = instance[rand_idx]
        return image, xy, target, rgb_info, nns, offset, instance, indexes, basename

    def set_random_seed(self, seed, deterministic=False):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def aug_training(self):
        self.image_path_list_aux = glob(
            os.path.join(self.root, "images", "{}_aug5x".format(self.split),
                         "images", "*.png"))
        self.anno_path_list_aux = glob(
            os.path.join(self.root, "annotations",
                         "{}_aug5x".format(self.split),
                         "constructed_graphs_withnninst", "*.npy"))
        self.image_path_list_aux = sorted(self.image_path_list_aux)
        self.anno_path_list_aux = sorted(self.anno_path_list_aux)
        try:
            assert len(self.image_path_list_aux) == len(
                self.anno_path_list_aux)
        except:

            def extra_same_elem(list1, list2):
                set1 = set(list1)
                set2 = set(list2)
                iset = set1.intersection(set2)
                return list(iset)

            img_list = [
                os.path.basename(x).split(".")[0]
                for x in self.image_path_list_aux
            ]
            ann_list = [
                os.path.basename(x).split(".")[0]
                for x in self.anno_path_list_aux
            ]
            intersect = extra_same_elem(img_list, ann_list)
            img_dir = os.path.dirname(self.image_path_list_aux[0])
            ann_dir = os.path.dirname(self.anno_path_list_aux[0])
            self.image_path_list_aux = [
                os.path.join(img_dir, "{}.png".format(x)) for x in intersect
            ]
            self.anno_path_list_aux = [
                os.path.join(ann_dir, "{}.npy".format(x)) for x in intersect
            ]
            assert len(self.image_path_list_aux) == len(
                self.anno_path_list_aux)

        aux_len = len(self.anno_path_list_aux)
        aug_n = int(self.aug_ratio * self.train_len)
        aug_n = min(aug_n, aux_len)
        idxes = random.sample(range(0, aux_len - 1), aug_n)
        self.image_path_list_aux = [self.image_path_list_aux[i] for i in idxes]
        self.anno_path_list_aux = [self.anno_path_list_aux[i] for i in idxes]
        self.image_path_list.extend(self.image_path_list_aux)
        self.anno_path_list.extend(self.anno_path_list_aux)

    def get_instance_center_tensor(self,
                                   instance,
                                   center,
                                   semantic=None,
                                   img_path=None):
        offset_list = []
        offset_dict = {}
        for idx, inst_num in enumerate(instance):
            inst_val = inst_num[0]
            if inst_val == -1:
                continue
            if inst_val in offset_dict.keys():
                offset_dict[inst_val]["cent"].append(center[idx])
            else:
                offset_dict[inst_val] = {}
                offset_dict[inst_val]["mean"] = None
                offset_dict[inst_val]["cent"] = []
                offset_dict[inst_val]["cent"].append(center[idx])

        for idx, inst_num in enumerate(instance):
            inst_val = inst_num[0]
            if inst_val != -1:
                offset_dict[inst_val]["mean"] = np.mean(
                    offset_dict[inst_val]["cent"], axis=0)

        for idx, inst_num in enumerate(instance):
            inst_val = inst_num[0]
            if inst_val is None:
                st()
            if inst_val == -1 or inst_val is None:
                offset_list.append([-999, -999])
            else:
                try:
                    offset_list.append([
                        offset_dict[inst_val]["mean"][0],
                        offset_dict[inst_val]["mean"][1]
                    ])
                except:
                    st()
        instance_center = torch.from_numpy(
            np.array(offset_list, dtype=np.float32)).cuda()
        return instance_center


class BackgroundGenerator(threading.Thread):

    def __init__(self, generator, local_rank, max_prefetch=64):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class CADDataLoader(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(CADDataLoader, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(CADDataLoader, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                if isinstance(self.batch[k], torch.Tensor):
                    self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                     non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch
