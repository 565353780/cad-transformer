import time
import torch
import argparse
import threading
import queue as Queue
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader

from cad_transformer.Config.image_net import IMAGENET_MEAN, IMAGENET_STD


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument("opts",
                        default=[],
                        nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args


class AMSoftmaxLayer(nn.Module):
    """AMSoftmaxLayer"""

    def __init__(self, in_feats, n_classes, s=30.):
        super(AMSoftmaxLayer, self).__init__()
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes),
                                    requires_grad=True)
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x):
        batch, pts_num, embed_dim = x.shape
        x = x.view(batch * pts_num, embed_dim)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm) * self.s
        costh = costh.view(batch, pts_num, -1)
        return costh


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


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
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


def imagenet_preprocess():
    return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


if __name__ == "__main__":
    xyz = torch.randn((4, 128, 2))
    npoint = 16
    aa = farthest_point_sample(xyz, npoint)
    # aa = random_point_sample(xyz, npoint)
    print(aa.shape)
