'''svg -> pytorch format '''
import sys

sys.path.append("./")

import os
import shutil
import argparse
from glob import glob
from functools import partial
from multiprocessing import Pool

from cad_transformer.Pre.utils_dataset import init_worker
from cad_transformer.Method.dist import svg2graph


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser = argparse.ArgumentParser(description='construct graph')
    parser.add_argument('-i',
                        '--input_dir',
                        type=str,
                        help='the input svg directory',
                        required=True)
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        help='the output npy directory',
                        required=True)
    parser.add_argument('-d',
                        '--max_degree',
                        type=int,
                        help='the maximum neighbor number of each node',
                        default=128)
    parser.add_argument('-v',
                        '--visualize',
                        type=bool,
                        help='the visualize flag',
                        default=False)
    parser.add_argument('--thread_num',
                        type=int,
                        help='multiprocess number',
                        default=os.cpu_count())
    args = parser.parse_args()
    return args


def main():
    '''Main entrance
    '''
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = os.path.join(args.output_dir, './temp')
    os.makedirs(temp_dir, exist_ok=True)
    if args.visualize:
        os.makedirs(os.path.join(args.output_dir, './visualize'),
                    exist_ok=True)

    print(f'> svg -> npy')
    svg_paths = sorted(glob(os.path.join(args.input_dir, "*.svg")))
    partial_func = partial(svg2graph,
                           output_dir=args.output_dir,
                           max_degree=args.max_degree,
                           visualize=args.visualize)
    p = Pool(args.thread_num, init_worker)
    try:
        p.map(partial_func, svg_paths)
    except KeyboardInterrupt:
        print("....\nCaught KeyboardInterrupt, terminating workers")
        p.terminate()
    else:
        p.close()
    p.join()

    shutil.rmtree(temp_dir)
    return True


if __name__ == '__main__':
    main()
