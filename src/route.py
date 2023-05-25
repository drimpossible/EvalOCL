import numpy as np
import h5py
import random
import multiprocessing
from tqdm import tqdm
from multiprocessing import Pool
from itertools import cycle, islice


def sampler(train_bs, stream_bs, curr_stream_idx, num_repeats, method="uniform"):
    '''
    This is a CPU parallelizable training index generation code.
    Three options for generation: 1. Uniform, 2. FIFO, 3. Mixed
    TODO: Fix Uniform when train_bs > stream_bs at first step this gives an error, we can simply force the samples at i=0.
    '''
    # assuming at start curr_stream_idx = 0
    # then sampling range is 0,stream_bs
    sampling_range = (curr_stream_idx+1)*stream_bs

    # this condition guarantees that we always have train_bs number of items for the training indices.
    # important for first few timesteps 
    if train_bs > sampling_range:
        index_list = np.arange(0,sampling_range).tolist()
        index_list = list(islice(cycle(index_list),train_bs))
    else:
        if method == "uniform":
            stream_indices = random.sample(range(0, sampling_range), num_repeats * train_bs)
        elif method == "fifo":
            stream_indices = (sampling_range-(num_repeats)*train_bs + np.arange(0, num_repeats*train_bs)).tolist()
        elif method =="mixed":
            stream_indices_uniform = random.sample(range(0, sampling_range), num_repeats *train_bs//2)
            stream_indices_fifo = (sampling_range-num_repeats*train_bs//2 + np.arange(0, num_repeats*train_bs//2)).tolist()
            stream_indices = []
            for i in range(num_repeats):
                stream_indices += stream_indices_uniform[i*train_bs//2:(i+1)*train_bs//2]+stream_indices_fifo[i*train_bs//2:(i+1)*train_bs//2]
    return stream_indices


def full_episode(train_bs, stream_bs, stream_len, num_repeats, method="uniform", num_workers=32):
    episode_samples = []
    pool = Pool(num_workers)
    results = []
    for timestep in range(stream_len // stream_bs):
        results.append(pool.apply_async(sampler, args=(train_bs, stream_bs, timestep, num_repeats, method)))
    pbar = tqdm(total=len(results))
    for i, result in enumerate(results):
        episode_samples += result.get()
        pbar.update(1)
    pbar.close()
    pool.close()
    pool.join()
    return episode_samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Routing Generation')
    parser.add_argument('--train_bs', default=128, type=int)
    parser.add_argument('--stream_bs', default=64, type=int)
    parser.add_argument('--num_gd', default=1, type=int)
    parser.add_argument('--stream_size', default=38003083, type=int)
    parser.add_argument('--method', type=str)
    args = parser.parse_args()

    complete_index_list = full_episode(args.train_bs, args.stream_bs, args.stream_size, args.num_gd, method=args.method)
    store_list = h5py.File(f"2k_{args.train_bs}_{args.stream_bs}_{args.method}_{args.num_gd}.hdf5", "w")
    store_list.create_dataset("store_list", data=complete_index_list)
    store_list.close()
