# CPU-only tests for anyview/kubric_dataset.py: per-scene scale factor and per-worker sampling.
'''
Builds a few tiny unified scenes on disk (frames variant, 8x8 JPEGs), then checks:

1. scale_factor comes from each scene's metadata (specific.scale_factor, else the source
   dataset's known value), and stays 1/32 for scenes that carry neither -- the Kubric case.
2. DataLoader workers do not draw identical camera-pair / window sequences, and a new epoch
   does not replay the previous epoch's draws; with a fixed torch seed the draws are still
   reproducible run to run.

Run:  python -m pytest tests/test_kubric_dataset_cpu.py -q     (or: python tests/test_kubric_dataset_cpu.py)
'''

import json
import os
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anyview.kubric_dataset import KUBRIC_SCALE_FACTOR, KubricDVSDataset  # noqa: E402

CAMS = ['cam0', 'cam1', 'cam2', 'cam3']
T = 30          # frames per scene
WINDOW = 5      # clip length sampled by the dataset


def write_scene(path, specific=None, name='kubric5d'):
    import imageio.v2 as imageio
    os.makedirs(path, exist_ok=True)
    K = np.array([[8.0, 0, 4], [0, 8.0, 4], [0, 0, 1]], np.float32)
    for c, cam in enumerate(CAMS):
        os.makedirs(os.path.join(path, 'rgb', cam), exist_ok=True)
        os.makedirs(os.path.join(path, 'lowdim', cam), exist_ok=True)
        for t in range(T):
            img = np.full((8, 8, 3), (10 * c + t) % 255, np.uint8)
            imageio.imwrite(os.path.join(path, 'rgb', cam, f'{t:010d}.jpg'), img)
            c2w = np.eye(4, dtype=np.float32)
            c2w[0, 3] = c
            np.savez(os.path.join(path, 'lowdim', cam, f'{t:010d}.npz'), camera=np.array(cam),
                     timestep=np.int64(t), intrinsics=K, extrinsics=c2w)
    meta = {'info': {'name': name, 'storage': 'frames'}, 'cameras': CAMS, 'resolution': [8, 8],
            'num_frames': T, 'rgb': {'extension': 'jpg'},
            'extrinsics': {'transform': 'cam2world'}}
    if specific is not None:
        meta['specific'] = specific
    with open(os.path.join(path, 'metadata.json'), 'w') as f:
        json.dump(meta, f)


def draws(loader):
    '''(camera pair, window start) of every sample, in loader order.'''
    out = []
    for batch in loader:
        for item in batch:
            out.append(item['episode'].split('_', 1)[1])   # e.g. cam2_to_cam0_f17
    return out


def make_loader(dataset, workers):
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers,
                      collate_fn=lambda items: items, persistent_workers=False)


def test_scale_factor_per_scene():
    with tempfile.TemporaryDirectory() as root:
        write_scene(os.path.join(root, 'a_kubric'))                                 # Kubric-like
        write_scene(os.path.join(root, 'b_driving'), specific={'scale_factor': 1.0 / 16.0},
                    name='nuscenes')                                                # explicit
        write_scene(os.path.join(root, 'c_named'), name='waymo')                    # by name
        ds = KubricDVSDataset(root, num_frames=WINDOW, seed=0)
        got = {os.path.basename(d): ds[i]['scale_factor'] for i, d in enumerate(ds.scene_dirs)}
        assert got['a_kubric'] == KUBRIC_SCALE_FACTOR, got
        assert got['b_driving'] == 1.0 / 16.0, got
        assert got['c_named'] == 1.0 / 16.0, got


def test_workers_and_epochs_draw_differently():
    with tempfile.TemporaryDirectory() as root:
        n = 16
        for i in range(n):
            write_scene(os.path.join(root, f's{i:02d}'))
        ds = KubricDVSDataset(root, num_frames=WINDOW, seed=0)

        torch.manual_seed(0)
        loader = make_loader(ds, workers=2)
        epoch1, epoch2 = draws(loader), draws(loader)
        # batch_size=1, no shuffle: worker 0 serves items 0,2,4..., worker 1 items 1,3,5...
        same_across_workers = sum(epoch1[2 * k] == epoch1[2 * k + 1] for k in range(n // 2))
        assert same_across_workers < n // 4, (
            f'{same_across_workers}/{n // 2} item pairs drew identical camera pair + window: '
            f'workers share one RNG sequence')
        assert epoch1 != epoch2, 'epoch 2 replayed epoch 1 draws'

        torch.manual_seed(0)
        again = draws(make_loader(ds, workers=2))
        assert again == epoch1, 'draws are not reproducible under a fixed torch seed'


def test_single_process_unchanged():
    '''num_workers=0 keeps the original behaviour: the dataset's own seeded RNG.'''
    with tempfile.TemporaryDirectory() as root:
        for i in range(4):
            write_scene(os.path.join(root, f's{i}'))
        a = draws(make_loader(KubricDVSDataset(root, num_frames=WINDOW, seed=7), workers=0))
        b = draws(make_loader(KubricDVSDataset(root, num_frames=WINDOW, seed=7), workers=0))
        assert a == b


if __name__ == '__main__':
    failed = 0
    for fn in (test_scale_factor_per_scene, test_workers_and_epochs_draw_differently,
               test_single_process_unchanged):
        try:
            fn()
            print(f'PASS {fn.__name__}')
        except AssertionError as e:
            failed += 1
            print(f'FAIL {fn.__name__}: {e}')
    sys.exit(1 if failed else 0)
