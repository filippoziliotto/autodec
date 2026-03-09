import os
import time
import torch
import numpy as np
from omegaconf import OmegaConf
from superdec.superdec import SuperDec
from superdec.data.dataloader import normalize_points
from superdec.data.transform import rotate_around_axis
from plyfile import PlyData


def main():
    checkpoints_folder = "checkpoints/shapenet_iou_371"
    checkpoint_file = "epoch_1000.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path_to_point_cloud = "examples/chair.ply"
    z_up = False
    normalize = True
    lm_optimization = False
    n_runs = 100        # number of inference repetitions
    n_warmup = 10       # warmup runs (excluded from stats)

    ckp_path = os.path.join(checkpoints_folder, checkpoint_file)
    config_path = os.path.join(checkpoints_folder, 'config.yaml')
    if not os.path.isfile(ckp_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckp_path}")

    checkpoint = torch.load(ckp_path, map_location=device, weights_only=False)
    with open(config_path) as f:
        configs = OmegaConf.load(f)

    model = SuperDec(configs.superdec).to(device)
    model.lm_optimization = lm_optimization
    print("Loading checkpoint from:", ckp_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    plydata = PlyData.read(path_to_point_cloud)
    v = plydata['vertex']
    points_tmp = np.stack([v['x'], v['y'], v['z']], axis=-1).astype(np.float32)
    n_points = points_tmp.shape[0]
    if n_points != 4096:
        replace = n_points < 4096
        idxs = np.random.choice(n_points, 4096, replace=replace)
        points_tmp = points_tmp[idxs]

    if normalize:
        points_tmp, _, _ = normalize_points(points_tmp)
    if z_up:
        points_tmp = rotate_around_axis(points_tmp, axis=(1, 0, 0), angle=-np.pi / 2, center_point=np.zeros(3))

    points = torch.from_numpy(points_tmp).unsqueeze(0).to(device).float()

    def sync():
        if device == 'cuda':
            torch.cuda.synchronize()

    print(f"Warming up ({n_warmup} runs)...")
    with torch.no_grad():
        for _ in range(n_warmup):
            model(points)

    print(f"Timing {n_runs} inference runs...")
    times = []
    with torch.no_grad():
        for i in range(n_runs):
            sync()
            t0 = time.perf_counter()
            model(points)
            sync()
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000
            times.append(elapsed_ms)
            print(f"  [{i+1:>{len(str(n_runs))}}/{n_runs}] {elapsed_ms:.2f} ms")

    times = np.array(times)
    print(f"\n--- Results over {n_runs} runs ---")
    print(f"  Mean  : {times.mean():.2f} ms")
    print(f"  Std   : {times.std():.2f} ms")
    print(f"  Min   : {times.min():.2f} ms")
    print(f"  Max   : {times.max():.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")


if __name__ == "__main__":
    main()
