import json
from dataclasses import dataclass
from pathlib import Path

import torch

from autodec.visualizations.pointcloud import (
    points_to_numpy,
    read_point_cloud_ply,
    write_point_cloud_ply,
)
from autodec.visualizations.sq_mesh import export_sq_mesh


@dataclass(frozen=True)
class VisualizationRecord:
    epoch: int
    split: str
    sample_index: int
    sample_dir: Path
    input_path: Path
    sq_mesh_path: Path
    reconstruction_path: Path
    metadata_path: Path


class AutoDecEpochVisualizer:
    """Write per-epoch AutoDec 3D visualization artifacts without trainer coupling."""

    def __init__(
        self,
        root_dir="data/viz",
        run_name="autodec",
        mesh_resolution=24,
        exist_threshold=0.5,
        max_points=4096,
        input_color=(180, 180, 180),
        reconstruction_color=(42, 157, 143),
    ):
        self.root_dir = Path(root_dir)
        self.run_name = run_name
        self.mesh_resolution = mesh_resolution
        self.exist_threshold = exist_threshold
        self.max_points = max_points
        self.input_color = input_color
        self.reconstruction_color = reconstruction_color

    def _sample_dir(self, epoch, split, sample_index):
        return (
            self.root_dir
            / self.run_name
            / split
            / f"epoch_{epoch:04d}"
            / f"sample_{sample_index:04d}"
        )

    @staticmethod
    def _batch_size(points):
        if torch.is_tensor(points):
            shape = tuple(points.shape)
        else:
            shape = tuple(points.shape)
        return shape[0] if len(shape) == 3 else 1

    def _active_primitives(self, outdict, sample_index):
        if "exist" in outdict:
            exist = outdict["exist"]
        else:
            exist = torch.sigmoid(outdict["exist_logit"])
        if torch.is_tensor(exist):
            exist = exist.detach().cpu()
        return int((exist[sample_index, :, 0] > self.exist_threshold).sum().item())

    def _write_metadata(
        self,
        path,
        epoch,
        split,
        sample_index,
        input_points,
        reconstruction_points,
        active_primitives,
    ):
        metadata = {
            "epoch": epoch,
            "split": split,
            "sample_index": sample_index,
            "input_points": int(input_points),
            "reconstruction_points": int(reconstruction_points),
            "active_primitives": int(active_primitives),
        }
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return path

    def write_epoch(self, batch, outdict, epoch, split="val", num_samples=1):
        """Write input, SQ mesh, reconstruction, and metadata for epoch samples."""

        points = batch["points"]
        decoded_points = outdict["decoded_points"]
        batch_size = min(self._batch_size(points), self._batch_size(decoded_points))
        num_samples = min(num_samples, batch_size)
        records = []

        for sample_index in range(num_samples):
            sample_dir = self._sample_dir(epoch, split, sample_index)
            sample_dir.mkdir(parents=True, exist_ok=True)

            input_path = sample_dir / "input_gt.ply"
            sq_mesh_path = sample_dir / "sq_mesh.obj"
            reconstruction_path = sample_dir / "reconstruction.ply"
            metadata_path = sample_dir / "metadata.json"

            input_points = points_to_numpy(points, sample_index, self.max_points)
            reconstruction_points = points_to_numpy(
                decoded_points,
                sample_index,
                self.max_points,
            )
            write_point_cloud_ply(input_path, input_points, color=self.input_color)
            export_sq_mesh(
                sq_mesh_path,
                outdict,
                sample_index=sample_index,
                resolution=self.mesh_resolution,
                exist_threshold=self.exist_threshold,
            )
            write_point_cloud_ply(
                reconstruction_path,
                reconstruction_points,
                color=self.reconstruction_color,
            )
            self._write_metadata(
                metadata_path,
                epoch,
                split,
                sample_index,
                input_points.shape[0],
                reconstruction_points.shape[0],
                self._active_primitives(outdict, sample_index),
            )
            records.append(
                VisualizationRecord(
                    epoch=epoch,
                    split=split,
                    sample_index=sample_index,
                    sample_dir=sample_dir,
                    input_path=input_path,
                    sq_mesh_path=sq_mesh_path,
                    reconstruction_path=reconstruction_path,
                    metadata_path=metadata_path,
                )
            )
        return records


def _default_object3d_factory(path):
    import wandb

    if Path(path).suffix.lower() == ".ply":
        return wandb.Object3D(read_point_cloud_ply(path))
    return wandb.Object3D(str(path))


def build_wandb_log(records, object3d_factory=None, prefix="visual"):
    """Build a WandB payload from local visualization records without logging it."""

    object3d_factory = object3d_factory or _default_object3d_factory
    return {
        f"{prefix}/gt": [object3d_factory(record.input_path) for record in records],
        f"{prefix}/sq_mesh": [object3d_factory(record.sq_mesh_path) for record in records],
        f"{prefix}/reconstruction": [
            object3d_factory(record.reconstruction_path) for record in records
        ],
    }


def log_wandb_visualizations(wandb_run, records, step=None, prefix="visual"):
    """Log visualization records to WandB when a run is supplied by training code."""

    if wandb_run is None or not records:
        return {}
    payload = build_wandb_log(records, prefix=prefix)
    wandb_run.log(payload, step=step)
    return payload
