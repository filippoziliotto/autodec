import os
import re
from pathlib import Path
import wandb

def cleanup_wandb_3d_files(run_id, keep_ratio=0.5):
  """
  Clean up 3D object files from a wandb run by deleting every other epoch.
  
  Args:
    run_id: The wandb run ID
    keep_ratio: Ratio of files to keep (default: keep every other = 0.5)
  """
  api = wandb.Api()
  run = api.run(f"{run.entity}/{run.project}/{run_id}")
  
  # Path to 3D objects
  object_path = Path(run.dir) / "media" / "object3D" / "visual"
  
  if not object_path.exists():
    print(f"Path {object_path} does not exist")
    return
  
  # Find all .glb files and extract epochs
  files_by_epoch = {}
  for file in object_path.glob("pred_*_*.glb"):
    match = re.match(r"pred_(\d+)_", file.name)
    if match:
      epoch = int(match.group(1))
      files_by_epoch[epoch] = file
  
  if not files_by_epoch:
    print("No files found")
    return
  
  # Sort by epoch
  sorted_epochs = sorted(files_by_epoch.keys())
  
  # Delete every other file
  for i, epoch in enumerate(sorted_epochs):
    if i % 2 == 1:  # Delete odd indices (keep even)
      file_path = files_by_epoch[epoch]
      file_path.unlink()
      print(f"Deleted: {file_path.name} (epoch {epoch})")
  
  print(f"Cleanup complete. Kept {len(sorted_epochs) // 2 + 1} files")

if __name__ == "__main__":
  run_id = "bpf5jqzw"
  cleanup_wandb_3d_files(run_id)