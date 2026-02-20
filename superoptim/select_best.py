import pandas as pd
import argparse
import sys
import os
import numpy as np
from superdec.utils.predictions_handler_extended import PredictionHandler 

def main():
  parser = argparse.ArgumentParser(description='Select best rows by a metric and plot means of other metrics.')
  parser.add_argument('csvs', nargs='+', help='CSV files to read and concatenate')
  parser.add_argument('--metric', default='iou', help='Metric to use for selecting best per group')
  parser.add_argument('--func', choices=['min', 'max'], default='max', help='Use min or max to pick best')
  parser.add_argument('--group-by', default='index', help='Column name to group by when selecting best')
  parser.add_argument('--save-npz', default=None, help='Path to save merged npz file with best primitives')

  args = parser.parse_args()

  dfs = []
  for f in args.csvs:
    try:
      d = pd.read_csv(f)
      d['_source_csv'] = f
      dfs.append(d)
    except Exception as e:
      print('Error reading CSV files:', e, file=sys.stderr)
      sys.exit(1)
  
  if not dfs:
      print("No CSV files loaded.")
      sys.exit(1)
      
  df = pd.concat(dfs, ignore_index=True)

  if args.group_by not in df.columns:
    print(f"Group-by column '{args.group_by}' not found in data", file=sys.stderr)
    sys.exit(1)

  if args.metric not in df.columns:
    print(f"Selection metric '{args.metric}' not found in data", file=sys.stderr)
    sys.exit(1)

  grp = df.groupby(args.group_by)
  if args.func == 'min':
    idx = grp[args.metric].idxmin()
  else:
    idx = grp[args.metric].idxmax()

  selected = df.loc[idx].reset_index(drop=True)

  numeric_cols = selected.select_dtypes(include='number').columns.tolist()
  # remove grouping and selection columns from the plotted metrics
  numeric_cols = [c for c in numeric_cols if c != args.group_by]

  if not numeric_cols:
    print('No numeric metrics found to compute means on.', file=sys.stderr)
    sys.exit(1)

  means = selected[numeric_cols].mean()

  means = means.sort_values()
  print(f"Means of metrics (selected by {args.func} {args.metric}):")
  print(means.to_string())

  if args.save_npz:
    print(f"Combining NPZ files into {args.save_npz}...")
    
    # 1. Load handlers for all sources
    handlers = {}
    valid_csvs = set(selected['_source_csv'].unique())
    
    for csv_file in valid_csvs:
        # Infer NPZ path
        npz_file = csv_file.replace('_metrics.csv', '.npz')
        if not os.path.exists(npz_file):
            print(f"Warning: NPZ file {npz_file} not found for {csv_file}. Skipping related objects.", file=sys.stderr)
            continue
            
        try:
            handlers[csv_file] = PredictionHandler.from_npz(npz_file)
        except Exception as e:
            print(f"Error loading NPZ {npz_file}: {e}", file=sys.stderr)

    if not handlers:
        print("No valid NPZ files loaded. Cannot save merged NPZ.", file=sys.stderr)
        return

    # Attributes to collect
    # Based on PredictionHandler structure
    keys = ['names', 'pc', 'assign_matrix', 
            'scale', 'rotation', 'translation', 'exponents', 'exist', 
            'tapering', 'bending']
    
    merged_lists = {k: [] for k in keys}
    
    # Sort selected dataframe to ensure deterministic order (e.g. by index)
    try:
        selected['index_int'] = selected['index'].astype(int)
        selected_sorted = selected.sort_values(by='index_int')
    except:
        selected_sorted = selected.sort_values(by=args.group_by)

    count = 0
    source_counts = {}
    for _, row in selected_sorted.iterrows():
        source = row['_source_csv']
        if source not in handlers:
            continue
        source_counts[source] = source_counts.get(source, 0) + 1

        # Find index in handler
        h = handlers[source]
        name = row['name']
        matches = np.where(h.names == name)[0]
        idx = matches[0] if len(matches) > 0 else -1
        
        if idx == -1:
            print(f"Warning: Object {name} not found in source NPZ associated with {source}")
            continue
            
        # Append data
        for k in keys:
            val = getattr(h, k, None)
            merged_lists[k].append(val[idx])
        
        count += 1
        
    print(f"Merged {count} objects.")
    print("Source distribution:")
    for src, cnt in source_counts.items():
        print(f"  {src}: {cnt} objects")
    
    if count == 0:
        print("No objects merged. Existing.")
        return

    final_dict = {}
    for k in keys:
        if not merged_lists[k]:
            final_dict[k] = np.array([])
            continue
        item_list = merged_lists[k]
        final_dict[k] = np.stack(item_list)

    # Save
    new_handler = PredictionHandler(final_dict)
    new_handler.save_npz(args.save_npz)
    print(f"Saved merged predictions to {args.save_npz}")


if __name__ == '__main__':
  main()