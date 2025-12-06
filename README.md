# Speed-up-Point-Feature-Histograms
Final project for ROB 422 in Umich

## Environment setting
```bash
conda env create -f environment.yml
conda activate pfh
```

If it is not working, try the following:
```bash
conda init cmd.exe
cmd /k "conda activate pfh"
```

## Github pull and push
```bash
git clone https://github.com/Pscjiayou/Speed-up-Point-Feature-Histograms.git
```

Getting the latest codes
```bash
git pull origin main
```

Creating branch
```bash
git checkout -b your_name-feature
```

Checking which branch it is
```bash
git branch
```

Pushing codes to your branch
```bash
git add .
git commit -m "comments on the updates"
git push origin your_name-feature
```

## Using the codes
The core of Point Feature Histogram is in `pfh_utils.py` and `icp_pfh.py`.

All runtime settings are read from `config_icp_pfh.json` when you run `python -u icp_pfh.py`. Modify the JSON instead of editing code:

- `"pc_source"` / `"pc_target"`: source/target point cloud CSV file names.
- `"ratio_or_k"`: neighbor ratio (0~1) or integer k (k≥3).
- `"use_target_indices"`: if true, the demo uses target indices given by the points selection part; set false to use all points or your own selector.
- `"for_FPFH"`: true to compute FPFH, false for SPFH only.
- `"which_cloud_for_bin"`: `"target"`, `"source"`, or `"both"` to decide which cloud defines histogram bins.
- `"bins_per_feature"`: int or list/ndarray of ints; bin counts per feature.
- `"bin_seperating_method"`: `"equal_width"` or `"percentile"`; binning strategy.
- `"distance_method"`: `"l1"`, `"l2"`, `"chi-distance"`, `"JSD"`, or `"cosine"`; correspondence metric.
- `"max_iteration"`, `"error_bound"`: ICP stopping criteria.
- `"colors"`, `"markers"`: matplotlib colors/markers for visualization.

Run the pipeline:
```bash
python -u icp_pfh.py
```
