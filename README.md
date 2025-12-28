# Speed-up-Point-Feature-Histograms
This is the final project of Umich ROB 422 by Shicheng Peng and Zehan Qin. The topic we chose is Point Cloud
processing. Our target aims to implement the Point Cloud Feature Histogram method with various configuration and come up with
an innovative way to reduce computational time. By the end, we will illustrate that our method works on
various testing data, including real-world point cloud examples.

## Environment Setup
In the project root, run:
```bash
./install.sh
```
If you have multiple Python versions, you can specify:
```bash
PYTHON_BIN=python3.10 ./install.sh
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

## Configuration and Run
Preprocessing procedure is in `preprocessing_and_filter.py`.
Core logic of PFH or FPFH is in `pfh_utils.py` and `icp_pfh.py`. All runtime settings are in `config_test.json`; edit the JSON instead of code:

**data_loading**
- `pc_source`: path to source point cloud (pcd/csv). Example: `data/object_template_0.pcd`
- `pc_target`: path to target point cloud, or `null` to auto-generate by transforming the source.
- `pcd_or_csv`: `"pcd"` | `"csv"` (default `"pcd"`)

**preprocessing**
- `voxel_size`: float, voxel downsample size.
- `need_translation`: bool, whether to add random translation when auto-generating target.
- `noise_level`: float, std of added Gaussian noise.

**target_point_selection**
- `use_filter_or_not`: bool, whether to enable target point filtering.
- `filter`: `"density_filter"` | `"iss"`.
- `density_number_of_neighbors`: int, KDTree neighbor count.
- `density_selection_ratio`: float, density threshold factor.

**pfh_parameters**
- `use_pfh`: bool; if false, fallback to nearest-neighbor ICP without PFH.
- `ratio_or_k`: float (0–1) ratio or int (k ≥ 3) neighbor count.
- `for_FPFH`: bool; compute FPFH if true, else SPFH.
- `which_cloud_for_bin`: `"target"` | `"source"` | `"both"`.
- `bins_per_feature`: int or list/ndarray, number of bins per feature.
- `bin_seperating_method`: `"equal_width"` | `"percentile"`.
- `distance_method`: `"l1"` | `"l2"` | `"chi-distance"` | `"JSD"` | `"cosine"`.
- `initialization_runs`: int, PFH initialization attempts.
- `ratio_init_sampling`: float, initialization sampling ratio.

**icp_parameters**
- `max_iteration`: int, ICP max iterations.
- `error_bound`: float, ICP stop threshold.

**visualization**
- `colors`: array of colors, e.g., `["#8ecae6", "#8bc34a", "#fcba03"]`.
- `markers`: array of markers, e.g., `["o", "^", "s"]`.

Run the full pipeline:
```bash
python -u icp_pfh.py
```

Run a simple demo (equivalent to the above):
```bash
python3 demo.py
```

