# Single-View Prediction

Metric-accurate 3D reconstruction from a single RGB image.

This is a course project that takes a regular photo, estimates its depth using a pretrained monocular depth model (Depth Anything V2), and then corrects the predicted depth to real-world metric scale using a small number of known depth points. The final metric depth map is then projected into a 3D point cloud that you can view in MeshLab or Open3D.

## What this project does

The core problem is that models like Depth Anything V2 are really good at predicting the relative structure of a scene (what is closer, what is farther), but the raw output is not in meters. It is just an arbitrary scale. So if you tried to build a 3D model from it directly, the scale would be wrong.

We fix this by sampling a few sparse points (say 100) where we know the true depth (from the ground truth in the dataset), and then we solve a simple linear equation to find the right scale and shift to convert the entire predicted depth map into metric depth. We use RANSAC to make this robust to outliers.

Once we have the corrected depth, we use the camera intrinsics to back-project every pixel into 3D space and export it as a `.ply` point cloud file.

## Pipeline overview

![Pipeline diagram](pipeline.png)

The pipeline has five stages:

1. **Data loading** - Load an RGB image, its ground truth depth map, and camera intrinsics from the NYU Depth V2 dataset (HDF5 format from Kaggle).
2. **Depth prediction** - Run Depth Anything V2 (Large) on the RGB image to get a dense relative depth map.
3. **Anchor sampling and alignment** - Randomly sample N points from the ground truth depth, then fit `d_metric = s * d_pred + t` using RANSAC to find the global scale (s) and shift (t). Apply this to the entire predicted map.
4. **3D projection** - Use the pinhole camera model with the intrinsic matrix K to project every pixel into (X, Y, Z) coordinates. Export as a .ply file.
5. **Evaluation** - Compare the aligned depth against the full ground truth using AbsRel, RMSE, and delta < 1.25 metrics.

## Project structure

```
Single-View_Prediction/
    src/
        config.py           - all default settings (paths, model name, camera params, RANSAC config)
        dataloader.py       - loads the NYU Depth V2 HDF5 dataset
        depth_estimator.py  - wrapper around Depth Anything V2 for inference
        aligner.py          - sparse anchor sampling + RANSAC scale-shift solver
        projector.py        - pinhole back-projection and .ply export using Open3D
        visualizer.py       - plotting functions (depth comparison, error maps, sparsity curves)
        metrics.py          - depth evaluation metrics (AbsRel, RMSE, delta thresholds)

    scripts/
        run_pipeline.py     - runs the full pipeline on a single image
        evaluate.py         - runs evaluation over many images and prints a results table
        sparsity_analysis.py - sweeps the number of anchors and plots RMSE vs N

    outputs/                - where all generated files go (gitignored)
```

## What each file does

**src/config.py** - Contains all the default hyperparameters in a single dictionary. This includes the model name, dataset path, NYU camera intrinsics (fx, fy, cx, cy), RANSAC parameters (iterations, threshold, min samples), and output paths. If you need to change anything, start here.

**src/dataloader.py** - Defines `NYUDepthV2Dataset`, a PyTorch Dataset that reads from the Kaggle HDF5 file. Each sample gives you the RGB image as a (3, H, W) tensor, a ground truth depth map in meters, and the 3x3 camera intrinsic matrix. There is also a `get_sample()` helper if you just want NumPy arrays.

**src/depth_estimator.py** - The `DepthEstimator` class loads Depth Anything V2 Large from HuggingFace and runs inference. You pass in an image (PIL, NumPy, or tensor) and get back a (H, W) depth map. The output is relative depth, not metric.

**src/aligner.py** - Two classes here. `SparseAnchorSampler` picks N random valid pixels from the ground truth depth (simulating what you would get from a LiDAR or SfM in a real scenario). `RANSACAligner` takes those anchor points and fits a linear model to find the scale and shift. The `align()` method applies the correction to the full predicted depth map.

**src/projector.py** - `PointCloudProjector` takes a metric depth map and uses the standard pinhole equations (Z = depth, X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy) to get 3D coordinates for every pixel. It can export the result as a colored .ply file using Open3D.

**src/visualizer.py** - Plotting helpers. `plot_depth_comparison` shows a side-by-side of the input image, raw prediction, aligned prediction, and ground truth. `plot_error_map` shows where the errors are. `plot_sparsity_curve` generates the RMSE vs N plot for the sparsity analysis.

**src/metrics.py** - Implements the three standard depth evaluation metrics: AbsRel (mean relative error), RMSE (root mean squared error in meters), and delta < 1.25 (percentage of pixels within a threshold ratio). The `evaluate()` function runs all of them at once.

## Setup

### Requirements

- Python 3.10 or later
- A CUDA-capable GPU (the model is large and CPU inference will be very slow)
- Around 4 GB disk space for the dataset
- Around 1.3 GB for the model weights (downloaded automatically on first run)

### GPU requirements

The Depth Anything V2 Large model needs at least 4 GB of VRAM. It runs fine on a single GPU like an RTX 3050, RTX 3060, T4, or A100. If you are running on something with less VRAM, you can try the base or small variants by changing the model name in `src/config.py`.

If you do not have a local GPU, you can run this on Google Colab (T4 runtime), Lightning AI, or any cloud GPU instance.

### Installation

```bash
git clone https://github.com/ankur777jinn/Single-View_Prediction.git
cd Single-View_Prediction

# create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate    # on Linux/Mac
venv\Scripts\activate       # on Windows

pip install -r requirements.txt
```

### Dataset

We use the NYU Depth V2 dataset from Kaggle. Download it from here:

https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2

After downloading, place the `nyu_depth_v2.h5` file in a `data/` folder at the project root:

```
Single-View_Prediction/
    data/
        nyu_depth_v2.h5
```

Or you can pass a custom path to any script using the `--data` flag.

## How to run

All scripts should be run from the project root directory.

### Run the full pipeline on a single image

```bash
python scripts/run_pipeline.py --index 0 --anchors 100
```

This will:
- Load image #0 from the dataset
- Run depth estimation
- Sample 100 anchors and align the depth
- Save a 3D point cloud to `outputs/pointcloud_idx0.ply`
- Save comparison plots to `outputs/`
- Print evaluation metrics

You can change `--index` to try different images and `--anchors` to change how many ground truth points are used for alignment. Add `--no-viz` to skip the plots. Use `--device cpu` if you do not have a GPU (will be slow).

### Run batch evaluation

```bash
python scripts/evaluate.py --num-images 50 --anchors 100
```

This evaluates the pipeline on 50 images and prints a table with the mean and standard deviation of all metrics. Results are also saved to `outputs/eval_results.txt`.

### Run the sparsity sensitivity analysis

```bash
python scripts/sparsity_analysis.py --num-images 20
```

This tests how the reconstruction quality changes when you use fewer anchor points. It sweeps N = 5, 10, 50, 100, 500 and plots RMSE vs N. The plot is saved to `outputs/sparsity_sensitivity.png`.

## Metrics

We evaluate using the standard metrics from the depth estimation literature:

- **AbsRel**: Average of |predicted - ground_truth| / ground_truth. Lower is better.
- **RMSE**: Root mean squared error in meters. Lower is better.
- **delta < 1.25**: Percentage of pixels where the ratio between predicted and ground truth depth is less than 1.25. Higher is better.

## References

- Eigen, Puhrsch, Fergus. "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network." 2014. https://arxiv.org/abs/1406.2283
- Yang, Kang, Huang, Zhao, Xu, Feng, Zhao. "Depth Anything V2." 2024. https://arxiv.org/abs/2406.09414
- Eigen, Fergus. "Predicting Depth, Surface Normals and Semantic Labels with a Common Multi-Scale Convolutional Architecture." 2015. https://arxiv.org/abs/1411.4734
