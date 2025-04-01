# Image Segmentation Inference Finetuned on Human

## Overview
This code performs image segmentation using a Facebook pretrianed sam2 model fintunned on human body. It takes as input a directory containing images or a text file listing image paths, runs inference using a pre-trained model, and outputs the segmented images along with corresponding JSON files containing the segmentation details.

## Features
- Loads a model checkpoint and performs inference on input images
- Supports batch processing for efficient inference
- Saves segmentation results as images with overlays
- Stores segmentation masks in `.npy` format
- Outputs segmentation details in JSON format, including labels and segmented regions
- Uses multiprocessing to optimize data loading and result saving

## Requirements
- Python 3.8+
- PyTorch
- Torchvision
- OpenCV
- NumPy
- TQDM
- Matplotlib

## Installation
Install the required dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
Run the script using:
```bash
python  .\\demo\\vis_seg.py  <checkpoint_path> --input <image_directory> --output_root <output_directory>
```

### Arguments
- `checkpoint`: Path to the model checkpoint file. https://huggingface.co/facebook/sapiens-seg-1b-torchscript/tree/main
- `--input`: Directory containing images or a `.txt` file listing image paths.
- `--output_root`: Directory to store output images and JSON files (default: `input_directory/output`).
- `--device`: Device for inference (default: `cuda:0`).
- `--batch_size`: Batch size for inference (default: 4).
- `--shape`: Input image size (default: `[1024, 768]`).
- `--fp16`: Use mixed precision inference (default: False).
- `--opacity`: Opacity of segmentation overlay (default: 0.5).
- `--title`: Identifier for output images (default: "result").
- `--num_workers`: Number of worker processes for data loading (default: 4).
- `--save_format`: Format to save segmentation results (options: `png`, `jpg`, `bmp`, default: `png`).

## Output
For each processed image, the script generates:
1. **Segmented Image**: The original image overlaid with segmentation results.
2. **Segmentation Mask (`.npy`)**: Binary mask representing segmented regions.
3. **Segmentation Details (`.json`)**:
   - `labels`: List of detected classes.
   - `regions`: Pixel coordinates of segmented areas.

## Example
```bash
python .\\demo\\vis_seg.py .\\checkpoints\sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2 --input .\\input_images\ \--output_root .\\output_images\\
```
This will process images in `./images`, save outputs in `./output`, use a batch size of 8, employ 8 worker processes for data loading, and save results in JPG format.

## License
This code is licensed under the Meta Platforms, Inc. License found in the `LICENSE` file.

## Acknowledgements
This project utilizes deep learning models for segmentation and leverages PyTorch for inference.

