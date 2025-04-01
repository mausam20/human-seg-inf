# Sapiens-Lite: Body Part Segmentation

## Model Zoo
We use 28 classes for body-part segmentation along with the background class.
You can checkout more details on the classes [here](../../seg/mmseg/datasets/goliath.py).

```
0: Background
1: Apparel
2: Face_Neck
3: Hair
4: Left_Foot
5: Left_Hand
6: Left_Lower_Arm
7: Left_Lower_Leg
8: Left_Shoe
9: Left_Sock
10: Left_Upper_Arm
11: Left_Upper_Leg
12: Lower_Clothing
13: Right_Foot
14: Right_Hand
15: Right_Lower_Arm
16: Right_Lower_Leg
17: Right_Shoe
18: Right_Sock
19: Right_Upper_Arm
20: Right_Upper_Leg
21: Torso
22: Upper_Clothing
23: Lower_Lip
24: Upper_Lip
25: Lower_Teeth
26: Upper_Teeth
27: Tongue
```

The body-part segmentation model checkpoints are available at,

| Model         | Checkpoint Path
|---------------|--------------------------------------------------------------------------------------------------
| Sapiens-0.3B  | `$SAPIENS_LITE_CHECKPOINT_ROOT/seg/checkpoints/sapiens_0.3b/sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_$MODE.pt2`
| Sapiens-0.6B  | `$SAPIENS_LITE_CHECKPOINT_ROOT/seg/checkpoints/sapiens_0.6b/sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_$MODE.pt2`
| Sapiens-1B  | `$SAPIENS_LITE_CHECKPOINT_ROOT/seg/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_$MODE.pt2`

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


