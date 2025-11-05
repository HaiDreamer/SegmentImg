# Detailed Guide to Training a Model on Kaggle GPU

## Overview

This guide walks you through training a segmentation model on Kaggle with GPUs (P100, T4, or T4 x2) step by step.

---

## Step 1: Prepare the Dataset on Kaggle

### Option 1: Create a Kaggle Dataset with a ZIP File (Recommended – Fastest)

**On Kaggle:**

1. **Prepare a ZIP file on your local machine:**
   - Zip the entire `data/` folder (containing the folders: cheetah, lion, wolf, tiger, hyena, fox)
   - The zip should have the structure: `data.zip` → `data/cheetah/`, `data/lion/`, ...
   - Or zip everything such that extracting it results in a top-level `data/` folder

2. **Go to:** https://www.kaggle.com/datasets
3. **Create a new dataset:** Click **"New Dataset"**
4. **Upload the ZIP file:**
   - Drag and drop `data.zip` (or your zip filename)
   - Upload `labelmap.txt` to the same dataset
   - **Note:** You can upload multiple files at once
5. **Name the dataset:** e.g., `animal-segmentation-dataset`
6. **Public or Private:** Choose as needed
7. **Click "Create"**

**Pros:**
- Faster upload (one file instead of many folders)
- Preserves directory structure
- Easy to manage and share

### Option 2: Upload Each Folder (If you don’t zip)

**On Kaggle:**

1. **Go to:** https://www.kaggle.com/datasets
2. **Create a new dataset:** Click **"New Dataset"**
3. **Upload data:**
   - Drag and drop or select the data folders:
     - `data/cheetah/`
     - `data/lion/`
     - `data/wolf/`
     - `data/tiger/`
     - `data/hyena/`
     - `data/fox/`
   - Upload `labelmap.txt` to the dataset root
4. **Name the dataset:** e.g., `animal-segmentation-dataset`
5. **Public or Private:** Choose as needed
6. **Click "Create"**

### Option 3: Upload Directly into the Notebook (For small files)

- Use the file browser in the Kaggle notebook to upload directly
- Files will be saved to `/kaggle/working/`

---

## Step 2: Create a Notebook on Kaggle

1. **Go to:** https://www.kaggle.com/code
2. **Create a new notebook:** Click **"New Notebook"**
3. **Choose GPU:**
   - Settings → Accelerator → **GPU** (P100, T4, or 2xT4)
   - **Note:** Kaggle provides free GPUs but with time limits
4. **Add Dataset:**
   - Click **"Add Data"** on the right
   - Find and select the dataset you created in Step 1
   - The dataset will be mounted at `/kaggle/input/YOUR_DATASET_NAME/`

---

## GPU Comparison: 2xT4 vs P100 – Which to Choose?

### Detailed Comparison Table

| Criteria | **P100 (Single)** | **2xT4 (Dual)** | **Recommendation** |
|----------|-------------------|-----------------|--------------------|
| **Total VRAM** | 16GB | 32GB (16GB x2) | 2xT4 for large models |
| **VRAM per GPU** | 16GB | 16GB | = |
| **Compute Power** | Higher than T4 | Moderate | P100 for speed |
| **Max Batch Size** | 12–16 (512x512) | 20–32 (512x512) | 2xT4 for large batches |
| **Max Patch Size** | 512–640 | 768–1024 | 2xT4 for high resolution |
| **Multi-GPU Setup** | Not needed | Required | P100 is simpler |
| **Training Speed** | Faster | Slower (overhead) | P100 is faster |
| **Code Complexity** | Simple | Requires multi-GPU strategy | P100 is easier |
| **Large Models (EfficientNetB4+)** | Possible OOM | Sufficient VRAM | 2xT4 for large models |
| **Kaggle Availability** | Often available | Less available | P100 easier to get |

### GPU Choice Recommendations

#### **Choose P100 if:**
- Small-to-medium models (EfficientNetB0–B3)
- You want faster training
- You prefer simpler code (single GPU)
- Patch size ≤ 640x640
- Batch size ≤ 16 is sufficient
- **This is the best choice for EfficientNetB3!**

#### **Choose 2xT4 if:**
- Large models (EfficientNetB4–B7, ResNet101+)
- Need large patch size (≥ 768x768)
- Need large batch size (> 20)
- Comfortable configuring multi-GPU
- Very large dataset with high throughput needs

### Conclusion for EfficientNetB3

**Recommended: P100**

**Why:**
1. EfficientNetB3 is moderate; P100 has enough VRAM
2. Faster training (no multi-GPU overhead)
3. Simpler code (single GPU)
4. 512x512 patch size is optimal; P100 handles it well
5. Batch size 8–12 is sufficient for stable training

**Choose 2xT4 only if:**
- You want to train EfficientNetB4 or higher
- You need patch size ≥ 768x768
- You need batch size ≥ 20

---

### Multi-GPU Training with 2xT4 (If Needed)

If you choose 2xT4, set up MirroredStrategy to use both GPUs:

```python
# Setup Multi-GPU Strategy (only needed if you have 2xT4)
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 1:
    # Create MirroredStrategy to use all GPUs
    strategy = tf.distribute.MirroredStrategy()
    print(f" Using {strategy.num_replicas_in_sync} GPU(s)")
    
    # Build model and train inside strategy scope
    with strategy.scope():
        # Build your model here
        model = build_unet_with_backbone(
            num_classes=num_classes,
            backbone="efficientnet",
            backbone_name="EfficientNetB3"
        )
        # Compile model
        model.compile(...)
        
    # Training will automatically distribute across GPUs
    model.fit(train_ds, epochs=EPOCHS, ...)
else:
    # Single GPU – no strategy needed
    model = build_unet_with_backbone(...)
    model.compile(...)
    model.fit(train_ds, epochs=EPOCHS, ...)
```

**Multi-GPU Notes:**
- Batch size is divided evenly across GPUs (batch_size=16 → 8 per GPU)
- Effective batch size = batch_size × number_of_GPUs
- Communication overhead can slow training by 10–20%
- Use only when single GPU VRAM is insufficient

---

## Step 3: Set Up the Environment

### Cell 1: Check GPU

```python
# Check if a GPU is available
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU Available: {len(gpus)} GPU(s)")
print(f"GPU Details: {gpus}")

# Inspect GPU details
if gpus:
    for i, gpu in enumerate(gpus):
        print(f"\nGPU {i}: {gpu}")
        print(f"  Name: {gpu.name}")
        # Enable memory growth to avoid allocating all VRAM upfront
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"   Memory growth enabled")
        except RuntimeError as e:
            print(f"   Cannot set memory growth: {e}")
    
    # If multiple GPUs, you can use a multi-GPU strategy
    if len(gpus) > 1:
        print(f"\n Multi-GPU detected: {len(gpus)} GPUs")
        print(" Tip: To use multi-GPU, set up MirroredStrategy (see Multi-GPU Training section)")
    else:
        print(f"\n Single GPU setup – simple and effective!")
else:
    print(" No GPU! Please select GPU in Settings → Accelerator")
```

### Cell 2: Install Dependencies

```python
# In Kaggle notebooks, run this cell BEFORE imports:
!pip install --upgrade scikit-image==0.23.0 --quiet

# Optional: CRF post-processing (if needed)
# !pip install -q git+https://github.com/lucasb-eyer/pydensecrf.git

print(" Dependencies installed!")
```

### Cell 3: Check Modules

```python
import tensorflow as tf
import numpy as np
import scipy
import keras

print(f"TensorFlow: {tf.__version__}")
print(f"NumPy: {np.__version__}")
print(f"SciPy: {scipy.__version__}")
print(f"Keras: {keras.__version__}")

# Test import of required modules
from skimage.segmentation import find_boundaries
import cv2
from PIL import Image

print(" All modules imported successfully!")
```

### Cell 4: Upload Code

**Option A: Upload file directly**
```python
# Use the file browser on the left to upload model_train_v3_kaggle.py
# The file will be saved to /kaggle/working/
```

**Option B: Paste code directly**
```python
# Create a new file
%%writefile /kaggle/working/model_train_v3_kaggle.py
# Paste the entire content of model_train_v3_kaggle.py here
```

**Option C: If the code is already in the dataset**
```python
# Copy from the input dataset
!cp /kaggle/input/YOUR_DATASET_NAME/model_train_v3_kaggle.py /kaggle/working/
!cp /kaggle/input/YOUR_DATASET_NAME/labelmap.txt /kaggle/working/
```

---

## Step 5: Configure Paths

### Cell 5: Set up paths

```python
import os
from pathlib import Path

# Find dataset in /kaggle/input
# Replace YOUR_DATASET_NAME with your dataset name
DATASET_NAME = "YOUR_DATASET_NAME"  # e.g., "animal-segmentation-dataset"
DATASET_PATH = f"/kaggle/input/{DATASET_NAME}"

# Check if dataset exists
if not Path(DATASET_PATH).exists():
    # Try other names
    possible_names = ["segmentimg", "animal-segmentation-dataset", "segmentation-data"]
    for name in possible_names:
        test_path = f"/kaggle/input/{name}"
        if Path(test_path).exists():
            DATASET_PATH = test_path
            DATASET_NAME = name
            break

# Determine data path (prefer /kaggle/working/ if already extracted)
WORKING_DATA = Path("/kaggle/working/data")
INPUT_DATA = Path(f"{DATASET_PATH}/data")

# Check where data is (extracted or not)
if WORKING_DATA.exists() and any(WORKING_DATA.iterdir()):
    # Data extracted to /kaggle/working/data
    DATA_BASE = "/kaggle/working"
    print(" Using data extracted at /kaggle/working/data")
elif INPUT_DATA.exists():
    # Data in the dataset (not zipped or extracted elsewhere)
    DATA_BASE = DATASET_PATH
    print(f" Using data from dataset: {DATASET_PATH}")
else:
    # Fallback: look elsewhere
    DATA_BASE = DATASET_PATH
    print(" Could not find the data folder, please check your dataset structure")

# Dataset folders
DATA_ROOTS = [
    f"{DATA_BASE}/{DATA_SUBPATH}/cheetah",
    f"{DATA_BASE}/{DATA_SUBPATH}/lion",
    f"{DATA_BASE}/{DATA_SUBPATH}/wolf",
    f"{DATA_BASE}/{DATA_SUBPATH}/tiger",
    f"{DATA_BASE}/{DATA_SUBPATH}/hyena",
    f"{DATA_BASE}/{DATA_SUBPATH}/fox",
]

# Labelmap from dataset or working directory
LABELMAP_PATH = f"{DATASET_PATH}/labelmap.txt"
if not Path(LABELMAP_PATH).exists():
    LABELMAP_PATH = "/kaggle/working/labelmap.txt"

# Directory to save models (always save to /kaggle/working/)
SAVE_DIR = "/kaggle/working/models"

# Verify paths
print("\n" + "="*60)
print("CHECKING DATA PATHS")
print("="*60)
print(f"Dataset: {DATASET_NAME}")
print(f"Data base: {DATA_BASE}")

all_exist = True
for root in DATA_ROOTS:
    root_path = Path(root)
    if root_path.exists():
        jpeg_path = root_path / "JPEGImages"
        n_images = len(list(jpeg_path.glob("*"))) if jpeg_path.exists() else 0
        print(f" {root_path.name}: {n_images} images")
    else:
        print(f" {root_path.name} - DOES NOT EXIST!")
        all_exist = False

if Path(LABELMAP_PATH).exists():
    print(f" Labelmap: {LABELMAP_PATH}")
else:
    print(f" Labelmap not found: {LABELMAP_PATH}")
    print(" Please upload labelmap.txt to the dataset or /kaggle/working/")
    all_exist = False

if not all_exist:
    print("\n Some paths do not exist!")
    print(" Please check:")
    print("   1. Did you extract the ZIP file? (Cell 3)")
    print("   2. Is the dataset name correct?")
    print("   3. Is the dataset folder structure correct?")
else:
    print("\n All paths are valid!")
print("="*60)
```

---

## Step 6: Import Code and Train

### Cell 6: Import and Setup

```python
import sys

# CRITICAL: Set mixed precision policy to float32 BEFORE importing training code
# This prevents dtype conflicts when loading EfficientNet backbones
import tensorflow as tf
from keras import mixed_precision

try:
    mixed_precision.set_global_policy('float32')
    print(" Mixed precision policy set to float32")
except:
    tf.keras.backend.set_floatx('float32')
    print(" TensorFlow dtype set to float32")

# Disable mixed precision graph rewrite (for Kaggle/Colab environments)
try:
    tf.config.experimental.enable_mixed_precision_graph_rewrite(False)
    print(" Mixed precision graph rewrite disabled")
except:
    pass

# Add path to import code
sys.path.insert(0, '/kaggle/working')  # Code uploaded to /kaggle/working

# Import code (must be imported AFTER setting mixed precision policy)
from model_train_v3_kaggle import (
    read_labelmap, EnhancedMultiRootVOCDataset,
    make_tf_dataset, build_attention_unet, build_unet_with_boundary,
    build_unet_with_backbone,
    sparse_ce_ignore_index,
    focal_loss, tversky_loss,
    EvalCallback
)
import keras
import numpy as np
import random
from pathlib import Path

print(" Code imported!")
```

### Cell 7: Training Configuration

```python
# Training configuration
EPOCHS = 200  # Kaggle allows longer runs; you can increase epochs
BATCH_SIZE = 8  # P100/T4: 8–16, depending on VRAM and crop_size
LR = 1e-3
CROP_SIZE = 512
ARCHITECTURE = "unet_backbone"  # 'unet', 'attention_unet', 'unet_backbone'
BACKBONE_NAME = "EfficientNetB3"  # 'EfficientNetB0', 'EfficientNetB3', 'EfficientNetB4'
LOSS = "focal"  # 'ce', 'focal', 'tversky'
USE_ADVANCED_AUG = True

# CRITICAL: Do NOT enable mixed precision here since we already set float32 in Cell 6
# Mixed precision may conflict with the EfficientNet backbone

# Seed
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

print(f"Config:")
print(f"  Architecture: {ARCHITECTURE}")
if ARCHITECTURE == "unet_backbone":
    print(f"  Backbone: {BACKBONE_NAME}")
print(f"  Loss: {LOSS}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Crop size: {CROP_SIZE}")
```

### Recommended Patch Size for EfficientNetB3 on P100 GPU

**For EfficientNetB3 + U-Net decoder on a P100 GPU (16GB VRAM):**

| Patch Size | Batch Size | Pros | Cons | Recommendation |
|------------|------------|------|------|----------------|
| **384x384** | 16–20 | Large batch, faster training; VRAM efficient | Lower resolution; may lose details | When you need fast training |
| **512x512** | 8–12 | Good balance; adequate resolution; reasonable batch size | – | **Main recommendation** |
| **640x640** | 4–6 | Higher resolution; better details | Smaller batch; slower training | When you need higher accuracy |
| **768x768** | 2–4 | Very high resolution | Very small batch; potential OOM | Only when necessary |

**Recommended config for EfficientNetB3:**
```python
# For EfficientNetB3 on P100
CROP_SIZE = 512      # Optimal patch size
BATCH_SIZE = 8       # Suitable for 512x512
ARCHITECTURE = "unet_backbone"
BACKBONE_NAME = "EfficientNetB3"
```

**Notes:**
- EfficientNetB3 is larger than B0 (~12M params vs ~5M), so it needs more VRAM
- If you encounter OOM (Out of Memory), reduce `batch_size` or reduce `crop_size` to 384
- If you still have spare VRAM, you can increase `crop_size` to 640 for better quality

### Cell 8: Load Data and Build Datasets

```python
# Load labelmap
names, colors = read_labelmap(Path(LABELMAP_PATH))
num_classes = len(names)
print(f"Classes ({num_classes}): {names}")

# Build datasets
train_ds_wrap = EnhancedMultiRootVOCDataset(
    roots=DATA_ROOTS, image_set="train",
    names=names, colors=colors,
    crop_size=CROP_SIZE,
    use_advanced_aug=USE_ADVANCED_AUG
)
val_ds_wrap = EnhancedMultiRootVOCDataset(
    roots=DATA_ROOTS, image_set="val",
    names=names, colors=colors,
    crop_size=CROP_SIZE,
    use_advanced_aug=False
)

print(f"Train samples: {len(train_ds_wrap)}")
print(f"Val samples: {len(val_ds_wrap)}")

# Create tf.data datasets
train_ds = make_tf_dataset(train_ds_wrap, batch_size=BATCH_SIZE, shuffle=True, ignore_index=255)
val_ds = make_tf_dataset(val_ds_wrap, batch_size=1, shuffle=False, ignore_index=255)
```

### Cell 9: Build Model

```python
# CRITICAL: Ensure the mixed precision policy is still float32
# DO NOT enable mixed_float16 here as it may conflict with EfficientNet
current_policy = str(mixed_precision.global_policy())
print(f"Current mixed precision policy: {current_policy}")
if 'float32' not in current_policy.lower():
    print("  Warning: Policy is not float32! Resetting to float32...")
    mixed_precision.set_global_policy('float32')

# Build model
if ARCHITECTURE == "unet":
    model = build_unet_with_boundary(num_classes=num_classes, dropout=0.2)
elif ARCHITECTURE == "attention_unet":
    model = build_attention_unet(num_classes=num_classes, dropout=0.2)
elif ARCHITECTURE == "unet_backbone":
    model = build_unet_with_backbone(
        num_classes=num_classes,
        backbone="efficientnet",
        backbone_name=BACKBONE_NAME,  # Use BACKBONE_NAME from Cell 7
        dropout=0.2
    )
else:
    model = build_unet_with_boundary(num_classes=num_classes, dropout=0.2)

print(f"Model parameters: {model.count_params():,}")
model.summary()
```

### Cell 10: Set Up Losses and Optimizer

```python
# Set up losses
if LOSS == "ce":
    sem_loss = sparse_ce_ignore_index(ignore_index=255, from_logits=True)
elif LOSS == "focal":
    sem_loss = focal_loss(alpha=0.25, gamma=2.0, ignore_index=255, from_logits=True)
elif LOSS == "tversky":
    sem_loss = tversky_loss(alpha=0.5, beta=0.5, ignore_index=255, from_logits=True)
else:
    sem_loss = sparse_ce_ignore_index(ignore_index=255, from_logits=True)

bce_logits = keras.losses.BinaryCrossentropy(from_logits=True)

# Handle multiple outputs for deep supervision
if ARCHITECTURE == "unet_plusplus" and DEEP_SUPERVISION:
    losses = {
        "ds1": sem_loss,
        "ds2": sem_loss,
        "ds3": sem_loss,
        "sem_logits": sem_loss,
        "boundary_logits": bce_logits
    }
    loss_weights = {
        "ds1": 0.25,
        "ds2": 0.25,
        "ds3": 0.25,
        "sem_logits": 1.0,
        "boundary_logits": 1.0
    }
else:
    losses = {
        "sem_logits": sem_loss,
        "boundary_logits": bce_logits
    }
    loss_weights = {"sem_logits": 1.0, "boundary_logits": 1.0}

optimizer = keras.optimizers.Adam(learning_rate=LR, clipnorm=1.0)
model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

print(" Model compiled!")
```

### Cell 11: Set Up Callbacks and Train

```python
# Create directory to save the model
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

# Callbacks
ckpt_path = Path(SAVE_DIR) / f"{ARCHITECTURE}_{LOSS}_best.keras"
eval_cb = EvalCallback(val_ds, num_classes=num_classes, ignore_index=255, ckpt_path=ckpt_path)
# Note: EvalCallback automatically adds the following metrics to logs:
# - val_loss: negative mIoU (used by ReduceLROnPlateau)
# - val_miou: mean Intersection over Union
# - val_pa: Pixel Accuracy
# - val_bce: Binary Cross Entropy (for boundary)

# Custom callback to save a checkpoint every 10 epochs
class PeriodicCheckpoint(keras.callbacks.Callback):
    def __init__(self, filepath, period=10):
        super().__init__()
        self.filepath = filepath
        self.period = period
    
    def on_epoch_end(self, epoch, logs=None):
        # Save at epochs 10, 20, 30, ... (epoch is 0-indexed, hence epoch+1)
        if (epoch + 1) % self.period == 0:
            filepath = self.filepath.format(epoch=epoch + 1)
            self.model.save(filepath)
            print(f"Saved checkpoint: {filepath}")

# Save a checkpoint every 10 epochs
periodic_checkpoint_cb = PeriodicCheckpoint(
    filepath=str(Path(SAVE_DIR) / f"{ARCHITECTURE}_{LOSS}_epoch{{epoch:02d}}.keras"),
    period=10
)

# Reduce learning rate when there is no improvement
# EvalCallback will automatically add val_loss to logs (computed from negative mIoU)
# When val_loss does not improve for 5 consecutive epochs, LR is reduced by 50%
lr_callback = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',  # EvalCallback automatically adds this metric
    factor=0.5,          # Reduce LR to 50% when not improving
    patience=5,          # Wait 5 epochs without improvement
    min_lr=1e-6,         # Minimum LR
    verbose=1            # Show messages when reducing LR
)

# TensorBoard (optional)
tensorboard_cb = keras.callbacks.TensorBoard(
    log_dir=str(Path(SAVE_DIR) / "logs"),
    histogram_freq=1
)

print("Starting training...")
print(f"Save directory: {SAVE_DIR}")
print("Best model will be saved automatically by EvalCallback")
print("Checkpoints will be saved every 10 epochs")
print("Learning rate will be reduced automatically when val_loss plateaus")

# Train
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=[eval_cb, periodic_checkpoint_cb, lr_callback, tensorboard_cb],
    verbose=1
)

print(f"\n Training completed!")
print(f"Best model: {ckpt_path}")
```

---

## Step 7: Save and Load the Model

### Save the model

```python
# The model is automatically saved by the callbacks to /kaggle/working/models/
# Best model: {SAVE_DIR}/{ARCHITECTURE}_{LOSS}_best.keras
# Checkpoints: {SAVE_DIR}/{ARCHITECTURE}_{LOSS}_epochXX.keras

# Files in /kaggle/working/ are automatically persisted when you commit the notebook
```

### Load the model for inference

```python
# Load the trained model
model_path = f"{SAVE_DIR}/{ARCHITECTURE}_{LOSS}_best.keras"
model = keras.models.load_model(model_path)
print(" Model loaded!")
```

### Download the model to your local machine

```python
# In Kaggle notebooks, files in /kaggle/working/ are saved when you commit
# Or download manually:
from IPython.display import FileLink
FileLink(f"{SAVE_DIR}/{ARCHITECTURE}_{LOSS}_best.keras")
```

---

## Step 7.5: Test the Model

### Upload the test script

**Option 1: Upload via File Browser**
- Click the file browser on the left
- Upload `test_model_kaggle.py` to `/kaggle/working/`

**Option 2: Paste code directly**
```python
# Create test_model_kaggle.py
%%writefile /kaggle/working/test_model_kaggle.py
# Paste the entire content of test_model_kaggle.py here
```

### Test Cell: Run with defaults

```python
# Import and run the test script
import sys
sys.path.insert(0, '/kaggle/working')

from test_model_kaggle import main

# Run with defaults (automatically finds model, image, labelmap)
main()
```

### Test Cell: Run with explicit arguments

```python
# Import and run the test script with specific arguments
import sys
sys.path.insert(0, '/kaggle/working')

from test_model_kaggle import main
import sys

# Set arguments
sys.argv = [
    'test_model_kaggle.py',
    '--model_path', '/kaggle/working/models/attention_unet_focal_best.keras',
    '--image_path', '/kaggle/working/data/cheetah/JPEGImages/00000000.jpg',
    '--output_dir', '/kaggle/working/test_results',
    '--labelmap', '/kaggle/working/labelmap.txt',
    '--save_boundary'
]

main()
```

### Or run directly from the command line

```python
# Run the script as a Python program
!python /kaggle/working/test_model_kaggle.py \
    --model_path /kaggle/working/models/attention_unet_focal_best.keras \
    --image_path /kaggle/working/data/cheetah/JPEGImages/00000000.jpg \
    --output_dir /kaggle/working/test_results \
    --labelmap /kaggle/working/labelmap.txt \
    --save_boundary
```

### View results

```python
# Display the result files
from pathlib import Path
from IPython.display import Image, display

output_dir = Path("/kaggle/working/test_results")

if output_dir.exists():
    print(" Files in test_results:")
    for file in output_dir.glob("*.png"):
        print(f"   - {file.name}")
        
    # Display some results
    if (output_dir / "pred_color.png").exists():
        print("\n  Colorized prediction:")
        display(Image(str(output_dir / "pred_color.png")))
    
    if (output_dir / "pred_overlay.png").exists():
        print("\n  Overlay prediction:")
        display(Image(str(output_dir / "pred_overlay.png")))
else:
    print(" test_results folder does not exist")
    print(" Please run the test script first")
```

### Download results to your local machine

```python
# Create download links for result files
from IPython.display import FileLink

output_dir = Path("/kaggle/working/test_results")
if output_dir.exists():
    for file in output_dir.glob("*.png"):
        print(f" Download {file.name}:")
        display(FileLink(str(file)))
```

---

## Step 8: Monitor Training

### View training history

```python
import matplotlib.pyplot as plt

# Plot loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history.get('val_loss', []), label='val')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history.get('sem_logits_loss', []), label='semantic')
plt.plot(history.history.get('boundary_logits_loss', []), label='boundary')
plt.title('Component Losses')
plt.legend()

plt.tight_layout()
plt.show()
```

---

## Important Notes

1. **Kaggle Session Limits:**
   - Free tier: ~30 GPU hours/week
   - Session timeout: ~9 hours
   - Auto-save when you commit the notebook

2. **Saving the Model:**
   - **ALWAYS save the model to `/kaggle/working/`** (auto-saved when you commit)
   - Models are automatically saved by callbacks into `SAVE_DIR`
   - Files in `/kaggle/working/` persist when you commit the notebook

3. **GPU Limits:**
   - Free tier: ~30 GPU hours/week
   - You can purchase Kaggle Pro for more GPU time
   - GPU will automatically disconnect after ~9 hours

4. **Batch Size:**
   - P100 GPU: batch_size = 16–32
   - T4 GPU: batch_size = 16–24
   - T4 x2 GPU: batch_size = 32–48
   - Adjust according to GPU VRAM

5. **Data Size:**
   - Datasets can be up to 20GB (free tier) or 100GB (Pro)
   - Upload the dataset once and reuse many times
   - Files in `/kaggle/input/` are read-only

6. **Internet Access:**
   - Kaggle notebooks have internet access to download ImageNet weights
   - No need to worry about downloading pre-trained models

---

## Quick Start – Run the Auto Script

If you want to run quickly, use the automation script:

```python
# Run the automation script (auto-detects Kaggle and sets paths)
exec(open('/kaggle/working/model_train_v3_kaggle.py').read())

# Or call the main function
from model_train_v3_kaggle import main_unet
main_unet()
```

---

## Troubleshooting

### Error: "No GPU available"
- Settings → Accelerator → GPU → Save
- Ensure the notebook is running with a GPU (not CPU)

### Error: "Out of memory"
- Reduce `BATCH_SIZE` to 8–12
- Reduce `CROP_SIZE` to 256
- Use gradient checkpointing

### Error: "Dataset not found"
- Check that you added the dataset to the notebook
- Verify the dataset name in your code is correct
- Dataset path: `/kaggle/input/YOUR_DATASET_NAME/`

### Error: "Data folder not found after extracting"
- Check the ZIP structure:
  - The ZIP should contain a top-level `data/` folder
  - Or extraction should create a `data/` folder
- Verify that Cell 3 (extraction) ran successfully
- Inspect extraction logs to see where files were extracted

### Model not saved
- Check whether `SAVE_DIR` exists
- Ensure you are saving to `/kaggle/working/`
- Files are saved when you commit the notebook

### Kaggle session timeout
- Kaggle automatically saves files in `/kaggle/working/` when you commit
- The model is saved automatically by callbacks
- You can resume training by loading a checkpoint

---

## Pre-Training Checklist

- [ ] Zipped the `data/` folder into a ZIP file
- [ ] Created a dataset on Kaggle and uploaded the ZIP + `labelmap.txt`
- [ ] Created a new notebook
- [ ] Chosen GPU in Settings → Accelerator
- [ ] Added the dataset to the notebook
- [ ] Uploaded the code (`model_train_v3_kaggle.py`) and `labelmap.txt` (if not in the dataset)
- [ ] Ran Cell 3 to extract the ZIP (if the dataset is a ZIP)
- [ ] Verified data paths in Cell 5
- [ ] Installed dependencies
- [ ] Configured `SAVE_DIR` to save to `/kaggle/working/`
- [ ] Checked that the batch size matches your GPU

---

