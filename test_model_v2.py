import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
import keras
import argparse

def read_labelmap(labelmap_path: Path):
    names, colors = [], []
    if not labelmap_path.exists():
        return names, colors
    for raw in Path(labelmap_path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        name, rest = line.split(":", 1)
        name = name.strip()
        color_field = rest.split(":", 1)[0]
        r, g, b = [int(c.strip()) for c in color_field.split(",")]
        names.append(name)
        colors.append((r, g, b))
    return names, colors

def colorize_index_mask(mask: np.ndarray, colors):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if colors:
        for idx, rgb in enumerate(colors):
            out[mask == idx] = rgb
    return Image.fromarray(out, mode="RGB")

def preprocess(img: Image.Image) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return (arr - mean) / std

def main():
    parser = argparse.ArgumentParser(description="Test model (LOCAL)")
    parser.add_argument("--model_path", type=str, help="Path to model (.keras)")
    parser.add_argument("--image_path", type=str, help="Path to test image")
    parser.add_argument("--output_dir", type=str, help="Folder to save results")
    parser.add_argument("--labelmap", type=str, help="Path to labelmap.txt")
    parser.add_argument("--save_boundary", action="store_true", help="Save boundary heatmap")

    if len(sys.argv) == 1:
        parser.set_defaults(
            model_path=r"D:\animal_data\models\unet_boundary_best.keras",
            image_path=r"D:\animal_data\data\cheetah\JPEGImages\00000000_512resized.png",
            output_dir=r"D:\animal_data\test_results",
            labelmap=r"D:\animal_data\img_segment\labelmap.txt",
            save_boundary=True
        )

    args = parser.parse_args()

    model_path   = Path(args.model_path)
    image_path   = Path(args.image_path)
    labelmap_path= Path(args.labelmap)
    output_dir   = Path(args.output_dir)

    print("="*60)
    print("TEST MODEL (LOCAL)")
    print("="*60)
    print(" Running on: Local Machine")
    print(f" Model: {model_path}")
    print(f"  Image: {image_path}")
    print(f" Output: {output_dir}")
    print(f"  Labelmap: {labelmap_path}")
    print("="*60)

    if not model_path.exists():
        print(f"\n Model not found: {model_path}")
        sys.exit(1)

    if not image_path.exists():
        print(f"\n Image not found: {image_path}")
        sys.exit(1)

    if not labelmap_path.exists():
        print(f"\n Labelmap not found: {labelmap_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n Loading model...")
    try:
        model = keras.models.load_model(model_path.as_posix(), compile=False)
        print(f" Model loaded! Input shape: {model.input_shape}")
    except Exception as e:
        print(f" Error loading model: {e}")
        sys.exit(1)

    names, colors = read_labelmap(labelmap_path)
    print(f" Labelmap loaded! Classes: {names}")

    print("\n  Loading image...")
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size
    print(f"   Original size: {orig_size}")

    in_shape = model.input_shape
    if len(in_shape) == 4 and in_shape[1] is not None and in_shape[2] is not None:
        exp_h, exp_w = in_shape[1], in_shape[2]
        if img.size != (exp_w, exp_h):
            img = img.resize((exp_w, exp_h), Image.BILINEAR)
            print(f"   Resized to: {exp_w}x{exp_h}")

    x = preprocess(img)[None, ...]

    print("\n Running inference...")

    input_name = None
    if hasattr(model, 'input_names') and model.input_names:
        input_name = model.input_names[0]
    elif hasattr(model, 'inputs') and len(model.inputs) > 0:
        input_layer = model.inputs[0]
        if hasattr(input_layer, 'name'):
            input_name = input_layer.name
    if not input_name:
        input_name = 'input_layer_1'

    try:
        model_input = {input_name: x}
        outputs = model(model_input, training=False)
    except (KeyError, TypeError, ValueError):
        outputs = model(x, training=False)

    if isinstance(outputs, list):
        sem_logits = outputs[0]
        boundary_logits = outputs[1] if len(outputs) > 1 else None
    elif isinstance(outputs, dict):
        sem_logits = outputs.get("sem_logits")
        boundary_logits = outputs.get("boundary_logits")
    else:
        sem_logits = outputs
        boundary_logits = None

    pred = tf.argmax(sem_logits, axis=-1)[0].numpy().astype(np.int32)

    print("\n Saving results...")

    Image.fromarray(pred.astype(np.uint8), mode="L").save(output_dir / "pred_index.png")

    pred_color = colorize_index_mask(pred, colors)
    pred_color.save(output_dir / "pred_color.png")

    if boundary_logits is not None and args.save_boundary:
        if boundary_logits.ndim == 4:
            boundary_prob = tf.nn.sigmoid(boundary_logits)[0, ..., 0].numpy()
        else:
            boundary_prob = tf.nn.sigmoid(boundary_logits[..., 0]).numpy()
        boundary_img = Image.fromarray((boundary_prob * 255).astype(np.uint8), mode="L")
        boundary_img.save(output_dir / "pred_boundary.png")
        print("    Saved boundary heatmap")

    img_orig = Image.open(image_path).convert("RGB")

    if orig_size != img.size:
        pred_resized = Image.fromarray(pred.astype(np.uint8), mode="L").resize(orig_size, Image.NEAREST)
        pred_color_resized = colorize_index_mask(np.array(pred_resized), colors)
    else:
        pred_color_resized = colorize_index_mask(pred, colors)

    overlay = Image.blend(img_orig, pred_color_resized, 0.5)
    overlay.save(output_dir / "pred_overlay.png")
    print("    Saved overlay")

    print("\n Test completed!")
    print(f"\n Results saved to: {output_dir}")
    print("   - pred_index.png (grayscale mask)")
    print("   - pred_color.png (colorized mask)")
    if boundary_logits is not None and args.save_boundary:
        print("   - pred_boundary.png (boundary heatmap)")
    print("   - pred_overlay.png (overlay on original image)")

    unique_classes, counts = np.unique(pred, return_counts=True)
    print("\n Prediction statistics:")
    for cls_id, count in zip(unique_classes, counts):
        if cls_id < len(names):
            print(f"   {names[cls_id]}: {count} pixels ({count/pred.size*100:.1f}%)")

if __name__ == "__main__":
    main()
