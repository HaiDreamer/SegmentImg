from keras import layers, Model
from .blocks import double_conv_block
EFFICIENTNET_AVAILABLE = RESNET_AVAILABLE = True
try:
    from keras.applications import EfficientNetB0, EfficientNetB3, EfficientNetB4
except Exception:
    EFFICIENTNET_AVAILABLE = False
try:
    from keras.applications import ResNet50, ResNet101
except Exception:
    RESNET_AVAILABLE = False

def build_unet_with_backbone(input_shape=(512,512,3), num_classes=6,
                             backbone="efficientnet", backbone_name="EfficientNetB0", dropout=0.2):
    inputs = layers.Input(shape=input_shape)
    if backbone == "efficientnet" and EFFICIENTNET_AVAILABLE:
        if backbone_name == "EfficientNetB0": bb = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)
        elif backbone_name == "EfficientNetB3": bb = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inputs)
        elif backbone_name == "EfficientNetB4": bb = EfficientNetB4(include_top=False, weights="imagenet", input_tensor=inputs)
        else: bb = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)
        skip_layers = []
        for name in ["block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation"]:
            try: skip_layers.append(bb.get_layer(name).output)
            except: pass
        x = bb.output
    elif backbone == "resnet" and RESNET_AVAILABLE:
        if backbone_name == "ResNet50": bb = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
        elif backbone_name == "ResNet101": bb = ResNet101(include_top=False, weights="imagenet", input_tensor=inputs)
        else: bb = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
        skip_layers = [bb.get_layer("conv2_block3_out").output,
                       bb.get_layer("conv3_block4_out").output,
                       bb.get_layer("conv4_block6_out").output]
        x = bb.output
    else:
        from .unet_boundary import build_unet_with_boundary as _fallback
        return _fallback(input_shape=input_shape, num_classes=num_classes, dropout=dropout)

    for i, skip in enumerate(reversed(skip_layers)):
        n_filters = 512 // (2 ** i)
        x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
        x = layers.Concatenate()([x, skip])
        x = double_conv_block(x, n_filters, use_bn=True)
        x = layers.Dropout(dropout)(x)

    x = layers.Conv2DTranspose(64, 3, 2, padding="same")(x)
    x = double_conv_block(x, 64, use_bn=True)

    sem = layers.Conv2D(num_classes, 1, padding="same", name="sem_logits")(x)
    bnd = layers.Conv2D(1, 1, padding="same", name="boundary_logits")(x)
    return Model(inputs, [sem, bnd], name=f"UNet_{backbone_name}")
