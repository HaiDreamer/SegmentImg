from keras import layers, Model
from .blocks import downsample_block, double_conv_block

'''faster spd but lower accuracy, NO use'''

def build_unet_plusplus(input_shape=(512,512,3), num_classes=6, dropout=0.2, use_batchnorm=True, deep_supervision=True):
    inputs = layers.Input(shape=input_shape)
    x00 = inputs
    f1, p1 = downsample_block(x00, 64, dropout, use_batchnorm); x10 = f1
    f2, p2 = downsample_block(p1, 128, dropout, use_batchnorm); x20 = f2
    f3, p3 = downsample_block(p2, 256, dropout, use_batchnorm); x30 = f3
    f4, p4 = downsample_block(p3, 512, dropout, use_batchnorm); x40 = f4
    x50 = double_conv_block(p4, 1024, use_bn=use_batchnorm)

    x01 = x00
    x11_up = layers.Conv2DTranspose(64, 3, 2, padding="same")(x10)
    x11 = double_conv_block(layers.Dropout(dropout)(layers.Concatenate()([x11_up, x01])), 64, use_bn=use_batchnorm)

    x21_up = layers.Conv2DTranspose(128, 3, 2, padding="same")(x20)
    x21 = double_conv_block(layers.Dropout(dropout)(layers.Concatenate()([x21_up, x11])), 128, use_bn=use_batchnorm)

    x31_up = layers.Conv2DTranspose(256, 3, 2, padding="same")(x30)
    x31 = double_conv_block(layers.Dropout(dropout)(layers.Concatenate()([x31_up, x21])), 256, use_bn=use_batchnorm)

    x41_up = layers.Conv2DTranspose(512, 3, 2, padding="same")(x40)
    x41 = double_conv_block(layers.Dropout(dropout)(layers.Concatenate()([x41_up, x31])), 512, use_bn=use_batchnorm)

    x51_up = layers.Conv2DTranspose(1024, 3, 2, padding="same")(x50)
    x51 = double_conv_block(layers.Dropout(dropout)(layers.Concatenate()([x51_up, x41])), 1024, use_bn=use_batchnorm)

    outputs = []
    if deep_supervision:
        outputs += [
            layers.Conv2D(num_classes, 1, padding="same", name="ds1")(x21),
            layers.Conv2D(num_classes, 1, padding="same", name="ds2")(x31),
            layers.Conv2D(num_classes, 1, padding="same", name="ds3")(x41),
        ]
    sem = layers.Conv2D(num_classes, 1, padding="same", name="sem_logits")(x51)
    bnd = layers.Conv2D(1, 1, padding="same", name="boundary_logits")(x51)
    outputs += [sem, bnd]
    return Model(inputs, outputs, name="UNetPlusPlus")
