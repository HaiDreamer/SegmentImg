from keras import layers, Model
from .blocks import downsample_block, upsample_block, double_conv_block

def build_attention_unet(input_shape=(512,512,3), num_classes=6, dropout=0.2, use_batchnorm=True):
    inputs = layers.Input(shape=input_shape)
    f1, p1 = downsample_block(inputs, 64, dropout, use_batchnorm)
    f2, p2 = downsample_block(p1, 128, dropout, use_batchnorm)
    f3, p3 = downsample_block(p2, 256, dropout, use_batchnorm)
    f4, p4 = downsample_block(p3, 512, dropout, use_batchnorm)
    bottleneck = double_conv_block(p4, 1024, use_bn=use_batchnorm)
    u6 = upsample_block(bottleneck, f4, 512, dropout, use_batchnorm, use_attention=True)
    u7 = upsample_block(u6, f3, 256, dropout, use_batchnorm, use_attention=True)
    u8 = upsample_block(u7, f2, 128, dropout, use_batchnorm, use_attention=True)
    u9 = upsample_block(u8, f1, 64, dropout, use_batchnorm, use_attention=True)
    sem = layers.Conv2D(num_classes, 1, padding="same", name="sem_logits")(u9)
    bnd = layers.Conv2D(1, 1, padding="same", name="boundary_logits")(u9)
    return Model(inputs, [sem, bnd], name="AttentionUNet")
