from keras import layers

'''Blocks for u-net model'''

def double_conv_block(x, n_filters, use_bn=True):
    x = layers.Conv2D(n_filters, 3, padding="same", kernel_initializer="he_normal", use_bias=not use_bn)(x)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(n_filters, 3, padding="same", kernel_initializer="he_normal", use_bias=not use_bn)(x)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def attention_gate(g, x, n_filters):
    theta_g = layers.Conv2D(n_filters, 1, padding="same")(g)
    phi_x = layers.Conv2D(n_filters, 1, padding="same")(x)
    add = layers.Add()([theta_g, phi_x])
    psi = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(layers.ReLU()(add))
    return layers.Multiply()([x, psi])

def downsample_block(x, n_filters, dropout=0.2, use_bn=True):
    f = double_conv_block(x, n_filters, use_bn=use_bn)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(dropout)(p)
    return f, p

def upsample_block(x, conv_feature, n_filters, dropout=0.2, use_bn=True, use_attention=False):
    x = layers.Conv2DTranspose(n_filters, kernel_size=3, strides=2, padding="same")(x)
    if use_attention:
        from .blocks import attention_gate
        conv_feature = attention_gate(x, conv_feature, n_filters)
    x = layers.Concatenate(axis=-1)([x, conv_feature])
    x = layers.Dropout(dropout)(x)
    x = double_conv_block(x, n_filters, use_bn=use_bn)
    return x
