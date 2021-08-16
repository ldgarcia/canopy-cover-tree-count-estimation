import tensorflow as tf

from dlc.relu import get_relu_variant_layer

# The function below was adapted for TF from [5], following the paper's
# code reference: https://git.io/Jsidw.
# Also influenced by the TF implementation by Yingkai Sha: https://git.io/JWKNK


def create_attention_block(
    x,
    g,
    *,
    in_channels,
    level,
    init_container,
    relu_variant,
    upsampling_size=(2, 2),
    kernel_regularizer=None,
    relu_activity_l1_regularizer=None,
    use_batch_norm=False,
    block_name=None,
):
    """Create an additive attention gate for U-Net [5]."""
    if block_name is None:
        block_name = f"attention_gate_{level}"
    else:
        block_name = f"{block_name}_attention_gate_{level}"

    inter_channels = in_channels // 4
    if inter_channels == 0:
        inter_channels = 1

    theta_x = tf.keras.layers.Conv2D(
        inter_channels,
        [1, 1],
        activation=None,
        use_bias=False,
        kernel_initializer=init_container.get_kernel_init(relu_variant),
        bias_initializer=init_container.get_bias_init(relu_variant),
        name=f"{block_name}_theta_x__lsuv",
        kernel_regularizer=kernel_regularizer,
    )(x)

    phi_g = tf.keras.layers.Conv2D(
        inter_channels,
        [1, 1],
        activation=None,
        use_bias=True,
        name=f"{block_name}_phi_g__lsuv",
        kernel_initializer=init_container.get_kernel_init(relu_variant),
        bias_initializer=init_container.get_bias_init(relu_variant),
        kernel_regularizer=kernel_regularizer,
    )(g)

    phi_g_up = tf.keras.layers.UpSampling2D(
        size=upsampling_size,
        interpolation="bilinear",
        name=f"{block_name}_g_up",
    )(phi_g)

    addition = tf.keras.layers.add(
        [theta_x, phi_g_up],
        name=f"{block_name}_add",
    )

    f = get_relu_variant_layer(
        variant=relu_variant,
        name=f"{block_name}_{relu_variant}__lsuv_activation",
    )(addition)

    if relu_activity_l1_regularizer is not None:
        x = tf.keras.layers.ActivityRegularization(
            l1=relu_activity_l1_regularizer,
            name=f"{block_name}_relu_reg",
        )(x)

    sigma_psi_f = tf.keras.layers.Conv2D(
        1,
        [1, 1],
        activation="sigmoid",
        use_bias=True,
        name=f"{block_name}_psi_f",
        kernel_initializer=init_container.get_kernel_init("sigmoid"),
        bias_initializer=init_container.get_bias_init("sigmoid"),
        kernel_regularizer=kernel_regularizer,
    )(f)

    y = tf.keras.layers.multiply(
        [x, sigma_psi_f],
        name=f"{block_name}_multiply",
    )

    if use_batch_norm:
        y = tf.keras.layers.BatchNormalization(
            name=f"{block_name}_batch_norm",
        )(y)

    return y
