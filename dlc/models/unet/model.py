import tensorflow as tf

from .decoder import creade_decoder_block
from .encoder import create_encoder_block
from .unit import create_unit_block
from .unit import UNIT_VARIANTS
from .upsampling import UPSAMPLING_VARIANTS
from dlc.initializers import InitializerContainer
from dlc.relu import RELU_VARIANTS


__all__ = [
    "create_model",
    "load_model",
    "add_cropping_if_necessary",
    "create_attention_block",
    "create_encoder_block",
    "creade_decoder_block",
]


def load_model(input_shape, batch_size, weights_file, **kwargs):
    model = create_model(
        input_shape=input_shape,
        batch_size=batch_size,
        **kwargs,
    )
    model.load_weights(weights_file)
    return model


def create_model(
    *,  # default values defined in settings file
    input_shape,
    output_channels,
    batch_size,
    input_dtype,
    name,
    output_activation,
    output_name,
    padding,  # "valid" means unpadded and is the default in [1]
    depth,  # 4 is the default depth value used in [1] and [2]
    layer_count,  # 64 is default layer count used in [1] and [2]
    output_resize_interpolation,
    hidden_activation,
    encoder_unit,  # "vanilla" is the variant from [1]
    encoder_pooling,  # "max" is the default in [1]
    encoder_pooling_batch_norm,
    decoder_unit,
    decoder_upsampling,
    decoder_upsampling_use_batch_norm,
    decoder_use_attention_gate,
    decoder_attention_relu_variant,
    decoder_attention_use_batch_norm,
    decoder_concat_use_batch_norm,
    # optional L2 weight-decay on output layers
    kernel_regularizer,
    # optional L1 regularization on ReLU activations
    relu_activity_l1_regularizer,
    # optional deep supervision
    deeply_supervised,
    # optional second output for supporting multi-task
    second_output_name=None,
    second_output_activation=None,
    second_output_preceding_unit=None,
    init_container=None,
    **kwargs,  # catch-all dummy
):
    """Create a U-Net model with the specified parameters."""
    assert padding in ["same", "valid"]
    assert output_activation in ["linear", "sigmoid"]
    assert hidden_activation in RELU_VARIANTS
    assert encoder_pooling in ["max", "avg"]
    assert decoder_upsampling in UPSAMPLING_VARIANTS
    assert encoder_unit in UNIT_VARIANTS
    assert decoder_unit in UNIT_VARIANTS
    if relu_activity_l1_regularizer is not None:
        assert hidden_activation == "relu"

    if init_container is None:
        init_container = InitializerContainer(**kwargs)

    hidden_outputs = [] if deeply_supervised else None

    input = tf.keras.Input(
        shape=input_shape,
        batch_size=batch_size,
        dtype=input_dtype,
        name="input",
    )
    x = input

    # Encoding/contracting path
    residuals = []
    for level in range(depth):
        x, r = create_encoder_block(
            x,
            level=level,
            layer_count=layer_count,
            unit=encoder_unit,
            padding=padding,
            pooling=encoder_pooling,
            pooling_batch_norm=encoder_pooling_batch_norm,
            activation=hidden_activation,
            init_container=init_container,
            kernel_regularizer=None,
            relu_activity_l1_regularizer=relu_activity_l1_regularizer,
        )
        residuals.append(r)

    # Encoding-decoding interface
    x = create_unit_block(
        x,
        variant=encoder_unit,
        num_filters=(2 ** depth) * layer_count,
        activation=hidden_activation,
        init_container=init_container,
        padding=padding,
        block_name="interface",
        is_encoder=True,
        kernel_regularizer=None,
        relu_activity_l1_regularizer=relu_activity_l1_regularizer,
    )

    # Decoding/expanding path
    for level in range(depth - 1, -1, -1):
        x = creade_decoder_block(
            x,
            residuals[level],
            level=level,
            layer_count=layer_count,
            unit=decoder_unit,
            padding=padding,
            upsampling=decoder_upsampling,
            upsampling_use_batch_norm=decoder_upsampling_use_batch_norm,
            activation=hidden_activation,
            use_attention_gate=decoder_use_attention_gate,
            attention_relu_variant=decoder_attention_relu_variant,
            attention_use_batch_norm=decoder_attention_use_batch_norm,
            concat_use_batch_norm=decoder_concat_use_batch_norm,
            init_container=init_container,
            kernel_regularizer=None,
            relu_activity_l1_regularizer=relu_activity_l1_regularizer,
        )
        if deeply_supervised and level > 0:
            hidden_out = tf.keras.layers.Conv2D(
                output_channels,
                (1, 1),
                activation=output_activation,
                padding="same",
                kernel_initializer=init_container.get_kernel_init(
                    output_activation,
                    is_output=True,
                ),
                bias_initializer=init_container.get_bias_init(
                    output_activation,
                    is_output=True,
                ),
                name=f"{output_name}_hidden_{level}",
                kernel_regularizer=kernel_regularizer,
            )(x)
            hidden_outputs.append(hidden_out)

    # Output layers
    x1 = tf.keras.layers.Conv2D(
        output_channels,
        (1, 1),
        padding="same",
        kernel_initializer=init_container.get_kernel_init(
            output_activation,
            is_output=True,
        ),
        bias_initializer=init_container.get_bias_init(
            output_activation,
            is_output=True,
        ),
        name="output_fusing_conv",
        kernel_regularizer=kernel_regularizer,
    )(x)

    x2 = None
    if second_output_name is not None:
        x2 = x
        if second_output_preceding_unit is not None:
            x2 = create_unit_block(
                x2,
                variant=second_output_preceding_unit,
                num_filters=layer_count,
                activation=hidden_activation,
                init_container=init_container,
                padding=padding,
                block_name="second_output_nonlinearity",
                is_encoder=True,
                kernel_regularizer=None,
                relu_activity_l1_regularizer=relu_activity_l1_regularizer,
            )

        x2 = tf.keras.layers.Conv2D(
            output_channels,
            (1, 1),
            padding="same",
            kernel_initializer=init_container.get_kernel_init(
                second_output_activation,
                is_output=True,
            ),
            bias_initializer=init_container.get_bias_init(
                second_output_activation,
                is_output=True,
            ),
            name="second_output_fusing_conv",
            kernel_regularizer=kernel_regularizer,
        )(x2)

    # Optional for resizing
    if output_resize_interpolation is not None:
        assert padding == "valid", "using 'same' padding with output resize"
        # Output activation
        x1 = tf.keras.layers.Activation(
            output_activation,
            name="output_activation",
        )(x1)
        # Output resize
        x1 = tf.keras.layers.experimental.preprocessing.Resizing(
            input_shape[0],
            input_shape[1],
            interpolation=output_resize_interpolation,
            name=output_name,
        )(x1)
        if x2 is not None:
            # Output activation
            x2 = tf.keras.layers.Activation(
                second_output_activation,
                name="second_output_activation",
            )(x2)
            # Output resize
            x2 = tf.keras.layers.experimental.preprocessing.Resizing(
                input_shape[0],
                input_shape[1],
                interpolation=output_resize_interpolation,
                name=second_output_name,
            )(x2)
    else:
        # Output activation
        x1 = tf.keras.layers.Activation(
            output_activation,
            name=output_name,
        )(x1)
        if x2 is not None:
            x2 = tf.keras.layers.Activation(
                second_output_activation,
                name=second_output_name,
            )(x2)

    outputs = x1 if x2 is None else [x1, x2]
    if deeply_supervised:
        outputs = [x1] if x2 is None else [x1, x2]
        outputs += list(reversed(hidden_outputs))

    model = tf.keras.Model(inputs=input, outputs=outputs, name=name)
    return model
