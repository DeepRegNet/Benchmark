"""This script provides an example of using custom backbone for training."""

import argparse
from typing import Tuple, Union

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as tfkl
from deepreg.model.backbone import UNet
from deepreg.registry import REGISTRY
from deepreg.train import train


@REGISTRY.register_backbone(name="vm_balakrishnan_2019")
class VoxelMorphBalakrishnan2019(UNet):
    """Reproduce https://arxiv.org/abs/1809.05231."""

    def __init__(self, **kwargs):
        """
        Init.

        Args:
            **kwargs:
        """
        super().__init__(**kwargs)

        self._out_ddf_upsampling = tf.keras.layers.UpSampling3D(size=2)
        self._out_ddf_conv = tfkl.Conv3D(
            filters=3,
            kernel_size=3,
            padding="same",
            activation=self.get_activation(),
        )

    def build_encode_conv_block(
        self, filters: int, kernel_size: int, padding: str
    ) -> Union[tf.keras.Model, tfkl.Layer]:
        """
        Build a conv block for down-sampling.

        :param filters: number of channels for output
        :param kernel_size: arg for conv3d
        :param padding: arg for conv3d
        :return: a block consists of one or multiple layers
        """
        return tfkl.Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            strides=2,
            activation=self.get_activation(),
        )

    def build_down_sampling_block(
        self, filters: int, kernel_size: int, padding: str, strides: int
    ) -> Union[tf.keras.Model, tfkl.Layer]:
        """
        Return identity layer.

        :param filters: number of channels for output, arg for conv3d
        :param kernel_size: arg for pool3d or conv3d
        :param padding: arg for pool3d or conv3d
        :param strides: arg for pool3d or conv3d
        :return: a block consists of one or multiple layers
        """
        return tfkl.Lambda(lambda x: x)

    def build_bottom_block(
        self, filters: int, kernel_size: int, padding: str
    ) -> Union[tf.keras.Model, tfkl.Layer]:
        """
        Return down sample layer.

        :param filters: number of channels for output
        :param kernel_size: arg for conv3d
        :param padding: arg for conv3d
        :return: a block consists of one or multiple layers
        """
        return tfkl.Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            strides=2,
            activation=self.get_activation(),
        )

    def build_up_sampling_block(
        self,
        filters: int,
        output_padding: int,
        kernel_size: int,
        padding: str,
        strides: int,
        output_shape: tuple,
    ) -> Union[tf.keras.Model, tfkl.Layer]:
        """
        Build a block for up-sampling.

        This block changes the tensor shape (width, height, depth),
        but it does not changes the number of channels.

        :param filters: number of channels for output
        :param output_padding: padding for output
        :param kernel_size: arg for deconv3d
        :param padding: arg for deconv3d
        :param strides: arg for deconv3d
        :param output_shape: shape of the output tensor
        :return: a block consists of one or multiple layers
        """
        return tf.keras.layers.UpSampling3D(size=strides)

    def build_decode_conv_block(
        self, filters: int, kernel_size: int, padding: str
    ) -> Union[tf.keras.Model, tfkl.Layer]:
        """
        Build a conv block for up-sampling.

        :param filters: number of channels for output
        :param kernel_size: arg for conv3d
        :param padding: arg for conv3d
        :return: a block consists of one or multiple layers
        """
        return tfkl.Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            strides=1,
            activation=self.get_activation(),
        )

    def build_output_block(
        self,
        image_size: Tuple[int],
        extract_levels: Tuple[int],
        out_channels: int,
        out_kernel_initializer: str,
        out_activation: str,
    ) -> Union[tf.keras.Model, tfkl.Layer]:
        """
        Build a block for output.

        The input to this block is a list of tensors.

        :param image_size: such as (dim1, dim2, dim3)
        :param extract_levels: number of extraction levels.
        :param out_channels: number of channels for the extractions
        :param out_kernel_initializer: initializer to use for kernels.
        :param out_activation: activation to use at end layer.
        :return: a block consists of one or multiple layers
        """
        return tf.keras.Sequential(
            [
                tfkl.Lambda(lambda x: x[0]),  # take the first one / depth 0
                tfkl.Conv3D(
                    filters=self.num_channel_initial,
                    kernel_size=3,
                    padding="same",
                    activation=self.get_activation(),
                ),
                tfkl.Conv3D(
                    filters=self.num_channel_initial,
                    kernel_size=3,
                    padding="same",
                ),
            ]
        )

    def call(self, inputs: tf.Tensor, training=None, mask=None) -> tf.Tensor:
        """
        Build LocalNet graph based on built layers.

        :param inputs: image batch, shape = (batch, f_dim1, f_dim2, f_dim3, ch)
        :param training: None or bool.
        :param mask: None or tf.Tensor.
        :return: shape = (batch, f_dim1, f_dim2, f_dim3, out_channels)
        """
        output = super().call(inputs=inputs, training=training, mask=mask)
        # upsample again
        output = self._out_ddf_upsampling(output)
        output = tf.concat([inputs, output], axis=4)
        output = self._out_ddf_conv(output)
        return output

    def get_activation(self) -> tf.keras.layers.Layer:
        """Return activation layer."""
        return tf.keras.layers.LeakyReLU(alpha=0.2)


@REGISTRY.register_loss(name="gradient-vm")
class GradientNorm(tf.keras.layers.Layer):
    """
    Calculate the L1/L2 norm of ddf using central finite difference.

    y_true and y_pred have to be at least 5d tensor, including batch axis.
    """

    def __init__(self, l1: bool = False, name: str = "GradientNorm"):
        """
        Init.

        :param l1: bool true if calculate L1 norm, otherwise L2 norm
        :param name: name of the loss
        """
        super().__init__(name=name)
        self.l1 = l1

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Return a scalar loss.

        :param inputs: shape = (batch, m_dim1, m_dim2, m_dim3, 3)
        :param kwargs: additional arguments.
        :return: shape = ()
        """
        assert len(inputs.shape) == 5
        tf.debugging.check_numerics(inputs, "GRAIDENT ddf value NAN/INF", name=None)
        ddf = inputs

        if self.l1:
            df = [tf.reduce_mean(tf.abs(f)) for f in self._diffs(ddf)]
        else:
            df = [tf.reduce_mean(f * f) for f in self._diffs(ddf)]
        return tf.add_n(df) / len(df)

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config["l1"] = self.l1
        return config

    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)

        df = []
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = K.permute_dimensions(y, r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df.append(K.permute_dimensions(dfi, r))

        return df


def main(args=None):
    """
    Launch training.

    Args:
        args:

    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu",
        "-g",
        help="GPU index for training."
        '-g "" for using CPU'
        '-g "0" for using GPU 0'
        '-g "0,1" for using GPU 0 and 1.',
        type=str,
        required=True,
    )
    args = parser.parse_args(args)

    config_path = "config_balakrishnan_2019.yaml"
    train(
        gpu=args.gpu,
        config_path=config_path,
        gpu_allow_growth=True,
        ckpt_path="",
    )


if __name__ == "__main__":
    main()
