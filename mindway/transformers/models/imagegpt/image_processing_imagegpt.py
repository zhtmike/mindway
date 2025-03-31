"""Image processor class for ImageGPT."""

from typing import Dict, List, Optional, Union

import numpy as np
import mindspore as ms

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import rescale, resize, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import TensorType, filter_out_non_signature_kwargs, is_vision_available, logging


if is_vision_available():
    import PIL


logger = logging.get_logger(__name__)


def squared_euclidean_distance(a, b):
    b = b.T
    a2 = np.sum(np.square(a), axis=1)
    b2 = np.sum(np.square(b), axis=0)
    ab = np.matmul(a, b)
    d = a2[:, None] - 2 * ab + b2[None, :]
    return d


def color_quantize(x, clusters):
    x = x.reshape(-1, 3)
    d = squared_euclidean_distance(x, clusters)
    return np.argmin(d, axis=1)


class ImageGPTImageProcessor(BaseImageProcessor):
    r"""
    Constructs a ImageGPT image processor. This image processor can be used to resize images to a smaller resolution
    (such as 32x32 or 64x64), normalize them and finally color quantize them to obtain sequences of "pixel values"
    (color clusters).

    Args:
        clusters (`np.ndarray` or `List[List[int]]`, *optional*):
            The color clusters to use, of shape `(n_clusters, 3)` when color quantizing. Can be overriden by `clusters`
            in `preprocess`.
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's dimensions to `(size["height"], size["width"])`. Can be overridden by
            `do_resize` in `preprocess`.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 256, "width": 256}`):
            Size of the image after resizing. Can be overridden by `size` in `preprocess`.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in `preprocess`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image pixel value to between [-1, 1]. Can be overridden by `do_normalize` in
            `preprocess`.
        do_color_quantize (`bool`, *optional*, defaults to `True`):
            Whether to color quantize the image. Can be overridden by `do_color_quantize` in `preprocess`.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        # clusters is a first argument to maintain backwards compatibility with the old ImageGPTImageProcessor
        clusters: Optional[Union[List[List[int]], np.ndarray]] = None,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_normalize: bool = True,
        do_color_quantize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 256, "width": 256}
        size = get_size_dict(size)
        self.clusters = np.array(clusters) if clusters is not None else None
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_normalize = do_normalize
        self.do_color_quantize = do_color_quantize

    # Copied from transformers.models.vit.image_processing_vit.ViTImageProcessor.resize
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        Returns:
            `np.ndarray`: The resized image.
        """
        size = get_size_dict(size)
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        output_size = (size["height"], size["width"])
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def normalize(
        self,
        image: np.ndarray,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Normalizes an images' pixel values to between [-1, 1].

        Args:
            image (`np.ndarray`):
                Image to normalize.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        image = rescale(image=image, scale=1 / 127.5, data_format=data_format, input_data_format=input_data_format)
        image = image - 1
        return image

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_normalize: bool = None,
        do_color_quantize: Optional[bool] = None,
        clusters: Optional[Union[List[List[int]], np.ndarray]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[Union[str, ChannelDimension]] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_normalize=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image
            do_color_quantize (`bool`, *optional*, defaults to `self.do_color_quantize`):
                Whether to color quantize the image.
            clusters (`np.ndarray` or `List[List[int]]`, *optional*, defaults to `self.clusters`):
                Clusters to use for color quantization.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.MINDSPORE` or `'ms'`: Return a batch of tensors.
                - `TensorType.NUMPY` or `'np'`: Return a batch of `np.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model, of shape (batch_size, num_channels, height, width)
            - **input_ids** -- Color cluster indices of each pixel, of shape (batch_size, seq_length).
        """
        validate_preprocess_arguments(
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        # Define values for compatibility
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        do_color_quantize = do_color_quantize if do_color_quantize is not None else self.do_color_quantize
        clusters = clusters if clusters is not None else self.clusters

        # Define values for conversion
        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        # Conversion to numpy arrays
        images = [to_numpy_array(image) for image in images]

        # Check what's the channel dimension format of the input images
        if input_data_format is None:
            # If no input_data_format given, try to infer it from the images
            image_channel_formats = [infer_channel_dimension_format(image) for image in images]
            if len(set(image_channel_formats)) > 1:
                raise ValueError(
                    "Images have different channel dimension formats. Please provide the input_data_format argument."
                )
            input_data_format = image_channel_formats[0]

        # Preprocess the images accordingly
        if do_resize:
            images = [
                self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
                for image in images
            ]

        # (conversion to floats for normalization and quantization)
        input_data_format_for_normalization = input_data_format
        if do_normalize:
            images = [
                self.normalize(image=image, input_data_format=input_data_format_for_normalization) for image in images
            ]

        # Color quantize if needed
        if do_color_quantize:
            if clusters is None:
                raise ValueError(
                    "No clusters provided for color quantization. Please provide clusters to the processor or during preprocessing"
                )
            images = [
                color_quantize(
                    to_channel_dimension_format(
                        image, input_data_format=input_data_format, output_data_format=ChannelDimension.LAST
                    ),
                    clusters,
                )
                for image in images
            ]

            # For clusters we set a new input format!
            input_data_format = ChannelDimension.NONE

        # Convert to the output format. Always output 3-dim raw pixels (no quantization).
        # If color quantized, these will be 1-dim arrays of indices.
        if do_color_quantize:
            # We have 1-dim arrays coming from color_quantize, so no format adjustments
            # If 1-dim integers, we put them into a dict under "input_ids"
            batch_input_ids = np.stack([np.array(image) for image in images])
            data = {"input_ids": batch_input_ids}
        else:
            images = [
                to_channel_dimension_format(
                    image, input_data_format=input_data_format, output_data_format=data_format
                )
                for image in images
            ]
            data = {"pixel_values": images}

        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)
        return encoded_inputs 