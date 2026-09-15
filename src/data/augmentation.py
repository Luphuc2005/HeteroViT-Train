"""Data augmentation pipeline for CIFAR-10."""
import tensorflow as tf


def get_train_augmentation(image_size: int = 32):
    """Returns training augmentation pipeline.

    Standard CIFAR-10 augmentation:
    1. Pad 4 pixels reflection or symmetric
    2. Random crop back to image_size x image_size
    3. Random horizontal flip
    4. Normalize pixel values to [0, 1]
    """
    def augment(image, label):
        # Convert uint8 to float32 [0, 1] if not already
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Pad by 4 pixels on each border: (32 + 8) = 40x40
        padding = 4
        padded = tf.pad(
            image,
            [[padding, padding], [padding, padding], [0, 0]],
            mode="REFLECT"
        )
        # Random crop back to 32x32x3
        cropped = tf.image.random_crop(padded, size=[image_size, image_size, 3])

        # Random horizontal flip
        flipped = tf.image.random_flip_left_right(cropped)

        return flipped, label

    return augment


def get_val_augmentation(image_size: int = 32):
    """Returns validation/testing normalization pipeline (deterministic)."""
    def normalize(image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Ensure correct shape
        image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
        return image, label

    return normalize
