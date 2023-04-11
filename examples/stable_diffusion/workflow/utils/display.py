""" Gallery Helper Functions """

from typing import List

import matplotlib.pyplot as plt
from PIL import Image


def plot_images(images: List[Image]) -> None:
    """
    Simple galley for jupyter notebook display.

    Parameters
    ----------
    images: List[Image]
        List of images to display.
    """

    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
