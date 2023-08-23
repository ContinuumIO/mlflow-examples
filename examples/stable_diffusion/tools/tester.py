import uuid
from typing import List

import keras
import keras_cv
import numpy
from keras_cv.models.stable_diffusion.stable_diffusion import StableDiffusion
from PIL import Image


def tester(prompt: str, image_width: int = 512, image_height: int = 512, batch_size: int = 3):
    keras.mixed_precision.set_global_policy("mixed_float16")

    model: StableDiffusion = keras_cv.models.StableDiffusion(
        img_width=image_width, img_height=image_height, jit_compile=True
    )

    arrays: List[numpy.ndarray] = model.text_to_image(prompt, batch_size=batch_size)
    for array in arrays:
        image: Image = Image.fromarray(array)
        filename: str = f"{str(uuid.uuid4())}.png"
        image.save(filename)


if __name__ == "__main__":
    input_prompt: str = "some input prompt"
    tester(prompt=input_prompt)
