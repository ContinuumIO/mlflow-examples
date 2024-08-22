""" Report Generation Methods"""

import io
from typing import Any

import graphviz
import mlflow
import numpy as np
import PIL
from graphviz import Source
from numpy import ndarray
from PIL import Image
from sklearn.tree import export_graphviz

# our images get large, disable size check in PIL
PIL.Image.MAX_IMAGE_PIXELS = None


def visualize_tree(tree: Any, name: str, feature_names: list[str], export: bool = True):
    """
    Decision Tree Visualization [and logging]

    Parameters
    ----------
    tree: Any
        A decision tree.
    name: str
        The name of the tree.
    feature_names: list[str]
        The list of feature names to graph.
    export: bool
        Default: True
        Control flag for logging the model to the MLflow run.

    Returns
    -------
    graph: Source
        The rendered graph as a `Source` object.
    """

    dot_data = export_graphviz(tree, out_file=None, feature_names=feature_names, class_names=["No Fraud", "Fraud"], filled=True)
    graph: Source = graphviz.Source(dot_data)
    if export:
        graph.format = "png"
        image_bytes: bytes = graph.pipe()
        image: Image = Image.open(io.BytesIO(image_bytes))
        image_array: ndarray = np.asarray(image)
        mlflow.log_image(image=image_array, artifact_file=f"graphs/{name}.png")
    return graph
