""" Worker Helper Functions """

from typing import List


def get_batches(batch_size: int, source_list: List[str]) -> List[List[str]]:
    """
    Given a batch size and a list of strings, will generate a list of lists of strings split by batch size.

    Parameters
    ----------
    batch_size: int
        The maximum size a batch can be.  It is possible to receive a batch that is smaller than this.
        This value must be greater than zero.
    source_list: List[str]
        A list of strings to split into batches.

    Returns
    -------
    batches: List[List[str]]
        The source list split up by batch size into smaller lists and returned together.
    """

    if batch_size < 1:
        raise ValueError(f"Batch size must be greater than zero.  Saw: ({batch_size})")

    batches: List = []

    while len(source_list) > 0:
        if len(source_list) >= batch_size:
            new_batch: List[str] = source_list[:batch_size]
            source_list[:batch_size] = []
        else:
            new_batch: List[str] = source_list
            source_list = []
        batches.append(new_batch)

    return batches
