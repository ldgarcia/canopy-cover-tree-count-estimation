import numpy as np


def standard_centroid(binary_mask: np.ndarray) -> np.ndarray:
    """Compute the standard centroid of a rasterized polygon."""
    # Assumes a binary mask of a polygon rasterized in its bounding box.
    # See: https://en.wikipedia.org/wiki/Centroid#Of_a_finite_set_of_points
    idx = np.argwhere(binary_mask)
    if len(idx) > 0:
        return np.round(np.sum(idx, axis=0) / len(idx)).astype(int)
    return np.array(list(binary_mask.shape)) // 2


def energy_centroid(energy_map: np.ndarray) -> np.ndarray:
    """Compute the energy centroid of a rasterized polygon."""
    # Assumes a polygon energy map computed on the whole polygon.
    idx = np.argwhere(np.logical_not(energy_map < np.max(energy_map)))
    if len(idx) > 0:
        return np.round(np.sum(idx, axis=0) / len(idx)).astype(int)
    return np.array(list(energy_map.shape)) // 2


def centroid_mask(
    polygon_mask: np.ndarray,
    centroid: np.ndarray,
) -> np.ndarray:
    mask = np.zeros_like(polygon_mask)
    mask[centroid[0], centroid[1]] = 1.0
    return mask


def get_segment_distance(
    start: np.ndarray,
    binary_mask: np.ndarray,
    direction: np.ndarray,
) -> float:
    """Get the distance to the background from the starting point."""
    current = start.copy()
    while True:
        new = current + direction
        # Check not out-of-bounds
        if 0 < new[0] < binary_mask.shape[0] and 0 < new[1] < binary_mask.shape[1]:
            # Check still inside blob/foreground
            if binary_mask[new[0], new[1]] > 0.0:
                current = new
            else:
                break
        else:
            break
    # We use the Euclidean distance.
    return np.linalg.norm(new - start)


def get_min_segment_distance(
    start: np.ndarray,
    blob_mask: np.ndarray,
) -> float:
    """Get the minimum distance to the background from the starting point."""
    # We consider the minimum distance going in 8 possible directions.
    sds = []
    sds.append(get_segment_distance(start, blob_mask, [0, -1]))
    sds.append(get_segment_distance(start, blob_mask, [0, 1]))
    sds.append(get_segment_distance(start, blob_mask, [-1, 0]))
    sds.append(get_segment_distance(start, blob_mask, [1, 0]))
    sds.append(get_segment_distance(start, blob_mask, [1, 1]))
    sds.append(get_segment_distance(start, blob_mask, [1, -1]))
    sds.append(get_segment_distance(start, blob_mask, [-1, -1]))
    sds.append(get_segment_distance(start, blob_mask, [-1, 1]))
    return np.min(sds)
