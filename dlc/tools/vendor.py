"""This module includes some vendored patches or implementations from \
    external sources."""
import geopandas as gpd
import tensorflow as tf
from osgeo import gdal
from osgeo import gdalconst
from shapely.geometry import box


def new_py_function(func, inp, Tout, name=None):
    # Taken from:
    # https://github.com/tensorflow/tensorflow/issues/27679#issuecomment-522578000
    # See also:
    # https://github.com/tensorflow/tensorflow/issues/36278#issuecomment-781484858
    def wrapped_func(*flat_inp):
        reconstructed_inp = tf.nest.pack_sequence_as(
            inp, flat_inp, expand_composites=True
        )
        out = func(*reconstructed_inp)
        return tf.nest.flatten(out, expand_composites=True)

    flat_Tout = tf.nest.flatten(Tout, expand_composites=True)
    flat_out = tf.py_function(
        func=wrapped_func,
        inp=tf.nest.flatten(inp, expand_composites=True),
        Tout=[_tensor_spec_to_dtype(v) for v in flat_Tout],
        name=name,
    )
    spec_out = tf.nest.map_structure(
        _dtype_to_tensor_spec, Tout, expand_composites=True
    )
    out = tf.nest.pack_sequence_as(spec_out, flat_out, expand_composites=True)
    return out


def _dtype_to_tensor_spec(v):
    # Taken from:
    # https://github.com/tensorflow/tensorflow/issues/27679#issuecomment-522578000
    return tf.TensorSpec(None, v) if isinstance(v, tf.dtypes.DType) else v


def _tensor_spec_to_dtype(v):
    # Taken from:
    # https://github.com/tensorflow/tensorflow/issues/27679#issuecomment-522578000
    return v.dtype if isinstance(v, tf.TensorSpec) else v


# Taken from:
# Reiner, F., Li, S., & Kariryaa, A. (2021).
# Deep learning using unet based architectures for individualtree detection
# in planet images [Code Repository]
def gdal_progress_callback(complete, message, data):
    """Callback function to show progress during GDAL operations such gdal.Warp() or gdal.Translate().

    Expects a tqdm progressbar in 'data', which is passed as the 'callback_data' argument of the GDAL method.
    'complete' is passed by the GDAL methods, as a float from 0 to 1
    """
    if data:
        data.update(int(complete * 100) - data.n)
        if complete == 1:
            data.close()
    return 1


def raster_copy(
    output_fp,
    input_fp,
    mode="warp",
    resample=1,
    out_crs=None,
    bands=None,
    bounds=None,
    bounds_crs=None,
    multi_core=False,
    pbar=None,
):
    """Copy a raster using GDAL Warp or GDAL Translate, with various options.

    The use of Warp or Translate can be chosen with 'mode' parameter. GDAL.Warp allows full multiprocessing,
    whereas GDAL.Translate allows the selection of only certain bands to copy.
    A specific window to copy can be specified with 'bounds' and 'bounds_crs' parameters.
    Optional resampling with bi-linear interpolation is done if passed in as 'resample'!=1.
    """

    # Common options
    base_options = dict(
        creationOptions=[
            "TILED=YES",
            "BLOCKXSIZE=256",
            "BLOCKYSIZE=256",
            "BIGTIFF=IF_SAFER",
            "NUM_THREADS=ALL_CPUS",
        ],
        callback=gdal_progress_callback,
        callback_data=pbar,
    )
    if resample != 1:
        # Get input pixel sizes
        raster = gdal.Open(input_fp)
        gt = raster.GetGeoTransform()
        x_res, y_res = gt[1], -gt[5]
        base_options["xRes"] = (x_res / resample,)
        base_options["yRes"] = (y_res / resample,)
        base_options["resampleAlg"] = gdalconst.GRA_Bilinear

    # Use GDAL Warp
    if mode.lower() == "warp":
        warp_options = dict(
            dstSRS=out_crs,
            outputBounds=bounds,
            outputBoundsSRS=bounds_crs,
            multithread=multi_core,
            warpOptions=["NUM_THREADS=ALL_CPUS"] if multi_core else [],
            warpMemoryLimit=1000000000,  # processing chunk size. higher is not always better, around 1-4GB seems good
        )
        return gdal.Warp(output_fp, input_fp, **base_options, **warp_options)

    # Use GDAL Translate
    elif mode.lower() == "translate":
        translate_options = dict(
            bandList=bands,
            outputSRS=out_crs,
            projWin=[
                bounds[0],
                bounds[3],
                bounds[2],
                bounds[1],
            ],  # what the hell gdal.Translate....
            projWinSRS=bounds_crs,
        )
        return gdal.Translate(output_fp, input_fp, **base_options, **translate_options)

    else:
        raise Exception(
            "Invalid mode argument, supported modes are 'warp' or 'translate'."
        )


# Taken from:
# Reiner, F., Li, S., & Kariryaa, A. (2021).
# Deep learning using unet based architectures for individualtree detection
# in planet images [Code Repository]
def calculate_boundary_weights(polygons, scale):
    """Find boundaries between close polygons.

    Scales up each polygon, then get overlaps by intersecting. The overlaps of the scaled polygons are the boundaries.
    Returns geopandas data frame with boundary polygons.
    """
    # Scale up all polygons around their center, until they start overlapping
    # NOTE: scale factor should be matched to resolution and type of forest
    scaled_polys = gpd.GeoDataFrame(
        {
            "geometry": polygons.geometry.scale(
                xfact=scale, yfact=scale, origin="center"
            )
        },
        crs=polygons.crs,
    )

    # Get intersections of scaled polygons, which are the boundaries.
    boundaries = []
    for i in range(len(scaled_polys)):

        # For each scaled polygon, get all nearby scaled polygons that intersect with it
        nearby_polys = scaled_polys[
            scaled_polys.geometry.intersects(scaled_polys.iloc[i].geometry)
        ]

        # Add intersections of scaled polygon with nearby polygons [except the intersection with itself!]
        for j in range(len(nearby_polys)):
            if nearby_polys.iloc[j].name != scaled_polys.iloc[i].name:
                boundaries.append(
                    scaled_polys.iloc[i].geometry.intersection(
                        nearby_polys.iloc[j].geometry
                    )
                )

    # Convert to df and ensure we only return Polygons (sometimes it can be a Point, which breaks things)
    boundaries = gpd.GeoDataFrame(
        {"geometry": gpd.GeoSeries(boundaries)},
        crs=polygons.crs,
    ).explode()
    boundaries = boundaries[boundaries.type == "Polygon"]

    # If we have boundaries, difference overlay them with original polygons to ensure boundaries don't cover labels
    if len(boundaries) > 0:
        boundaries = gpd.overlay(boundaries, polygons, how="difference")
    else:
        boundaries = boundaries.append({"geometry": box(0, 0, 0, 0)}, ignore_index=True)

    return boundaries


# Taken from:
# Reiner, F., Li, S., & Kariryaa, A. (2021).
# Deep learning using unet based architectures for individualtree detection
# in planet images [Code Repository]
def image_normalize(im, axis=(0, 1), c=1e-8):
    """Normalize to zero mean and unit standard deviation along the given axis"""
    return (im - im.mean(axis)) / (im.std(axis) + c)
