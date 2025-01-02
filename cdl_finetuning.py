import xarray
from pyproj import Transformer
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset

import presto
import os

import rioxarray
from shapely.geometry import box

# this is to silence the xarray deprecation warning.
# Our version of xarray is pinned, but we'll need to fix this
# when we upgrade
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# CDL label
cdl_label_file = '/projects/dali/yichia3/harmonized_global_crops/cdl_harmonized_block/2023_30m_cdls.tif'

# Sentinel2 data
train_directory = "/projects/dali/yichia3/harmonized_global_crops/sentinel2_subsample_1000/sentinel2_cdl_2023_subsampled"
test_directory = "/projects/dali/yichia3/harmonized_global_crops/sentinel2_subsample_test/sentinel2_cdl_2023_subsampled"

def get_tif_files(directory):
    all_tif_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".tif") and os.path.isfile(os.path.join(directory, f))
    ]
    return all_tif_files

train_files = get_tif_files(train_directory)
test_files = get_tif_files(test_directory)

# INDICES_IN_TIF_FILE = list(range(0, 6, 2))
INDICES_IN_TIF_FILE = list(range(122, 134, 2))
# INDICES_IN_TIF_FILE = list(range(100, 200, 2))
CDL_S2_BANDS = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"]


def clip_to_intersection(img_path, label_path):
    """
    Clips the second GeoTIFF file to the intersection of its bounds with the first GeoTIFF file.

    Parameters:
        img_path (str): Path to the first GeoTIFF file.
        label_path (str): Path to the second GeoTIFF file.

    Returns:
        xarray.DataArray: The clipped version of the second GeoTIFF file.
    """
    img = rioxarray.open_rasterio(img_path)
    label = rioxarray.open_rasterio(label_path)
    img_bounds = box(*img.rio.bounds())
    label_bounds = box(*label.rio.bounds())

    intersection = img_bounds.intersection(label_bounds)

    if not intersection.is_empty:
        minx, miny, maxx, maxy = intersection.bounds
        clipped_label = label.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
        return clipped_label
    else:
        raise ValueError("No intersection found between the GeoTIFF files.")


def process_images(filenames):
    arrays, masks, latlons, image_names, labels, dynamic_worlds = [], [], [], [], [], []

    for filename in tqdm(filenames):
        tif_file = xarray.open_rasterio(filename)
        crs = tif_file.crs.split("=")[-1]
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        clipped_label = clip_to_intersection(filename, cdl_label_file)

        for x_idx in INDICES_IN_TIF_FILE:
            for y_idx in INDICES_IN_TIF_FILE:

                x, y = tif_file.x[x_idx], tif_file.y[y_idx]
                lon, lat = transformer.transform(x, y)
                latlons.append(torch.tensor([lat, lon]))

                # then, get the eo_data, mask and dynamic world
                s2_data_for_pixel = torch.from_numpy(tif_file.values[:, x_idx, y_idx].astype(int)).float()
                s2_data_with_time_dimension = s2_data_for_pixel.unsqueeze(0)
                x, mask, dynamic_world = presto.construct_single_presto_input(
                    s2=s2_data_with_time_dimension, s2_bands=CDL_S2_BANDS
                )
                label = clipped_label.values[:, x_idx, y_idx].astype(int).item()

                arrays.append(x)
                masks.append(mask)
                dynamic_worlds.append(dynamic_world)
                labels.append(label)
                # image_names.append(filename)
    return (torch.stack(arrays, axis=0),
            torch.stack(masks, axis=0),
            torch.stack(dynamic_worlds, axis=0),
            torch.stack(latlons, axis=0),
            torch.tensor(labels),
        )

train_data = process_images(train_files)
test_data = process_images(test_files)

batch_size = 64
pretrained_model = presto.Presto.load_pretrained()
pretrained_model.eval()

# the CDL data was collected during the summer,
# so we estimate the month to be 8 (August)

month = torch.tensor([8] * train_data[0].shape[0]).long()

dl = DataLoader(
    TensorDataset(
        train_data[0].float(),  # x
        train_data[1].bool(),  # mask
        train_data[2].long(),  # dynamic world
        train_data[3].float(),  # latlons
        month
    ),
    batch_size=batch_size,
    shuffle=False,
)

features_list = []
for (x, mask, dw, latlons, month) in tqdm(dl):
    with torch.no_grad():
        encodings = (
            pretrained_model.encoder(
                x, dynamic_world=dw, mask=mask, latlons=latlons, month=month
            )
            .cpu()
            .numpy()
        )
        features_list.append(encodings)
features_np = np.concatenate(features_list)

model = RandomForestClassifier(class_weight="balanced", random_state=42)
model.fit(features_np, train_data[4].numpy())

# the CDL data was collected during the summer,
# so we estimate the month to be 8 (August)
month = torch.tensor([8] * test_data[0].shape[0]).long()

dl = DataLoader(
    TensorDataset(
        test_data[0].float(),  # x
        test_data[1].bool(),  # mask
        test_data[2].long(),  # dynamic world
        test_data[3].float(),  # latlons
        month
    ),
    batch_size=batch_size,
    shuffle=False,
)

test_preds = []
for (x, mask, dw, latlons, month) in tqdm(dl):
    with torch.no_grad():
        pretrained_model.eval()
        encodings = (pretrained_model.encoder(
            x, dynamic_world=dw, mask=mask, latlons=latlons, month=month)
            .cpu()
            .numpy()
        )
        test_preds.append(model.predict_proba(encodings))

pix_per_image = len(INDICES_IN_TIF_FILE) ** 2

test_preds_np = np.concatenate(test_preds, axis=0)
test_preds_np = np.reshape(
    test_preds_np,
    (int(len(test_preds_np) / pix_per_image), pix_per_image, test_preds_np.shape[-1]),
)
# then, take the mode of the model predictions
test_preds_np_argmax = stats.mode(
    np.argmax(test_preds_np, axis=-1), axis=1, keepdims=False
)[0]

target = np.reshape(test_data[4], (int(len(test_data[4]) / pix_per_image), pix_per_image))[:, 0]
target = target.cpu().numpy()

f1_score(target, test_preds_np_argmax, average="weighted")

accuracy = np.mean(test_preds_np_argmax == target)
print(f"Overall Accuracy: {accuracy:.4f}")