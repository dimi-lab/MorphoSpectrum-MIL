import time
import sys
import cv2
import h5py
import numpy as np
import openslide
import torch
from PIL import ImageDraw
from shapely.affinity import scale
from shapely.geometry import Polygon
from shapely.ops import unary_union
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def segment_tissue(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mthresh = 7
    img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)
    _, img_prepped = cv2.threshold(img_med, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    close = 4
    kernel = np.ones((close, close), np.uint8)
    img_prepped = cv2.morphologyEx(img_prepped, cv2.MORPH_CLOSE, kernel)

    # Find and filter contours
    contours, hierarchy = cv2.findContours(
        img_prepped, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    return contours, hierarchy


def detect_foreground(contours, hierarchy):
    hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

    # find foreground contours (parent == -1)
    hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
    foreground_contours = [contours[cont_idx] for cont_idx in hierarchy_1]

    all_holes = []
    for cont_idx in hierarchy_1:
        all_holes.append(np.flatnonzero(hierarchy[:, 1] == cont_idx))

    hole_contours = []
    for hole_ids in all_holes:
        holes = [contours[idx] for idx in hole_ids]
        hole_contours.append(holes)

    return foreground_contours, hole_contours


def construct_polygon(foreground_contours, hole_contours, min_area):
    polys = []
    for foreground, holes in zip(foreground_contours, hole_contours):
        # We remove all contours that consist of fewer than 3 points, as these won't work with the Polygon constructor.
        if len(foreground) < 3:
            continue

        # remove redundant dimensions from the contour and convert to Shapely Polygon
        poly = Polygon(np.squeeze(foreground))

        # discard all polygons that are considered too small
        if poly.area < min_area:
            continue

        if not poly.is_valid:
            # This is likely becausee the polygon is self-touching or self-crossing.
            # Try and 'correct' the polygon using the zero-length buffer() trick.
            # See https://shapely.readthedocs.io/en/stable/manual.html#object.buffer
            poly = poly.buffer(0)

        # Punch the holes in the polygon
        for hole_contour in holes:
            if len(hole_contour) < 3:
                continue

            hole = Polygon(np.squeeze(hole_contour))

            if not hole.is_valid:
                continue

            # ignore all very small holes
            if hole.area < min_area:
                continue

            poly = poly.difference(hole)

        polys.append(poly)

    if len(polys) == 0:
        raise Exception("Raw tissue mask consists of 0 polygons")

    # If we have multiple polygons, we merge any overlap between them using unary_union().
    # This will result in a Polygon or MultiPolygon with most tissue masks.
    return unary_union(polys)


def generate_tiles(
    tile_width_pix, tile_height_pix, img_width, img_height, offsets=[(0, 0)]
):
    # Generate tiles covering the entire image.
    # Provide an offset (x,y) to create a stride-like overlap effect.
    # Add an additional tile size to the range stop to prevent tiles being cut off at the edges.
    range_stop_width = int(np.ceil(img_width + tile_width_pix))
    range_stop_height = int(np.ceil(img_height + tile_height_pix))

    rects = []
    for xmin, ymin in offsets:
        cols = range(int(np.floor(xmin)), range_stop_width, tile_width_pix)
        rows = range(int(np.floor(ymin)), range_stop_height, tile_height_pix)
        for x in cols:
            for y in rows:
                rect = Polygon(
                    [
                        (x, y),
                        (x + tile_width_pix, y),
                        (x + tile_width_pix, y - tile_height_pix),
                        (x, y - tile_height_pix),
                    ]
                )
                rects.append(rect)
    return rects


def make_tile_QC_fig(tiles, slide, level, line_width_pix=1, extra_tiles=None):
    # Render the tiles on an image derived from the specified zoom level
    img = slide.read_region((0, 0), level, slide.level_dimensions[level])
    downsample = 1 / slide.level_downsamples[level]

    draw = ImageDraw.Draw(img, "RGBA")
    for tile in tiles:
        bbox = tuple(np.array(tile['rect'].bounds) * downsample)
        draw.rectangle(bbox, outline="lightgreen", width=line_width_pix)

    # allow to display other tiles, such as excluded or sampled
    if extra_tiles:
        for tile in extra_tiles:
            bbox = tuple(np.array(tile.bounds) * downsample)
            draw.rectangle(bbox, outline="blue", width=line_width_pix + 1)

    return img


def create_tissue_mask(wsi, seg_level):
    # Determine the best level to determine the segmentation on
    level_dims = wsi.level_dimensions[seg_level]

    #check purpose
    region_pil = wsi.read_region((0, 0), seg_level, level_dims)
    print("PIL image mode:", region_pil.mode)   
    print("PIL image size:", region_pil.size)   
    print("PIL image format:", region_pil.format)  

    img = np.array(wsi.read_region((0, 0), seg_level, level_dims))
    print("NumPy array shape:", img.shape)  
    print("NumPy array dtype:", img.dtype)  


    # Get the total surface area of the slide level that was used
    level_area = level_dims[0] * level_dims[1]

    # Minimum surface area of tissue polygons (in pixels)
    # Note that this value should be sensible in the context of the chosen tile size
    min_area = level_area / 500

    contours, hierarchy = segment_tissue(img)
    foreground_contours, hole_contours = detect_foreground(contours, hierarchy)
    tissue_mask = construct_polygon(foreground_contours, hole_contours, min_area)

    # Scale the tissue mask polygon to be in the coordinate space of the slide's level 0
    scale_factor = wsi.level_downsamples[seg_level]
    tissue_mask_scaled = scale(
        tissue_mask, xfact=scale_factor, yfact=scale_factor, zfact=1.0, origin=(0, 0)
    )

    return tissue_mask_scaled

def compute_mpp_scale_factor(wsi):
    assert (
        openslide.PROPERTY_NAME_MPP_X in wsi.properties
    ), "microns per pixel along X-dimension not available"
    assert (
        openslide.PROPERTY_NAME_MPP_Y in wsi.properties
    ), "microns per pixel along Y-dimension not available"

    mpp_x = float(wsi.properties[openslide.PROPERTY_NAME_MPP_X])
    print(f"mpp_x equals {mpp_x}")
    mpp_y = float(wsi.properties[openslide.PROPERTY_NAME_MPP_Y])
    print(f"mpp_y equals {mpp_y}")
    mpp_scale_factor = min(mpp_x, mpp_y)
    print(f"mpp_scale_factor equals {mpp_scale_factor}")
    if mpp_x != mpp_y:
        print(
            f"mpp_x of {mpp_x} and mpp_y of {mpp_y} are not the same. Using smallest value: {mpp_scale_factor}"
        )
    return mpp_scale_factor



def create_tissue_tiles(
    mpp_scale_factor, wsi, tissue_mask_scaled, tile_size_microns, offsets_micron=None
):

    print(f"tile size is {tile_size_microns} um")
    # Compute the tile size in pixels from the desired tile size in microns and the image resolution
    tile_size_pix = round(tile_size_microns / mpp_scale_factor)

    # Use the tissue mask bounds as base offsets (+ a margin of a few tiles) to avoid wasting CPU power creating tiles that are never going
    # to be inside the tissue mask.
    tissue_margin_pix = tile_size_pix * 2
    minx, miny, maxx, maxy = tissue_mask_scaled.bounds
    min_offset_x = minx - tissue_margin_pix
    min_offset_y = miny - tissue_margin_pix
    offsets = [(min_offset_x, min_offset_y)]

    if offsets_micron is not None:
        assert (
            len(offsets_micron) > 0
        ), "offsets_micron needs to contain at least one value"
        # Compute the offsets in micron scale
        offset_pix = [round(o / mpp_scale_factor) for o in offsets_micron]
        offsets = [(o + min_offset_x, o + min_offset_y) for o in offset_pix]

    # Generate tiles covering the entire WSI
    all_tiles = generate_tiles(
        tile_size_pix,
        tile_size_pix,
        maxx + tissue_margin_pix,
        maxy + tissue_margin_pix,
        offsets=offsets,
    )

    # Filter tiles and record their ID and coordinates
    tile_id=0
    filtered_tiles=[]
    for rect in all_tiles:
        if tile_is_not_empty(crop_rect_from_slide(wsi, rect),threshold_white=50):
            filtered_tiles.append(
                {
                'tile_id':f"{tile_id}",
                'rect': rect
                }
            )
        tile_id+=1

    return filtered_tiles


def tile_is_not_empty(tile, threshold_white):
    histogram = tile.histogram()

    # Take the median of each RGB channel. Alpha channel is not of interest.
    # If roughly each chanel median is below a threshold, i.e close to 0 till color value around 250 (white reference) then tile mostly white.
    whiteness_check = [0, 0, 0]
    for channel_id in (0, 1, 2):
        whiteness_check[channel_id] = np.median(
            histogram[256 * channel_id : 256 * (channel_id + 1)][100:200]
        )

    if all(c <= threshold_white for c in whiteness_check):
        # exclude tile
        return False

    # keep tile
    return True


def crop_rect_from_slide(slide, rect):
    minx, miny, maxx, maxy = rect.bounds
    # Note that the y-axis is flipped in the slide: the top of the shapely polygon is y = ymax,
    # but in the slide it is y = 0. Hence: miny instead of maxy.
    top_left_coords = (int(minx), int(miny))
    return slide.read_region(top_left_coords, 0, (int(maxx - minx), int(maxy - miny)))


class BagOfTiles(Dataset):
    def __init__(self, wsi, tiles, resize_to=224,preprocess=None):
        self.wsi = wsi
        self.tiles = tiles
        self.preprocess = preprocess

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]
        img = crop_rect_from_slide(self.wsi, tile['rect'])

        # RGB filtering - calling here speeds up computation since it requires crop_rect_from_slide function.
        is_tile_kept = tile_is_not_empty(img, threshold_white=50)

        # Ensure the img is RGB, as expected by the pretrained model.
        # See https://pytorch.org/docs/stable/torchvision/models.html
        img = img.convert("RGB")

        # Ensure we have a square tile in our hands.
        # We can't handle non-squares currently, as this would requiring changes to
        # the aspect ratio when resizing.
        width, height = img.size
        assert width == height, "input image is not a square"

        img = self.preprocess(img).unsqueeze(0)
        coord = tile['rect'].bounds
        tile_id = tile['tile_id']
        return img, coord, is_tile_kept, tile_id


def collate_features(batch):
    # Item 2 is the boolean value from tile filtering.
    filtered_items = [item for item in batch if item[2]]
    if len(filtered_items)==0:
        return [torch.empty(0), np.empty((0,4)), np.empty((0,))]
    img = torch.cat([item[0] for item in batch if item[2]], dim=0)
    coords = np.vstack([item[1] for item in batch if item[2]])
    tile_ids = np.array([item[3] for item in batch if item[2]],dtype=object)
    return [img, coords, tile_ids]


def write_to_h5(file, asset_dict):
    for key, val in asset_dict.items():
        if val.dtype.kind in ('U', 'O'):  
            ds_type = h5py.vlen_dtype(str)
        else:
            ds_type = val.dtype

        if key not in file:
            maxshape = (None,) + val.shape[1:]
            dset = file.create_dataset(
                key, shape=val.shape, maxshape=maxshape, dtype=ds_type
            )
            dset[:] = val
        else:
            dset = file[key]
            dset.resize(len(dset) + val.shape[0], axis=0)
            dset[-val.shape[0] :] = val


def load_encoder(device):
    from conch.open_clip_custom import create_model_from_pretrained

    model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token="")
    model = model.to(device)
    model.eval()

    return model,preprocess



def extract_features(model, preprocess, device, wsi, filtered_tiles, workers, out_size, batch_size):
    # Use multiple workers if running on the GPU, otherwise we'll need all workers for
    # evaluating the model.
    kwargs = (
        {"num_workers": workers, "pin_memory": True} if device.type == "cuda" else {}
    )
    loader = DataLoader(
        dataset=BagOfTiles(wsi, filtered_tiles, resize_to=out_size, preprocess=preprocess),
        batch_size=batch_size,
        collate_fn=collate_features,
        **kwargs,
    )
    with torch.no_grad():
        for batch, coords, tile_ids in loader:
            if batch.shape[0] == 0:
                continue
            batch = batch.to(device, non_blocking=True)
            with torch.inference_mode():
                features = model.encode_image(batch, proj_contrast=False, normalize=False).cpu().numpy()
                yield features, coords, tile_ids


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Preprocessing script")
    parser.add_argument(
        "--input_slide",
        type=str,
        help="Path to input WSI file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save output data",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--tile_size",
        help="Desired tile size in microns (should be the same value as used in feature extraction model).",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--out_size",
        help="Resize the square tile to this output size (in pixels).",
        type=int,
        default=224,
    )
    parser.add_argument(
        "--workers",
        help="The number of workers to use for the data loader. Only relevant when using a GPU.",
        type=int,
        default=4,
    )
    args = parser.parse_args()

    # Derive the slide ID from its name
    slide_id, _ = os.path.splitext(os.path.basename(args.input_slide))
    wip_file_path = os.path.join(args.output_dir, slide_id + "_wip.h5")
    output_file_path = os.path.join(args.output_dir, slide_id + "_features.h5")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Starts to pre-processing {slide_id}")

    # Check if the _features output file already exist. If so, we terminate to avoid
    # overwriting it by accident. This also simplifies resuming bulk batch jobs.
    if os.path.exists(output_file_path):
        raise Exception(f"{output_file_path} already exists")

    # Open the slide for reading
    wsi = openslide.open_slide(args.input_slide)
    mpp_factor = compute_mpp_scale_factor(wsi)

    # Decide on which slide level we want to base the segmentation
    seg_level = wsi.get_best_level_for_downsample(64)

    print(f"seg_level is {seg_level}")

    # Run the segmentation and  tiling procedure
    start_time = time.time()
    tissue_mask_scaled = create_tissue_mask(wsi, seg_level)
    filtered_tiles = create_tissue_tiles(mpp_factor, wsi, tissue_mask_scaled, args.tile_size)
    
    # Build a figure for quality control purposes, to check if the tiles are where we expect them.
    qc_img = make_tile_QC_fig(filtered_tiles, wsi, seg_level, 2)
    qc_img_target_width = 1920
    qc_img = qc_img.resize(
        (qc_img_target_width, int(qc_img.height / (qc_img.width / qc_img_target_width)))
    )
    print(
        f"Finished creating {len(filtered_tiles)} tissue tiles in {time.time() - start_time}s"
    )

    # Extract the rectangles, and compute the feature vectors
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model,preprocess = load_encoder(
        device=device,
    )

    generator = extract_features(
        model,
        preprocess,
        device,
        wsi,
        filtered_tiles,
        args.workers,
        args.out_size,
        args.batch_size,
    )
    start_time = time.time()
    count_features = 0
    with h5py.File(wip_file_path, "w") as file:
        for i, (features, coords, tile_ids) in enumerate(generator):
            count_features += features.shape[0]
            write_to_h5(file, {"features": features, "coords": coords, "tile_ids": tile_ids})
            print(
                f"Processed batch {i}. Extracted features from {count_features}/{len(filtered_tiles)} tiles in {(time.time() - start_time):.2f}s."
            )

    # Rename the file containing the patches to ensure we can easily
    # distinguish incomplete bags of patches (due to e.g. errors) from complete ones in case a job fails.
    os.rename(wip_file_path, output_file_path)

    # Save QC figure while keeping track of number of features/tiles used since RBG filtering is within DataLoader.
    qc_img_file_path = os.path.join(
        args.output_dir, f"{slide_id}_{count_features}_features_QC.png"
    )
    qc_img.save(qc_img_file_path)
    print(
        f"Finished processing {slide_id} extracting {count_features} features in {(time.time() - start_time):.2f}s"
    )
