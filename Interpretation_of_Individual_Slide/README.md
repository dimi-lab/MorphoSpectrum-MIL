# Readme

## Overview of Local Explanation Generation Module

In the **interpretability analysis process**, we constructed a global explanation — the **Morphologic Spectrum** — using high-contribution patch embeddings and their clustering results derived from the training set. This spectrum characterizes the **core** and **transitional** histomorphological patterns learned by the model during training, providing a global view of the model’s decision structure.

Building on this spectrum, the present module generates **slide-level explanations (local explanations)** for predictions made by the **full-data model** on independent test slides. These local explanations **indicate which regions of the slide the model relies on most heavily** and **specify which morphologic-spectrum patterns these regions correspond to**.

The resulting local explanations are exported in **GeoJSON format**, which can be imported into **MorphoExplainer** for interactive visualization directly on the whole-slide image.

The figure below provides a macroscopic view of the workflow within this module. It summarizes how different scripts are connected, the inputs they consume, and the outputs they produce.

![Execution Flow of Local Explanation Generation](Execution_Flow_of_Local_Explanation_Generation.png)

## High-Contribution Patch Extraction for Independent Test Slide

### 1. Purpose of This Script

This script extracts high-contribution patches from independent test slides for *each* prediction class of the full-data model.
 Given the full-data model checkpoint and the CONCH feature bags of the independent slides, the script:

- performs per-class forward inference on each slide,
- computes patch-level attention scores and attention weights,
- selects the top-k high-contribution patches by cumulative contribution (default: top 90%), and
- writes all extracted patches (across all prediction classes) into a unified CSV file.

### 2. Key Inputs Required

This script requires three essential inputs:

#### **1. Independent Test Metadata CSV**

Defined in config.yaml:

```python
paths:
  manifest: "/path/to/independent_svs_file_mapping.csv"
```

Loaded via:

```python
result_df = pd.read_csv(config["paths"]["manifest"])
```

Used to determine:

- the list of slides to process,
- the `SVS Filename` (converted to slide ID),
- the long subtype label → short subtype (`subtypes.long_to_short` mapping).

The CSV must contain the following columns:

- **Person ID** – unique patient identifier
- **Tissue ID** – unique tissue or block identifier
- **Subtype** – diagnostic label for the slide
- **SVS Filename** – filename of the WSI (without or with `.svs` extension)
- **source_folder** – directory where the raw WSI file is located

An example can be found in `./Data_Split/output_svs_file_mapping.csv`.

------

##### **2. Full-Data Model Checkpoint**

Defined in config.yaml:

```python
paths:
  attn_checkpoint: "/path/to/full_data_model_checkpoint.pt"
```

Loaded in:

```python
model = load_trained_model(
    device=DEVICE,
    checkpoint_path=config["paths"]["attn_checkpoint"],
    model_size="small",
    input_feature_size=FEATURE_SIZE,
    n_classes=N_CLASSES,
)
```

This checkpoint is used to compute class-specific attention matrices and contribution scores for each independent slide.

------

##### **3. Feature Bags of Independent Slides (.h5 files)**

Defined in config.yaml:

```python
paths:
  h5_base: "/path/to/feature_bags"
```

Loaded via:

```python
h5_path = f"{h5_base_path}/{slide_id}_features.h5"
test_features, tile_ids = getCONCHFeatures(h5_path)
```

Each feature bag must contain:

- `features` (CONCH embeddings)
- `tile_ids`
- `coords` (patch positions on the WSI)

These are required to score patches for each prediction class and extract top-K high-contribution patches.

### Output of This Script

Running this script produces **two types of outputs** under the directory:

```
./high_contri_path_independent/
```

------

#### **1. Unified CSV: `high_attened_patches_indep_data.csv`**

This file aggregates **all high-contribution patches** extracted from **all independent test slides** and **all prediction classes**.

Each row corresponds to one high-contribution patch and contains the following fields:

| Column       | Description                                                  |
| ------------ | ------------------------------------------------------------ |
| **slide_id** | Slide identifier (WSI ID, without extension)                 |
| **label**    | Ground-truth subtype (short name)                            |
| **pred**     | Prediction class for which this patch is considered high-contribution |
| **tile_id**  | Unique patch ID within the feature bag                       |
| **weight**   | Attention weight                                             |
| **score**    | Logits-based contribution score for the prediction class     |
| **embed**    | Patch-level CONCH embedding                                  |

------

#### **2. Per-Label Subset CSVs (automatically generated)**

For convenience in downstream processing, the unified CSV is automatically **split by ground-truth label** into multiple subtype-specific CSVs:

```
independent_Clear.csv
independent_Endo.csv
independent_High.csv
independent_Border.csv
```



## Spectrum Mapping for Independent Slides

### Purpose of the Script

`spectrum_mapping.py` maps **high-contribution patches from independent test slides** onto the **morphologic spectrum** constructed in the interpretability analysis module.

For each slide and each prediction class:

1. It uses **training-set spectrum statistics** (cluster means, shrunk covariances, r95 thresholds, τ parameters).
2. It performs **Mahalanobis-based soft assignment** of patch embeddings to spectrum clusters.
3. For EC–HGSC pairs, it optionally applies a **hybrid rule with kNN-based transition detection**.
4. It merges all prediction subsets of the same slide into **one patch-level assignment CSV** and, optionally, a **slide×cluster proportion matrix**.

The resulting patch-level assignments are later used to generate MorphoXAI GeoJSON files.

------

### Key Inputs

This script relies on three categories of inputs, all configured in `config.yaml`.

1. **Training-set spectrum inputs (global morphologic spectrum)**
    These are the outputs of the interpretability analysis module and are used to build the consensus spectrum statistics:

   - `paths.train_embeds`
      – CSV files containing **high-contribution patch embeddings** for each subtype (Clear/Endo/High/Border).
      – Required columns: `slide_id`, `tile_id`, `embed` (semicolon-separated CONCH embedding).
   - `paths.train_clusters`
      – Directories with per-slide cluster CSVs (e.g. `<slide>_hc_coords.csv`) for **core clusters**.
      – Each CSV must contain: slide ID, tile ID, cluster name (`hc_*`), and patch coordinates (`x,y,w,h`).
   - `paths.transition_embeds` & `paths.transition_clusters` (optional)
      – Embeddings and cluster CSVs for **transition phenotypes** (Endo_to_High, High_to_Endo).
      – Used to build the kNN bank for EC–HGSC transition detection.

2. **Independent high-contribution patches (test inputs)**
    These are produced by the “High-Contribution Patch Extraction for Independent Test Slides” script.

   - `paths.test_embeds`
      – CSV files such as `independent_Clear.csv`, `independent_Endo.csv`, etc.
      – Required columns:
     - `slide_id`
     - `tile_id`
     - `embed` (semicolon-separated)
     - `pred` (short subtype code: `Clear`, `Endo`, `High`, `Border`)
        – Optional columns:
     - `label` (short true label)
     - `x, y, w, h` (patch coordinates; if present, they are carried through to the outputs)

3. **Slide-level summary of full-data model predictions (`each_slide_result`)**
    This file stores the **full-data model’s evaluation results on the independent test set** and is used to determine:

   - the **true subtype** of each slide, and

   - whether a slide belongs to an EC–HGSC pair that should use the hybrid mapping rule.

   - `paths.each_slide_result`
      – CSV containing slide-level results. Required columns:

     - `slide_id` — slide identifier
     - `name_label` — ground-truth subtype name (long form, e.g. `Ovarian_Endometrioid_Carcinoma`)
     - `name_pred` — predicted subtype name (long form)
     - `correctness` — `True` / `False` indicating whether the prediction is correct

     These long names are mapped to short codes using:

     ```
     subtypes:
       long_to_short:
         Clear_Cell_Carcinoma: "Clear"
         Ovarian_Endometrioid_Carcinoma: "Endo"
         Ovarian_High_Grade_Serous_Carcinoma: "High"
         Ovarian_Serous_Borderline: "Border"
     ```

     The script then builds lookup tables `slide2true`, `slide2pred`, and `slide2corr`.
      The true short label `slide2true[slide_id]` is used to:

     - define the **output subfolder** (grouped by true subtype), and
     - decide whether to apply the **Endo↔High hybrid rule** for spectrum mapping.

     If a slide is not found in `each_slide_result` but the test CSV contains a `label` column, the script falls back to that label as the true subtype.

------

#### Outputs

All outputs are written under the directory specified by:

```
paths:
  output_dir: "./spectrum_assign_knn4"
```

1. **Patch-level assignment CSVs (one per slide)**
    Location:

   ```
   {output_dir}/{true_short}/{slide_id}_patch_assign.csv
   ```

   where `{true_short}` is the true subtype of the slide (e.g. `Clear`, `Endo`, `High`, `Border`).

   Each CSV contains one row per patch with at least:

   - `slide_id`
   - `tile_id`
   - `pred` — prediction short code for which this patch was extracted
   - `assigned_cluster` — final assigned spectrum cluster (e.g. `Endo:hc_3`, `High_to_Endo:hc_1`, or `Outlier`)
   - `min_d` — minimum Mahalanobis distance across clusters
   - `nearest_cluster`, `nearest_r95` — closest cluster and its 95% distance threshold
   - `is_outlier` — 1 if outside r95 (or relaxed threshold), 0 otherwise
   - `top2_cluster`, `top2_dist`, `top2_r95` — second-best cluster diagnostics
   - `top1_w`, `top2_w` — RBF-based soft assignment weights
   - plus optional `x, y, w, h` if coordinates were present in the test CSV

2. **Slide×cluster matrix (wide format)**
    Location:

   ```
   {output_dir}/slide_cluster_matrix.csv
   ```

   (file name controlled by `output.matrix_name`)

   Each row corresponds to one slide and contains:

   - `slide_id`
   - `group` — true subtype short code
   - one column per cluster in `cluster_list` — proportion of patches assigned to that cluster
   - `Outlier` — proportion of patches assigned as `Outlier`

This matrix provides a compact summary of **how strongly each slide is enriched for each spectrum pattern**, and can be used for downstream visualization or statistical analysis.

## **Spectrum Mapping for Independent Slides**

### **Purpose of the Script**

`spectrum_mapping.py` maps **independent-test high-contribution patches** onto the **Morphologic Spectrum** constructed in the interpretability analysis module.

For each slide:

1. it builds spectrum statistics from training embeddings and cluster results *(paths defined in `config.yaml`)*,
2. performs **Mahalanobis-based soft assignment** for each prediction subset,
3. applies an EC–HGSC **hybrid assignment rule** when needed, and
4. merges per-prediction results into a **single patch-level assignment CSV**, grouped by true subtype.

This output is used by `spectrum_atten_geojson_gen.py` to generate slide-level MorphoXAI explanations.

------

### **Inputs**

The script requires three categories of inputs, all specified in **`config.yaml`**.

------

### **1. Training spectrum inputs** *(defined in `config.yaml: paths.train_embeds`, `paths.train_clusters`, `paths.transition_\*`)*

Used to reconstruct the morphologic spectrum statistics:

- `paths.train_embeds` — high-contribution patch embeddings for core subtypes
- `paths.train_clusters` — cluster CSV directories for core subtypes
- `paths.transition_embeds`, `paths.transition_clusters` — optional transition-phenotype embeddings/clusters

Required columns in embedding CSVs include `slide_id`, `tile_id`, and `embed`.
 Cluster CSVs include `slide_id`, `tile_id`, cluster name (`hc_*`), and patch coordinates.

These inputs allow the script to compute:

- per-cluster means μ
- shrunk covariance Σ
- r95 thresholds
- tau values for RBF weighting
- kNN banks for EC↔HGSC transition detection

------

### **2. Independent high-contribution patch files**

*(defined in `config.yaml: paths.test_embeds`)*

These are generated by the **High-Contribution Patch Extraction for Independent Slides** script.

Each CSV (e.g., `independent_Clear.csv`, `independent_Endo.csv`) must contain:

- `slide_id`
- `tile_id`
- `embed`
- `pred` (short subtype: Clear / Endo / High / Border)

Optional columns such as `label` or `x,y,w,h` will be preserved if present.

Loaded via:

```
df_test = read_test_embeddings(cfg["paths"], EMBED_COL, COORD_COLS)
```

------

### **3. Slide-level summary of full-data model predictions**

*(defined in `config.yaml: paths.each_slide_result`, with subtype mapping in `subtypes.long_to_short`)*

`each_slide_result.csv` stores the **full-data model’s performance on the independent test set** and must include:

- `slide_id`
- `name_label` — ground truth subtype (long form)
- `name_pred` — predicted subtype (long form)
- `correctness` — True/False

These long names are converted into **short subtype codes** (Clear / Endo / High / Border) based on:

```
subtypes:
  long_to_short:  # defined in config.yaml
```

Used to:

- assign each slide to the correct output subfolder
- determine whether to activate the **Endo↔High hybrid mode**

If a slide is missing in `each_slide_result`, the script falls back to the `label` column in test embeddings (if available).

------

### **Outputs**

All outputs are written under:

```
paths.output_dir  # defined in config.yaml
```

#### **1. Slide-level assignment CSV**

This file contains the **spectrum-mapping results for a single independent test slide**.

Location:

```
{output_dir}/{true_short}/{slide_id}_patch_assign.csv
```

Contains one row per patch with:

- `slide_id`, `tile_id`, `pred`
- `assigned_cluster`
- `min_d`, `nearest_cluster`, `nearest_r95`
- `is_outlier`
- `top2_cluster`, `top2_dist`, `top2_r95`
- `top1_w`, `top2_w`
- optional coordinates (`x,y,w,h`) if provided in the test input

#### **2. Slide×cluster proportion matrix**

```
{output_dir}/slide_cluster_matrix.csv
```

*(file name defined in `output.matrix_name` in config.yaml)*

Each row contains:

- `slide_id`
- `group` (true subtype)
- proportion of patches assigned to each spectrum cluster
- proportion of `Outlier`

## Attention + Spectrum GeoJSON Generation (Single Slide)

### **Purpose of the Script**

This script generates a **slide-level explanation** for a single independent WSI by **combining attention heatmaps and morphologic-spectrum annotations**.

Given the spectrum-mapping results produced by `spectrum_mapping.py`, the corresponding CONCH feature bag, and the full-data model checkpoint, this script:

1. **Reads the mapped spectrum clusters** (`{slide_id}_patch_assign.csv`, output of `spectrum_mapping.py`)
2. **Matches patch IDs with the original H5 feature bag** to recover patch coordinates
3. **Runs the AttentionNet model** to compute per-class attention scores
4. **Fuses attention tiles + spectrum annotations** into **ONE GeoJSON file**, ready to be imported into **MorphoExplainer**
5. *(Optional)* Generates per-class JPG overlays for QA visualization

The resulting GeoJSON provides a **morphology-grounded, interpretable, slide-level explanation** for how the model makes predictions.

------

### **Inputs (all paths defined in `config.yaml`)**

This script requires several key inputs, each loaded from paths specified in `config.yaml`.

------

#### **1. Patch-level spectrum mapping results**

*(defined in `paths.output_dir` and `output.patch_csv_pattern`)*

`spectrum_mapping.py` generates per-slide mapping files:

```
{output_dir}/{true_short}/{slide_id}_patch_assign.csv
```

This CSV contains:

- `tile_id`
- `assigned_cluster` (will be renamed to `cluster`)
- `pred` (short subtype code)
- optional patch coordinates (`x,y,w,h`)

The script locates this file via:

```
assign_csv, group = find_assign_csv_for_slide(slide_id, mapping_root, patch_pat)
```

------

#### **2. CONCH feature bag (.h5)**

*(defined in `paths.h5_base`)*

Used to retrieve:

- patch embeddings (`features`)
- patch IDs (`tile_ids`)
- patch coordinates (`coords`: minx, miny, maxx, maxy)

Loaded using:

```
features, coords_np, tile_ids = get_features_from_h5(h5_path, device=device)
```

Required for:

- matching cluster assignments to real patch locations
- drawing polygons in GeoJSON
- computing attention maps

------

#### **3. The WSI (.svs) file**

*(provided via `--input-slide` argument)*

Used to:

- load the slide via OpenSlide
- read a downsampled background thumbnail
- scale patch coordinates for optional JPG overlays

The slide ID (`slide_path.stem`) determines:

- the H5 feature path
- the patch_assign file name
- the GeoJSON output name

------

#### **4. Full-data model checkpoint (AttentionNet)**

*(defined in `paths.attn_checkpoint`)*

Required to compute:

- class-level attention maps (`A_raw_np`)
- predicted class probabilities (`prob_np`)

Loaded via:

```
model = load_trained_model(device, paths["attn_checkpoint"], ...)
```

------

#### **5. Manifest file for class labels**

*(defined in `paths.manifest`)*

The manifest CSV must contain:

- `label` (integer index)
- `class` (full subtype name)

Used to build:

```
class_names, n_classes = get_class_names_and_count(paths["manifest"])
```

These human-readable class names are used in GeoJSON metadata.

------

#### **6. Output directory settings**

*(defined in `paths.spectrum_tiles_dir`, `paths.geojson_root`, `output.tiles_csv_name`, `output.geojson_name`)*

- Intermediate tiles CSV is written under:
   `{spectrum_tiles_dir}/{true_short}/`
- Final GeoJSON is written under:
   `{geojson_root}/{true_short}/`

Example:

```
spectrum_qupath4_mul_predictions/High/HGSC12_spectrum_tiles.csv
spectrum_geojson_final/High/HGSC12_ATTN+SPECTRUM.geojson
```

------

### **Example Run**

```
python spectrum_atten_geojson_gen.py \
    --config config.yaml \
    --input-slide /absolute/path/to/YourSlide.svs \
    --save-jpg
```

- `--save-jpg` is optional; if enabled, the script produces per-class heatmap overlays.

------

### **Outputs**

This script produces two major outputs for a single slide.

------

#### **1. Tiles CSV (Spectrum + Coordinates)**

Location:

```
{spectrum_tiles_dir}/{true_short}/{slide_id}_spectrum_tiles.csv
```

Contains (per patch):

- `slide_id`
- `tile_id`
- `cluster` (spectrum cluster assigned by spectrum_mapping.py)
- `pred` (subtype for which this patch is relevant)
- `x, y, width, height` (patch coordinates derived from H5)

This file is the bridge between spectrum assignments and GeoJSON visualization.

------

#### **2. Final GeoJSON (Attention + Spectrum)**

Location:

```
{geojson_root}/{true_short}/{slide_id}_ATTN+SPECTRUM.geojson
```

This GeoJSON contains:

#### **Attention tiles**

- polygons with per-class attention scores
- stored with `objectType: "tile"`

#### **Spectrum annotations**

- polygons colored by spectrum cluster
- cluster metadata (name, color, pattern description)
- stored with `objectType: "annotation"`

#### **Prediction summary block**

For each class:

- predicted probability
- list of spectrum patterns present in that class
- classification sets / morphological descriptions

This GeoJSON can be directly loaded into **MorphoExplainer** for interactive analysis.

------

#### *(Optional)* Per-class Attention JPG Overlays

If `--save-jpg` is supplied, the script generates files like:

```
{slide_id}_attn_PRED_{class_name}.jpg
```

These overlays combine:

- a WSI thumbnail
- a heatmap of class-specific attention scores

Useful for inspection and debugging.

## MorphoExplainer: QuPath Extension for Visualizing MorphoXAI slide-level explanation

The usage manuals for MorphoExplainer Manual is provided in the plugins/ directory.