# Data Preparation

G2VD does not redistribute video files. This repository provides code,
configuration templates, and metadata used to organize public datasets for
AI-generated video detection. Please obtain each dataset from its official
release channel and follow the corresponding terms of use.

## Dataset Root

Set `dataset_root` in your copied config to the directory containing the video
datasets:

```text
/path/to/gvd_datasets/
  genvidbench-143k/
  genvideo-2271k/
  gvd-11k/
  gvf-2.8k/
```

The metadata files contain relative paths such as:

```text
genvidbench-143k/hd-vg-130m/xxx.mp4
genvideo-2271k/lavie/xxx.mp4
gvd-11k/sora/xxx.mp4
gvf-2.8k/kling/xxx.mp4
```

The dataloader resolves each video by joining `dataset_root` and the relative
`video_path` field in `dataset_metadata/`.

## Splits and Source Lists

Config templates use neutral split names. Replace the following fields with
your local setup:

```yaml
train_metadata_dir: "./dataset_metadata/your_train_split/"
train_video_data_list:
  - "real_source"
  - "fake_source_1"

val_metadata_dir: "./dataset_metadata/your_val_split/"
val_video_data_list:
  - "real_source"
  - "fake_source_1"
```

For testing, edit:

```yaml
test_metadata_dir: "./dataset_metadata/your_test_split/"
test_video_data_list:
  - "real_source"
  - "fake_source_1"
```

Available dataset/source names are summarized in:

```text
configs/dataset_detail.yaml
```

If you remove or reorganize metadata files, update these config fields
accordingly.

## Dataset References

The repository includes metadata for four public dataset families:

- GenVidBench: "GenVidBench: A 6-Million Benchmark for AI-Generated Video Detection".
- GenVideo: introduced with "DeMamba: AI-generated video detection on million-scale GenVideo benchmark".
- GVD: "AI-Generated Video Detection via Spatial-Temporal Anomaly Learning".
- GVF: "DeCoF: Generated Video Detection via Frame Consistency: The First Benchmark Dataset".

Academic citations identify the dataset sources, but they do not grant data
redistribution rights. Users should download datasets from the official project
pages or release channels associated with those papers.

## Metadata

`dataset_metadata/` contains JSON metadata with relative video paths, base
labels, generation-source labels, semantic labels, and basic video properties.
It does not contain videos.
