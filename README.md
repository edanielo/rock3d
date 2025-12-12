# Photogrammetry & AI Segmentation Pipeline (MVP)
- Final project for "Object Detection and Video Processing" course.

This project implements an automated pipeline for generating 3D models from lithographic samples (rocks), combining Deep Learning techniques for semantic segmentation and AliceVision, a Photogrammetric Computer Vision Framework for geometric reconstruction.

## Features

- Intelligent Segmentation.\
  U-2-Net is used to automatically remove the background overcoming deficiencies in lightning and image capture, which traditional image Chroma Key (HSV) methods could not process.

- Photogrammetry Pipeline.\
  Complete orchestration of AliceVision (Meshroom) nodes via Python. From `CameraInit` to `Texturing`. Sequential matching optimized for burst sequential captures.

- Robust Preprocessing:
  Green spill reduction and automatic EXIF metadata injection and correction for uncommon sensors (phone camera).

## Installation

Prerequisites
- Python 3.8+
- AliceVision Framework.
- ExifTool

Setup the Environment

1. Clone this repository:

2. Install python dependencies. Use the `environment.yml` file to create the conda environment or use your preferred environment manager but make sure you include all libraries mentioned in `environment.yml`.

3. Configure AliceVision path. Run the configuration script to detect the AliceVision installation and generate the necessary environment variables.
```
python setup_env.py
source vars.sh
```

**Note** If the sensor you are using is not included in `cameraSensors.db`, modify the setup_env.py file and change the information of the sensor you are using so it is injected into the sensor database (currently it injects the information for the camera of a Samsung Galaxy S22 Ultra phone).


## Usage

Place the raw photographs in the `data/raw` folder inside the source directory of this project. The script will remove the background and correct metadata. For better result make sequential pictures at 0°, 45° and 90° of the sample while rotating it 10° between pictures.

```
python src/preprocess_ai.py
```

The output will be the clean images in `data/sanitized`.

Finally, execute the photogrammetry pipeline. This process make take several minutes depending on the hardware available.

```
python src/pipeline.py
```

Credits and References
- [AliceVision](https://github.com/alicevision/AliceVision) Meshroom: Griwodz et al., 2021.

- U-2-Net: Qin et al., 2020.

- Team: Edgar Ocampo, Angel Rosado, Astrid Osorno, Maricarmen Buenfil.

