# DICOM Reader — CT and PET/CT Examination

A PyQt6 DICOM viewer focused on CT and PET/CT examination workflows. It
reads DICOM series, builds 3D volumes, and provides the tools a reader
typically needs at the workstation: windowing presets, multi-planar
reformation, PET SUV conversion, PET/CT fusion, distance and ROI
measurements, a pixel probe, and a full DICOM tag browser.

## Features

- **Loading**: open a folder or a single file. Multiple series are
  auto-grouped by `SeriesInstanceUID` and sorted by
  `ImagePositionPatient` along the slice normal. Rescale slope and
  intercept are applied, so CT values are in Hounsfield Units and PET
  values are in `Bq/ml` (when `Units = BQML`).
- **Tri-planar view**: axial, coronal, and sagittal viewports with
  mouse-wheel scroll and per-plane sliders.
- **CT windowing**: presets for Soft Tissue, Lung, Mediastinum, Bone,
  Brain, Liver, Abdomen, and Angiography; editable W/L spinboxes.
- **PET SUV**: body weight (BW), body surface area (BSA — Du Bois), and
  lean body mass (LBM — James) normalizations, with automatic decay
  correction from injection to acquisition time.
- **PET/CT fusion**: PET is resampled onto the CT grid in patient
  coordinates and blended as a colormapped overlay (`hot`, `pet`,
  `gray`) with adjustable alpha, threshold, and SUV-max ceiling.
- **Examination tools**:
  - Pixel probe (shows HU / Bq/ml / SUV at cursor)
  - Distance (mm)
  - Rectangular ROI (count, mean, std, min, max, area in mm²; SUV stats
    when applicable)
  - Elliptical ROI (same stats)
- **DICOM tag browser**: inspects any file's header as a tree, including
  nested sequences.

## Install

```bash
cd dicom-reader
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Dependencies: `pydicom`, `numpy`, `scipy`, `PyQt6`, `pyqtgraph`,
`Pillow`. Python 3.10+.

## Run

```bash
dicom-reader                    # launch with no data
dicom-reader /path/to/study     # open a study folder
dicom-reader /path/to/file.dcm  # open a single file
```

## Tests

```bash
pip install -e '.[dev]'
pytest
```

The test suite covers SUV conversion (including F-18 decay), CT
windowing, and ROI/distance math. The UI is not exercised in CI.

## Layout

```
dicom_reader/
  io/          # DICOM loader, series model
  imaging/     # windowing, SUV, fusion, MPR
  tools/       # distance and ROI stats
  ui/          # PyQt6 main window, viewports, tools panel, tag browser
```

## Clinical notes / caveats

- SUV values depend on correct acquisition, injection, and dose time
  stamps in the DICOM header. Incomplete metadata disables SUV.
- The viewer displays voxel values directly; it performs **no**
  partial-volume correction, attenuation check, or motion correction.
- Measurements use the displayed plane's spacing. On non-isotropic
  volumes, MPR views scale based on slice spacing rather than resampling
  to isotropic — a future upgrade.
- Not a medical device. For research, teaching, and development only.

## Roadmap

- Oblique MPR and thick-slab MIP
- Automatic registration between CT and PET when UIDs do not match
- Cine playback
- Preset management and hanging protocols
- Export screenshots and measurement reports
