# d2trckr

d2trckr 

## Getting Started

### Prerequisites

```
```

### Installing

Use d2trckr.yml to create conda evironment 

```
conda env create -f d2trckr.yml
```

## Usage

Example:
```
python run.py configs/config_1_meeting.yml --only_n_frames 1500
```

Then, use `detections` variable inside loop to process available detections from Detectron2.
