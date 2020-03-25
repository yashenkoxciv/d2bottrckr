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

Put ReID model's weights to configs directory:
https://drive.google.com/open?id=1kGn-c6e0LDsSKG8QkQNeQJ0ySCBE7Inu

## Usage

Example:
```
python run.py configs/config_1_meeting.yml --only_n_frames 1500
```

Then, use `detections` variable inside loop to process available detections from Detectron2.

See report about this project
https://drive.google.com/open?id=1jJN71wJbVBdswYePqjiXEIb5SiChG6k7
