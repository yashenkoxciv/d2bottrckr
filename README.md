## Getting Started

This project is based on detectron2 and Bag of tricks ReID model. The main idea is that every detected object is close enough to its detection on the previous frame. Unfortunately, detectron is tend to miss some detections and produce false-positives (duplicates). See the guy on the left side of the scene.
![Odessa demo](output/demo.gif).

### Installing

Use d2trckr.yml to create conda evironment 

```
conda env create -f d2trckr.yml
```

Put ReID model's weights to configs directory:
https://drive.google.com/open?id=1kGn-c6e0LDsSKG8QkQNeQJ0ySCBE7Inu

## Usage

```
python run.py configs/config_1_meeting.yml --only_n_frames 1500

```

And look the results in output folder.

## Developing

Then, use `detections` variable inside loop to process available detections from Detectron2.

## Additional info

See report about this project
https://drive.google.com/open?id=1jJN71wJbVBdswYePqjiXEIb5SiChG6k7
