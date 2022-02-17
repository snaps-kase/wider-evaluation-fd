# WiderFace-Evaluation
This code is evaluate for wider-face medium validation

test provided : `13--Interview` Only

you can test all of validation Image.
Just Download Wider-face ImageDataset(Valid)

## Usage
* python3 3.7
* face_detection Model : RetinaFace

##### before evaluating ....
````
pip install retinaface
````

##### evaluating
**Wider-face ImageDataset(Valid) Download from Under Link**
- https://drive.google.com/file/d/1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q/view
- Need valid_imageset_path for evaluation

**GroungTruth:** `wider_face_val.mat`, `wider_easy_val.mat`, `wider_medium_val.mat`,`wider_hard_val.mat`
I converted wider_face_val.mat and wider_medium_val.mat to wider_medium_val.json
It has all validation Information

e.g.) {`dir name`: {`file name`: {`gt_index`:[], `gt_bbx_list`:[]}}}


````
TEST : streamlit run fd_streamlit.py 
 - set dir root before TEST Run
```