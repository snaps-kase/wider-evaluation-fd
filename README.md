# WiderFace-Evaluation
Python Evaluation Code for [Wider Face Dataset](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)


## Usage
python3 3.7

##### before evaluating ....

````
pip install Cython
python3 setup.py build_ext --inplace
````

##### evaluating
**Wider-face ImageDataset(Valid) You can Download from Under Link**
- https://drive.google.com/file/d/1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q/view
- Need valid_imageset_path for evaluation

**GroungTruth:** `wider_face_val.mat`, `wider_easy_val.mat`, `wider_medium_val.mat`,`wider_hard_val.mat`

````
CLI : python3 src/evaluation.py -p <your prediction dir> -g <groud truth dir>
Run : evaluation.py 
TEST : streamlit run fd_streamlit.py 
 - set dir root before TEST Run
````

## Acknowledgements
some code borrowed from Sergey Karayev