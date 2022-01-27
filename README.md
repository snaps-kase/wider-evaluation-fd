# WiderFace-Evaluation
Python Evaluation Code for [Wider Face Dataset](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)


## Usage


##### before evaluating ....

````
python3 setup.py build_ext --inplace
````

##### evaluating

**GroungTruth:** `wider_face_val.mat`, `wider_easy_val.mat`, `wider_medium_val.mat`,`wider_hard_val.mat`

````
deprecated : python3 evaluation.py -p <your prediction dir> -g <groud truth dir>
Run : evaluation.py 
````

## Bugs & Problems
please issue

## Acknowledgements

some code borrowed from Sergey Karayev