This is the SER30K dataset from our research "*SER30K: A Large-Scale Dataset for Sticker Emotion Recognition*".

* The './Images' folder contains 1,887 folders corresponding to 1,887 sticker topics, containing a total of 30,739 images.

* The './Annotations' folder:
  
  * all_annos.json contains all the annotations.
  
  * The './image-level' folder contains train.json/val.json/test.json corresponding to training set/validation set/test set, which is split from all_annos.json.

The specific meaning of each sample label in the json file:

* topic: Sticker topic, i.e. folder name

* file_name: Image file name

* text：The text content contained in the image, or '' if it does not exist

* anno1/anno2/anno3: The original three annotators' emotional labels

* anno: Emotion labels for model learning

Sample code for reading images:

```python
import os
import cv2
import json
from os.path import join
from tqdm import tqdm

def main():
    p = 'SER_Dataset/Annotations'
    with open(join(p, 'all_annos.json'), 'r') as f:
        all_annos = json.load(f)
    annos = all_annos['annotations']
    for anno in tqdm(annos):
        topic, name = anno['topic'], anno['file_name']
        img = cv2.imread(join('SER_Dataset/Images', topic, name))

if __name__=='__main__':
    main()
```


