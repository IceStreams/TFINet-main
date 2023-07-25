# The paper "A Difference-Enhanced Semantic Change Detection Network Based on Tri-Branch Feature Interaction."



## Our Method


### Model

![image](https://github.com/xbysq/TFINet-main/blob/main/img/TFINet.png)



## Getting Started

### Dataset

    ├── dataset                    # download from the link above
    │   ├── train                  # training set
    |   |   ├── im1
    |   |   └── ...
    │   └── Val                    # the final testing set (without labels)
    |

### Training
```
# Please refer to utils/options.py for more arguments

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py 
```

### Testing
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py
```
## Reference
- [1st place in SenseTime2020 competition](https://github.com/LiheYoung/SenseEarth2020-ChangeDetection)

## Replication of other methods

Continuing to publicize

- [sscdl and bisrnet]()

**The code regarding the CD will also be made public subsequently.
