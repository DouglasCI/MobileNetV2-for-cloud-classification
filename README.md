
# PyTorch MobileNetV2 in cloud classification
### This code uses a PyTorch MobileNetV2 model pre-trained with ImageNet data for feature extraction of cloud images from CCSN_v2 database.

#### These 3 arguments are mandatory for execution:  
> ```-dp, --data-path ``` path to dataset directory.  
> ```-m, --model``` model output filename (you need to add *.pth* in the name).  
> ```-p, --plot``` plot output filename (you need to add desired image *format* in the name).
#### These arguments are optional:
> ```-bs, --batch-size``` batch size for training. Default = 22.  
> ```-ep, --epochs``` number of epochs to train. Default = 10.  
> ```-lr, --learn-rate``` learning rate. Default = 0.0001.  
> ```--no-cuda``` disables CUDA training. Default = false.  
> ```--no-mps``` disables macOS GPU training. Default = false.  
> ```--freeze-top``` freeze only top layers. Yields better results, takes longer to train. Default = false.  
> ```--augment-data``` choose to augment train data at runtime. Default = false.  
> ```--seed``` random seed. Default = 1.

#### Example:  
> ```python main.py -dp dataset/CCSN_v2 -m output/model.pth -p output/plot.png -bs 32 -ep 30 -lr 0.001 --freeze-top --seed 42```

#### You can check these arguments at any time in command line by running ```python main.py -h``` or ```python main.py --help```.

#### The dataset used for training is [here](https://www.kaggle.com/datasets/mmichelli/cirrus-cumulus-stratus-nimbus-ccsn-database).

#### The program requires you to separate the original dataset in two folders: train and test. 
#### Like this:  
dataset/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;train/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;class1/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;img1.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;class2/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;img2.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;test/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;class1/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;img3.jpg   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;class2/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;img4.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...  

#### Check [splitfolders](https://pypi.org/project/split-folders/) library for a simple way to do that.

## License

[CC0 1.0 Public Domain](http://creativecommons.org/publicdomain/zero/1.0/)