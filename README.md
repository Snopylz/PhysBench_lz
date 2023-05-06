# PhysBench 
Please use the [Tutorial/Noob Heart.ipynb](https://github.com/KegangWangCCNU/PhysBench/blob/main/Tutorial/Noob%20Heart.ipynb) to learn about this framework.

## Models  

We implemented 6 neural models and 3 unsupervised models, DeepPhys, TS-CAN, PhysNet, PhysFormer, 1D CNN, NoobHeart, Chrom, ICA, and POS. Among them, the 1D CNN is a new model we proposed that uses only one-dimensional convolution with minimal computational complexity and high performance. NoobHeart is a toy model used in the tutorial with only 361 parameters and includes a simple 2 layers 3-dimensional convolution structure; however it has decent performance making it suitable as an entry-level model. Chrom，ICA，and POS are three unsupervised models. Among the neural models，PhysFormer is implemented using Pytorch while others use Tensorflow.  

For unsupervised methods, please refer to `unsupervised_methods.py`; for methods implemented using TensorFlow, please refer to `models.py`; for methods implemented using PyTorch, please refer to `models_torch.py`. Our framework is not dependent on a specific deep learning framework. Please configure the environment as needed and install the required packages using `requirements.txt`.
|Model|Resolution|Params|Frame FLOPs|Type|  
|:-:|:-:|:-:|:-:|---|  
|DeepPhys|36x36|532K|52M|2D CNN|  
|TS-CAN|36x36|532K|52M|2D CNN|  
|PhysNet|32x32|770K|54M|3D CNN|  
|PhysFormer|128x128|7.03M|324M|Transformer|  
|1D CNN|8x8|196K|261K|1D CNN|  
|NoobHeart|8x8|361|5790|3D CNN|  
## Datasets  
Adding a dataset is simple, just write a loader and include a file directory (usually only 20 lines of code). Currently supported loaders are RLAP (i.e., CCNU), UBFC-rPPG2, PURE, and SCAMPS. You can use our recording program `PhysRecorder/PhysRecorder.exe` to record datasets, just need a webcam and Contec CMS50E to collect strictly synchronized lossless format datasets, which can be directly used with the RLAP loader.
|Dataset|Participants|Frames|Synchronicity|  
|:-:|:-:|:-:|:-:|  
|RLAP|58|3.53M|Good|   
|PURE|10|106K|Good|  
|UBFC|42|75K|Bad| 
|SCAMPS|2800|1.68M|Good|  
## Train and Test
Train on our RLAP dataset, please see the `benchmark_RLAP` folder. Train on the SCAMPS dataset, please see the `benchmark_SCAMPS` folder. In addition, for ablation experiments and training on PURE and UBFC, please see `benchmark_addition`. All code is provided in Jupyter notebooks with our replication included; if you have read the tutorial, replicating results should be easy.   

## Training evaluation on RLAP  
RLAP is an appropriate training set, and we divide RLAP into training ,validation and testing set. In addition, tests were also conducted on the entire UBFC and PURE datasets. For code and results, please refer to `benchmark_RLAP`.  

### Intra-dataset testing on RLAP  
|Model|MAE|RMSE|Pearson Coef.|   
|:-:|:-:|:-:|:-:|  
|DeepPhys|1.52|4.40|0.906|  
|TS-CAN|1.23|3.59|0.937|  
|PhysNet|1.12|4.13|0.916|  
|PhysFormer|1.56|6.28|0.803|  
|1D CNN|1.07|4.15|0.917|  
|NoobHeart|1.79|5.85|0.832|  

### Intra-dataset testing on RLAP-rPPG  
|Model|MAE|RMSE|Pearson Coef.|   
|:-:|:-:|:-:|:-:|  
|DeepPhys|1.76|4.87|0.877|  
|TS-CAN|1.23|3.82|0.922|  
|PhysNet|1.04|3.80|0.923|  
|PhysFormer|0.78|2.83|0.957|  
|1D CNN|0.81|2.97|0.953|  
|NoobHeart|1.57|4.71|0.883|  

### Cross-dataset testing on UBFC  
|Model|MAE|RMSE|Pearson Coef.|   
|:-:|:-:|:-:|:-:|  
|DeepPhys|1.06|1.51|0.997|  
|TS-CAN|0.99|1.44|0.997|  
|PhysNet|0.92|1.46|0.997|  
|PhysFormer|1.06|1.53|0.997|  
|1D CNN|0.87|1.40|0.997|  
|NoobHeart|1.14|1.69|0.996|  

### Cross-dataset testing on PURE  
|Model|MAE|RMSE|Pearson Coef.|   
|:-:|:-:|:-:|:-:|  
|DeepPhys|2.80|8.31|0.937|  
|TS-CAN|2.12|6.67|0.960|  
|PhysNet|0.51|0.91|0.999|  
|PhysFormer|1.63|9.45|0.941|  
|1D CNN|0.37|0.63|1.000|  
|NoobHeart|0.45|0.70|1.000|  

## Training evaluation on SCAMPS  
Training on synthetic datasets is difficult, and we observed that overfitting can easily occur, requiring many steps to prevent overfitting, such as controlling the learning rate, additional regularization operations, etc. We were unable to reproduce the performance of rPPG Toolbox but believe it is reproducible with more parameter tuning. Smaller models may not be prone to overfitting; NoobHeart is an example where we froze the LayerNormalization layer with initial parameters and trained for 5 epochs while achieving similar performance as training on real datasets. This could be the first step in training on synthetic datasets.  

For caution, we only display the results of NoobHeart.
### Cross-dataset testing on UBFC  
MAE: 1.05  
RMSE: 1.49  
Pearson Coef.: 0.997  

### Cross-dataset testing on PURE  
MAE: 0.53  
RMSE: 0.88  
Pearson Coef.: 0.999  

## Visualization  
Please run `visualization.py` to open the visualization webpage. Before visualizing, make sure all result files are saved in the `results` folder. When the framework generates result files, it links to the dataset files, so the visualization webpage can display face images synchronously. Once the link is invalid, such as when dataset files are moved, faces cannot be displayed on the webpage.  

![image](https://github.com/KegangWangCCNU/PICS/blob/main/PhysBench.gif)  

## Request RLAP dataset  

If you wish to obtain the RLAP dataset, please send an email to kegangwang@mails.ccnu.edu.cn and cc yantaowei@ccnu.edu.cn, with the Data Usage Agreement attached.

## Citation  

If you use PhysBench framework, PhysRecorder data collection tool, or the models included in this framework, please cite the following paper:
```
```
I am looking for a CS Ph.D. position, my research field is computer vision and physiological remote sensing, and I will graduate with a master's degree in June 2024. If anyone is interested, please send an email to kegangwang@mails.ccnu.edu.cn. 
