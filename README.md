# PhysBench 
### Please use the [Tutorial/Noob Heart.ipynb](https://github.com/KegangWangCCNU/PhysBench/blob/main/Tutorial/Noob%20Heart.ipynb) to learn about this framework.
![image](https://github.com/KegangWangCCNU/PICS/blob/main/PhysBench.gif)  
## Models  

We implemented 6 neural models and 3 unsupervised models, DeepPhys, TS-CAN, PhysNet, PhysFormer, 1D CNN, NoobHeart, Chrom, ICA, and POS. Among them, the 1D CNN is a new model we proposed that uses only one-dimensional convolution with minimal computational complexity and high performance. NoobHeart is a toy model used in the tutorial with only 361 parameters and includes a simple 2 layers 3-dimensional convolution structure; however it has decent performance making it suitable as an entry-level model. Chrom，ICA，and POS are three unsupervised models. Among the neural models，PhysFormer is implemented using Pytorch while others use Tensorflow.  

For unsupervised methods, please refer to `unsupervised_methods.py`; for methods implemented using TensorFlow, please refer to `models.py`; for methods implemented using PyTorch, please refer to `models_torch.py`. Our framework is not dependent on a specific deep learning framework. Please configure the environment as needed and install the required packages using `requirements.txt`.
|Model|Publication|Resolution|Params|Frame FLOPs|Type|  
|:-:|:-:|:-:|:-:|:-:|---|  
|DeepPhys|ECCV 19|36x36|532K|52M|2D CNN|  
|TS-CAN|NIPS 20|36x36|532K|52M|2D CNN|  
|PhysNet|BMVC 19|32x32|770K|54M|3D CNN|  
|PhysFormer|CVPR 22|128x128|7.03M|324M|Transformer|  
|Seq-rPPG|This paper|8x8|196K|261K|1D CNN|  
|NoobHeart|This paper|8x8|361|5790|3D CNN|  
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

## Inference on a single video  
To extract BVP signals from your own collected video, please execute the following code.  
`python inference.py --video face.avi --out BVP.csv `  
Its output `BVP.csv` contains the BVP signal values corresponding to each frame.   
By default, it uses a pre-trained 1D CNN model (i.e., Seq-rPPG) on RLAP.

## Training evaluation on RLAP  
RLAP is an appropriate training set, and we divide RLAP into training ,validation and testing set. In addition, tests were also conducted on the entire UBFC and PURE datasets. For code and results, please refer to `benchmark_RLAP`.  

### Intra-dataset testing on RLAP  
|Model|MAE|RMSE|Pearson Coef.|   
|:-:|:-:|:-:|:-:|  
|DeepPhys|1.52|4.40|0.906|  
|TS-CAN|1.23|3.59|0.937|  
|PhysNet|1.12|4.13|0.916|  
|PhysFormer|1.56|6.28|0.803|  
|Seq-rPPG|1.07|4.15|0.917|  
|NoobHeart|1.79|5.85|0.832|  

### Intra-dataset testing on RLAP-rPPG  
<form action="" method="post" name="form1" class="form" id="form1">
<table width="100%" cellpadding="0" cellspacing="0">
<thead>
<tr>
<th rowspan="2" colspan="1">Model</td>
<th colspan="3">HR</td>
<th colspan="3">HRV-SDNN</td>
</tr>
<tr>
<th rowspan="1" colspan="1">MAE</td>
<th rowspan="1" colspan="1">RMSE</td>
<th rowspan="1" colspan="1">Pearson Coef.</td>
<th rowspan="1" colspan="1">MAE</td>
<th rowspan="1" colspan="1">RMSE</th>
<th rowspan="1" colspan="1">Pearson Coef.</th>
</tr>
</thead>
<tbody align="center">
<tr>
<td rowspan="1" colspan="1">DeepPhys</td>
<td rowspan="1" colspan="1">1.76</td>
<td rowspan="1" colspan="1">4.87</td>
<td rowspan="1" colspan="1">0.877</td>
<td rowspan="1" colspan="1">57.6</td>
<td rowspan="1" colspan="1">64.2</td>
<td rowspan="1" colspan="1">0.338</td>
</tr>
<tr>
<td rowspan="1" colspan="1">TS-CAN</td>
<td rowspan="1" colspan="1">1.23</td>
<td rowspan="1" colspan="1">3.82</td>
<td rowspan="1" colspan="1">0.922</td>
<td rowspan="1" colspan="1">50.1</td>
<td rowspan="1" colspan="1">59.3</td>
<td rowspan="1" colspan="1">0.395</td>
</tr>
<tr>
<td rowspan="1" colspan="1">PhysNet</td>
<td rowspan="1" colspan="1">1.04</td>
<td rowspan="1" colspan="1">3.80</td>
<td rowspan="1" colspan="1">0.923</td>
<td rowspan="1" colspan="1">36.4</td>
<td rowspan="1" colspan="1">43.8</td>
<td rowspan="1" colspan="1">0.306</td>
</tr>
<tr>
<td rowspan="1" colspan="1">PhysFormer</td>
<td rowspan="1" colspan="1">0.78</td>
<td rowspan="1" colspan="1">2.83</td>
<td rowspan="1" colspan="1">0.957</td>
<td rowspan="1" colspan="1">28.8</td>
<td rowspan="1" colspan="1">34.4</td>
<td rowspan="1" colspan="1">0.450</td>
</tr>
<tr>
<td rowspan="1" colspan="1">Seq-rPPG</td>
<td rowspan="1" colspan="1">0.81</td>
<td rowspan="1" colspan="1">2.97</td>
<td rowspan="1" colspan="1">0.953</td>
<td rowspan="1" colspan="1">14.4</td>
<td rowspan="1" colspan="1">22.1</td>
<td rowspan="1" colspan="1">0.424</td>
</tr>
<tr>
<td rowspan="1" colspan="1">NoobHeart</td>
<td rowspan="1" colspan="1">1.57</td>
<td rowspan="1" colspan="1">4.71</td>
<td rowspan="1" colspan="1">0.883</td>
<td rowspan="1" colspan="1">52.3</td>
<td rowspan="1" colspan="1">57.3</td>
<td rowspan="1" colspan="1">0.488</td>
</tr>
</tbody></table>
</form>

### Cross-dataset testing on UBFC  
<form action="" method="post" name="form1" class="form" id="form1">
<table width="100%" cellpadding="0" cellspacing="0">
<thead>
<tr>
<th rowspan="2" colspan="1">Model</td>
<th colspan="3">HR</td>
<th colspan="3">HRV-SDNN</td>
</tr>
<tr>
<th rowspan="1" colspan="1">MAE</td>
<th rowspan="1" colspan="1">RMSE</td>
<th rowspan="1" colspan="1">Pearson Coef.</td>
<th rowspan="1" colspan="1">MAE</td>
<th rowspan="1" colspan="1">RMSE</th>
<th rowspan="1" colspan="1">Pearson Coef.</th>
</tr>
</thead>
<tbody align="center">
<tr>
<td rowspan="1" colspan="1">DeepPhys</td>
<td rowspan="1" colspan="1">1.06</td>
<td rowspan="1" colspan="1">1.51</td>
<td rowspan="1" colspan="1">0.997</td>
<td rowspan="1" colspan="1">30.0</td>
<td rowspan="1" colspan="1">37.8</td>
<td rowspan="1" colspan="1">0.648</td>
</tr>
<tr>
<td rowspan="1" colspan="1">TS-CAN</td>
<td rowspan="1" colspan="1">0.99</td>
<td rowspan="1" colspan="1">1.44</td>
<td rowspan="1" colspan="1">0.997</td>
<td rowspan="1" colspan="1">25.6</td>
<td rowspan="1" colspan="1">31.8</td>
<td rowspan="1" colspan="1">0.588</td>
</tr>
<tr>
<td rowspan="1" colspan="1">PhysNet</td>
<td rowspan="1" colspan="1">0.92</td>
<td rowspan="1" colspan="1">1.46</td>
<td rowspan="1" colspan="1">0.997</td>
<td rowspan="1" colspan="1">12.2</td>
<td rowspan="1" colspan="1">14.9</td>
<td rowspan="1" colspan="1">0.887</td>
</tr>
<tr>
<td rowspan="1" colspan="1">PhysFormer</td>
<td rowspan="1" colspan="1">1.06</td>
<td rowspan="1" colspan="1">1.53</td>
<td rowspan="1" colspan="1">0.997</td>
<td rowspan="1" colspan="1">8.37</td>
<td rowspan="1" colspan="1">11.1</td>
<td rowspan="1" colspan="1">0.921</td>
</tr>
<tr>
<td rowspan="1" colspan="1">Seq-rPPG</td>
<td rowspan="1" colspan="1">0.87</td>
<td rowspan="1" colspan="1">1.40</td>
<td rowspan="1" colspan="1">0.997</td>
<td rowspan="1" colspan="1">4.73</td>
<td rowspan="1" colspan="1">8.25</td>
<td rowspan="1" colspan="1">0.911</td>
</tr>
<tr>
<td rowspan="1" colspan="1">NoobHeart</td>
<td rowspan="1" colspan="1">1.14</td>
<td rowspan="1" colspan="1">1.69</td>
<td rowspan="1" colspan="1">0.996</td>
<td rowspan="1" colspan="1">33.1</td>
<td rowspan="1" colspan="1">36.5</td>
<td rowspan="1" colspan="1">0.697</td>
</tr>
</tbody></table>
</form>

### Cross-dataset testing on PURE  
<form action="" method="post" name="form1" class="form" id="form1">
<table width="100%" cellpadding="0" cellspacing="0">
<thead>
<tr>
<th rowspan="2" colspan="1">Model</td>
<th colspan="3">HR</td>
<th colspan="3">HRV-SDNN</td>
</tr>
<tr>
<th rowspan="1" colspan="1">MAE</td>
<th rowspan="1" colspan="1">RMSE</td>
<th rowspan="1" colspan="1">Pearson Coef.</td>
<th rowspan="1" colspan="1">MAE</td>
<th rowspan="1" colspan="1">RMSE</th>
<th rowspan="1" colspan="1">Pearson Coef.</th>
</tr>
</thead>
<tbody align="center">
<tr>
<td rowspan="1" colspan="1">DeepPhys</td>
<td rowspan="1" colspan="1">2.80</td>
<td rowspan="1" colspan="1">8.31</td>
<td rowspan="1" colspan="1">0.937</td>
<td rowspan="1" colspan="1">86.0</td>
<td rowspan="1" colspan="1">92.0</td>
<td rowspan="1" colspan="1">0.297</td>
</tr>
<tr>
<td rowspan="1" colspan="1">TS-CAN</td>
<td rowspan="1" colspan="1">2.12</td>
<td rowspan="1" colspan="1">6.67</td>
<td rowspan="1" colspan="1">0.960</td>
<td rowspan="1" colspan="1">61.4</td>
<td rowspan="1" colspan="1">74.1</td>
<td rowspan="1" colspan="1">0.293</td>
</tr>
<tr>
<td rowspan="1" colspan="1">PhysNet</td>
<td rowspan="1" colspan="1">0.51</td>
<td rowspan="1" colspan="1">0.91</td>
<td rowspan="1" colspan="1">0.999</td>
<td rowspan="1" colspan="1">22.5</td>
<td rowspan="1" colspan="1">35.7</td>
<td rowspan="1" colspan="1">0.560</td>
</tr>
<tr>
<td rowspan="1" colspan="1">PhysFormer</td>
<td rowspan="1" colspan="1">1.63</td>
<td rowspan="1" colspan="1">9.45</td>
<td rowspan="1" colspan="1">0.941</td>
<td rowspan="1" colspan="1">21.6</td>
<td rowspan="1" colspan="1">32.0</td>
<td rowspan="1" colspan="1">0.576</td>
</tr>
<tr>
<td rowspan="1" colspan="1">Seq-rPPG</td>
<td rowspan="1" colspan="1">0.37</td>
<td rowspan="1" colspan="1">0.63</td>
<td rowspan="1" colspan="1">1.000</td>
<td rowspan="1" colspan="1">9.51</td>
<td rowspan="1" colspan="1">15.8</td>
<td rowspan="1" colspan="1">0.872</td>
</tr>
<tr>
<td rowspan="1" colspan="1">NoobHeart</td>
<td rowspan="1" colspan="1">0.45</td>
<td rowspan="1" colspan="1">0.70</td>
<td rowspan="1" colspan="1">1.000</td>
<td rowspan="1" colspan="1">50.8</td>
<td rowspan="1" colspan="1">58.1</td>
<td rowspan="1" colspan="1">0.657</td>
</tr>
</tbody></table>
</form>

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

## Limitation  
The test data used by PhysBench may not necessarily reflect the accuracy in real-world scenarios, where there are more diverse lighting conditions, head movements, skin tones and age groups. The heart rate provided by the algorithm through Welch method may not fully comply with medical standards and requires further rigorous evaluation before clinical use. We aim to inform users of the weaknesses and limitations of the algorithm as much as possible through the visualization webpage.

## Request RLAP dataset  

If you wish to obtain the RLAP dataset, please send an email to kegangwang@mails.ccnu.edu.cn and cc yantaowei@ccnu.edu.cn, with the Data Usage Agreement attached.

## Citation  

If you use PhysBench framework, PhysRecorder data collection tool, or the models included in this framework, please cite the following <a href="https://github.com/KegangWangCCNU/PICS/raw/main/PhysBench.pdf" target="_blank">paper</a>
```
@misc{wang2023physbench,
      title={PhysBench: A Benchmark Framework for Remote Physiological Sensing with New Dataset and Baseline}, 
      author={Kegang Wang and Yantao Wei and Mingwen Tong and Jie Gao and Yi Tian and YuJian Ma and ZhongJin Zhao},
      year={2023},
      eprint={2305.04161},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
I am looking for a CS Ph.D. position, my research field is computer vision and remote physiological sensing, and I will graduate with a master's degree in June 2024. If anyone is interested, please send an email to kegangwang@mails.ccnu.edu.cn. 
