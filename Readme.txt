# Cross-View Kernel Collaborative Representation Classifier

This package includes source codes of our CV-KCRC work " Guoqing Zhang, Tong Jiang, Junchuan Yang, Jiang Xu, Yuhui Zheng [Cross-View Kernel Collaborative Representation Classification for Person Re-identification] ”

Created by [Tong Jiang], on April 24, 2020.

##Summary
In this package, you find the MATLAB code for the following paper:
Guoqing Zhang, Tong Jiang, Junchuan Yang, Jiang Xu, Yuhui Zheng. Cross-View Kernel Collaborative Representation Classification for Person Re-identification. 

##Demos
One demo is available for reproducing the results.
- demo_ viper.m : perform evaluation over VIPeR dataset

##Remarks
- If you run demo_viper.m for the first time, you need to uncomment the “ % setup_viper “ line to download and prepare the dataset. Make sure you are connected to the network. 
- To run for other datasets, insert the features in .\data\ folder and do the modifications in the code.
- The datasets can be downloaded below:
    - VIPeR: https://vision.soe.ucsc.edu/node/178
    - CUHK01: http://www.ee.cuhk.edu.hk/~rzhao/
    - PRID450_S: http://lrs.icg.tugraz.at/download.php
    - GRID: http://www.eecs.qmul.ac.uk/~ccloy/downloads_qmul_underground_reid.html
    - CUHK03: http://www.ee.cuhk.edu.hk/~rzhao/
- Parallel Toolbox can accellerate the computation, use matlabpool if necessary
- This demo was tested on MATLAB (R2018a), 64-bit Win10, Intel Xeon 3.30 GHz CPU
- Memory cost:
  - Running demo_viper.m would consume around 2.0 GB memory
