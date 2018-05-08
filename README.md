# DODC (Dynamic Outlier Detector Combination)
### Supplementary materials: datasets, demo source codes and sample outputs.

Y. Zhao and M.K. Hryniewicki, "DODC: Selecting and Combining Detector Scores Dynamically for Outlier Ensembles" *ACM KDD Workshop on Outlier Detection De-constructed (ODD)*, 2018. **Submitted, under review**.

------------

Additional notes:
1. Three versions of codes are (going to be) provided:
   1. **Demo version** (demo_lof.py and demo_knn) are created for the fast reproduction of the experiment result. The demo version only compares the baseline algorithms with DODC algorithms. The effect of parameters are not included.
   2.  **Full version** (tba)  will be released after moderate code cleanup and optimization. In contrast to the demo version, the full version also considers the impact of parameter setting. However,  the demo version should be sufficient to prove the idea. The full version is therefore relatively slow, which will be further optimized. We suggest to using the demo version while playing with DODC, during the full version is being optimized.
   3. **Production version** (tba) will be released with full optimization and testing. The purpose of this version is to be used in real applications, which requires fewer library dependencies and faster execution.
3. It is understood that there are **small variations** in the results due to the random process, such as spliting the training and test set. Again, running demo code would only give you similar results but not the exact results.
------------

##  Introduction
DODC (Dynamic Outlier Detector Combination) is proposed, demonstrated and assessed for the dynamic selection of most competent base detectors for each test object with an emphasis on data locality. The proposed DODC framework first defines the local region of a test instance by its k nearest neighbors, and then identifies most performing base detectors within the local region.

DODC has two key stages as well. In Generation stage, the chosen base detector method, e.g., LOF, is initialized with distinct parameters to build a pool of diversified detectors, and all are then fitted on the entire training data. In Selection stage, DODC picks the most competent base detector in the local region defined by the test instance. Finally, the selected detector is used to predict the outlier score for the test instance.

![ Flowchart](https://github.com/yzhao062/DODC/blob/master/md_figs/flowchart3.png)

## Dependency
The experiement codes are writted in Python 3 and built on a number of Python packages:
**TBF**

Batch installation is possible using the supplied "requirements.txt"

------------


## Datasets
Five datasets are used (see dataset folder):

|  Datasets | #  Points (*n*)  | Dimension (*d*)  | # Outliers  | % Outliers
| ------------ | ------------ | ------------ | ------------ |------------|
|Pima 	|768	|8	|268	|34.8958|
|Vowels|	1456	|12|	50|	3.4341|
|Letter	|1600|	32|	100	|6.2500|
|Cardio|	1831	|21	|176|	9.6122|
|Thyroid	|3772	|6	|93	|2.4655|
|Satellite	|6435	|36	|2036	|31.6394|
|Pendigits	|6870	|16	|156	|2.2707|
|Annthyroid	|7200	|6	|534	|7.4167|
|Mnist	|7603	|100	|700	|9.2069|
|Shuttle	|49097	|9	|3511|	7.1511|

All datasets are accesible from http://odds.cs.stonybrook.edu/. Citation Suggestion for the datasets please refer to: 
> Shebuti Rayana (2016).  ODDS Library [http://odds.cs.stonybrook.edu]. Stony Brook, NY: Stony Brook University, Department of Computer Science.

------------

## Usage and Sample Output (Demo Version)
Experiments could be reproduced by running **demo_lof.py** and **demo_knn.py** directly. You could simply download/clone the entire repository and execute the code by "python demo_lof.py".

The difference between **demo_lof.py** and **demo_knn.py** is simply at the base detector choice. Apparently, the former uses LOF as the base detector, while the latter uses *k*NN instead,
