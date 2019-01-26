# DCSO (Dynamic Combination of Detector Scores for Outlier Ensembles)
### Supplementary materials: datasets, demo source codes and sample outputs.

Y. Zhao and M.K. Hryniewicki, "DCSO: Dynamic Combination of Detector Scores for Outlier Ensembles" *ACM KDD Workshop on Outlier Detection De-constructed (ODD v5.0)*, 2018. 

Please cite the paper as:

    @conference{zhao2018dcso,
        author     = {Zhao, Yue and Hryniewicki, Maciej K},
        title      = {{DCSO:} Dynamic Combination of Detector Scores for Outlier Ensembles},
        booktitle  = {ACM SIGKDD ODD Workshop},
        year       = {2018},
        address    = {London, UK},
        timestamp  = {Mon, 22 Oct 2018 13:07:32 +0200},
    }

    
**[PDF](https://www.andrew.cmu.edu/user/lakoglu/odd/accepted_papers/ODD_v50_paper_3.pdf)** | 
**[Presentation Slides](https://yuezhao.squarespace.com/s/ODD-Zhao-DCSO.pdf)** ]

**Note**: [LSCP](https://github.com/yzhao062/lscp) is an upgraded version of DCSO, which has been accepted at SDM' 19.

------------

Additional notes:
1. Three versions of codes are (going to be) provided:
   1. **Demo version** (demo_lof.py and demo_knn.py) are created for the fast reproduction of the experiment results. The demo version only compares the baseline algorithms with DCSO algorithms. The effect of parameters, e.g., the choice of *k*, are not included.
   2.  **Full version** (tba)  will be released after moderate code cleanup and optimization. In contrast to the demo version, the full version also considers the impact of parameter setting. The full version is therefore relatively slow, which will be further optimized. It is noted the demo version is sufficient to prove the idea. We suggest to using the demo version while playing with DCSO, during the full version is being optimized.
   3. **Production version** (tba) will be released with full optimization and testing as a framework. The purpose of this version is to be used in real applications, which should require fewer dependencies and faster execution.
3. It is understood that there are **small variations** in the results due to the random process, e.g., spliting the training and test sets. Thus, running demo codes would only result in similar results to the paper but not the exactly same results.
------------

##  Introduction
In this paper, an unsupervised outlier detector combination framework called DCSO (Dynamic Combination of Detector Scores for Outlier Ensembles) is proposed, demonstrated and assessed for the dynamic selection of most competent base detectors, with an emphasis on data locality. The proposed DCSO framework first defines the local region of a test instance by its k nearest neighbors and then identifies the top-performing base detectors within the local region.
As classification ensembles, DCSO has two key stages. In the Generation stage, the chosen base detector algorithm is initialized with distinct parameters to build a pool of diversified detectors, and all are then fitted on the entire training dataset. In the Combination stage, DCSO picks the most competent detector in the local region defined by the test instance. Finally, the selected detector is used to predict the outlier score for the test instance.

![Flowchart](https://github.com/yzhao062/DCSO/blob/master/md_figs/flowchart.png)

## Dependency
The experiment codes are writen in Python 3.6 and built on a number of Python packages:
- numpy>=1.13
- scipy>=0.19
- scikit_learn>=0.19

Batch installation is possible using the supplied "requirements.txt" with pip or conda.

------------

## Datasets
Ten datasets are used (see dataset folder):

| Datasets   | #  Points (*n*)  | Dimension (*d*)  | # Outliers  | % Outliers
| ---------- | ---------------- | ---------------- | ----------- |------------|
| Pima 	     | 768	            | 8	               | 268	     | 34.8958    |
| Vowels     | 1456	            | 12               | 50          | 3.4341     |
| Letter	 | 1600             | 32               | 100	     | 6.2500     |
| Cardio     | 1831	            | 21	           | 176         | 9.6122     |
| Thyroid	 | 3772	            | 6	               | 93	         | 2.4655     |
| Satellite	 | 6435	            | 36	           | 2036	     | 31.6394    |
| Pendigits	 | 6870	            | 16	           | 156	     | 2.2707     |
| Annthyroid | 7200	            | 6	               | 534	     | 7.4167     |
| Mnist	     | 7603	            | 100	           | 700	     | 9.2069     |
| Shuttle	 | 49097	        | 9	               | 3511        | 7.1511     |

All datasets are accesible from http://odds.cs.stonybrook.edu/. Citation Suggestion for the datasets please refer to: 
> Shebuti Rayana (2016).  ODDS Library [http://odds.cs.stonybrook.edu]. Stony Brook, NY: Stony Brook University, Department of Computer Science.

To replicate the demo, you should download the datasets from http://odds.cs.stonybrook.edu/ and place them in ./datasets/. We do not provide the data download.

------------

## Usage and Sample Output (Demo Version)
Experiments could be reproduced by running **demo_lof.py** and **demo_knn.py** directly. You could simply download/clone the entire repository and execute the code by 
```bash
python demo_lof.py
```

The difference between **demo_lof.py** and **demo_knn.py** is simply at the base detector choice. Apparently, the former uses LOF as the base detector, while the latter uses *k*NN instead. We introduce two evalution methods:
1.  The area under receiver operating characteristic curve (**ROC**)
2.  Precision at rank m (***P*@*m***) 

The results of **demo_lof.py** and **demo_knn.py**  are presented below. Table 1 and 2 illustrate the results when **LOF** is used as the base detector, while Table 3 and 4 are based when ***k*NN** is used as the base detector. The highest score is highlighted in **bold**, while the lowest is marked with an **asterisk (*)**.

![ LOF_ROC](https://github.com/yzhao062/DCSO/blob/master/md_figs/lof_roc.png)
![ LOF_PRC](https://github.com/yzhao062/DCSO/blob/master/md_figs/lof_prc.png)
![ KNN_ROC](https://github.com/yzhao062/DCSO/blob/master/md_figs/knn_roc.png)
![ KNN_PRC](https://github.com/yzhao062/DCSO/blob/master/md_figs/knn_prc.png)

## Visualizations (based on demo_lof.py )
The figure below visually compares the performance of SG and DCSO methods on **Cardio**, **Thyroid** and **Letter** using t-distributed stochastic neighbor embedding (t-SNE). Normal and outlying points are denoted as **orange dots** and **red squares**, respectively. The normal points that are only correctly detected by SG methods are named SG_N (** green triangle_down**), and only by DCSO are named as DCSO_N (**blue cross sign**). Similarly, outliers are denoted as SG_N (**green triangle_up**) and DCSO_N (**blue plus sign**), given they can only be detected by SG or DCSO methods, respectively.

![ tsne](https://github.com/yzhao062/DCSO/blob/master/md_figs/tsne.png)

Full visulization could be found at [t-SNE](https://github.com/yzhao062/DCSO/tree/master/viz "t-SNE"). To replicate the  visualization, please use "**viz_tsne.py**". It is noted this script is not fully optimized and could be cubersome.
