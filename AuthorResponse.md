# Extended Author Response

Click on any of the links below to naviagate to corresponding sections.

* [Reviewer 1](https://github.com/ouranonymoussubmission/AAAI22/blob/main/AuthorResponse.md#r1)
    * [Q1: Constant Noise](https://github.com/ouranonymoussubmission/AAAI22/blob/main/AuthorResponse.md#r1q1)
    * [Q2: Experiment on Non-missing Data](https://github.com/ouranonymoussubmission/AAAI22/blob/main/AuthorResponse.md#r1q2)
    * [Q3: Uncertainty Quantification](https://github.com/ouranonymoussubmission/AAAI22/blob/main/AuthorResponse.md#r1q3)
    * [Q4: Clarity Needed in Section 4.2, 4.3 and Final Model](https://github.com/ouranonymoussubmission/AAAI22/blob/main/AuthorResponse.md#r1q4)
* [Reviewer 2](https://github.com/ouranonymoussubmission/AAAI22/blob/main/AuthorResponse.md#r2)
* [Reviewer 3](https://github.com/ouranonymoussubmission/AAAI22/blob/main/AuthorResponse.md#r3)
* [Reviewer 4](https://github.com/ouranonymoussubmission/AAAI22/blob/main/AuthorResponse.md#r4)
* [London Data Experiment](https://github.com/ouranonymoussubmission/AAAI22/blob/main/AuthorResponse.md#london-data-experiment)
* [Evaluation of Models on Beijing Dataset with an extended set of metrics](https://github.com/ouranonymoussubmission/AAAI22/blob/main/AuthorResponse.md#evaluation-of-models-on-beijing-dataset-with-an-extended-set-of-metrics)

## R1
### R1Q1
**Q: Is the assumption that the variance of the noise in (2) or (7) is constant restrictive ? It seems that considering feature-dependent observation variance could provide a more general model.**

We  agree  that  constant  noise  can  be  restrictive  based on  nature  of  the  dataset.  However,  a  model  with  input-dependent  noise  is  not *exactly* tractable.  We  are  currently exploring [several approximations](https://icml.cc/Conferences/2011/papers/456_icmlpaper.pdf) that may allow us to model this phenomena efficiently.

### R1Q2
**Q: In the data preprocessing the authors highlight the fact that many stations have a lot of missing values, for instance for the pressure feature. They claim that their methodology provides a data-driven, systematic approach to
data filling. Obviously dealing with missing data is required for real-life datasets, but could it be possible to highlight the performance of the proposed method with an additional dataset which does not require to fill so many data (to provide additional guarantees that the preprocessing did not impact the results) ?**

We totally agree with the comment and thus, we run our models on London Air Quality dataset given in [KDD Cup 2018](https://www.kdd.org/kdd2018/kdd-cup). We found that our model still outperforms all the baselines with multiple metrics. Click [here](https://github.com/ouranonymoussubmission/AAAI22/blob/main/AuthorResponse.md#london-data-experiment) to navigate to the further details/results.

### R1Q3
**Q: In the introduction, the authors highlight that data-driven approaches do not quantify uncertainty. The proposed method, based on Gaussian processes which are well understood in statistics could overcome this difficulty. However the authors only report RMSE on their predictions. Is this possible to provide confidence intervals on the prediction to quantify uncertainty ?**

* We have updated Figure 3 in the paper to show the confidence intervals on the predictions . Here we show a comparison between old figure (on left) and updated figure (on right)

|Old figure| Updated Figure|
|:-:|:-:|
|![image](https://user-images.githubusercontent.com/72247818/140639614-f8078d40-f600-4151-b512-f4e566ebfa67.png)| ![image](https://user-images.githubusercontent.com/72247818/140639645-809c72e1-cb13-4913-84e4-e665f1c898cb.png)|



* We have provided the results with an extended set of metrics on [Beijing dataset](https://github.com/ouranonymoussubmission/AAAI22/blob/main/AuthorResponse.md#evaluation-of-models-on-beijing-dataset-with-an-extended-set-of-metrics) and [London dataset](https://github.com/ouranonymoussubmission/AAAI22/blob/main/AuthorResponse.md#results). We also explain the probabilistic metrics briefly [here](https://github.com/ouranonymoussubmission/AAAI22/blob/main/AuthorResponse.md#probabilistic-metrics).

### R2Q4
**Q: The overall description of the kernels is sometimes hard to follow (I believe for instance that length scale in Section 4.2 is not clearly defined, Section 4.3 is a bit sketchy). Maybe after Section 4.6 the authors could also write explicitly the final proposed model ?**

* We agree with this comment and we have revised the confusing notations in the manucript. We have also added a notation table in [extended appendix](https://github.com/ouranonymoussubmission/AAAI22/blob/main/Appendix_v2.pdf). A snapshot of which is given below.

![image](https://user-images.githubusercontent.com/72247818/140640881-dac9a96e-b49c-47f8-957b-9b3eb880b676.png)

* Following is the snapshot of final model description from updated manuscript.

![image](https://user-images.githubusercontent.com/72247818/140642296-5a562899-b0ca-4e15-b770-d2dc8b7ea255.png)



### Probabilistic metrics

* Negative Log Predictive Density (NLPD)

Log Predictive Density (LPD) is a log probability of sampling the test data given the posterior predictive distribution (higher is better). Negative of LPD is generally known as NLPD metric (lower is better). A set of work uses NLPD as a metric to evaluate probabilistic models [[LLS GP](https://link.springer.com/chapter/10.1007/978-3-540-87481-2_14), [KCN](https://ojs.aaai.org/index.php/AAAI/article/view/5716), [ICML21](https://arxiv.org/pdf/2102.08314.pdf)].

* Mean Standardized Log Loss (MSLL)

Standardized Log Loss (LL) is log probability of sampling a test data point given univariate normal distribution with mean as predictive mean and variance as predictive variance for the data point. An average of this metric over all the data points is called as Mean Standardized Log Loss (MSLL).

* [CE (Coverage Error)](https://arxiv.org/pdf/1710.01720v1.pdf)

A predictive distribution is called well-calibrated if x% of samples lie within x% confidence interval. For example, 95% of the test samples should lie within 95% confidence interval. Coverage error for x% confidence interval is an absolute difference between (x/100) and fraction of samples falling within x% confidence interval. For more details, please visit the reference.

### Evaluation of Models on Beijing Dataset with an extended set of metrics

<details open>
<summary>Root Mean Squared Error (lower is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Fold-3 | Mean |
| :- | -:| -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖	| 5.06 | 4.32 | 5.61	| 4.45 | 4.86 |
|ARD ✔ N ✖ Cat.✖ Per. ✖	| **4.82** | 3.99 | 5.57	| 4.18 | **4.64** |
|ARD ✔ N ✖ Cat.✖ Per. ✔	| 4.83 | **3.99** | 5.60	| **4.16** | 4.65 |
|RF	                    | 5.16 | 4.64 | **4.78** | 4.17 | 4.69 |
|IDW	                    | 7.79 | 7.48 | 8.61 | 8.15 | 8.01 |
|KNN	                    | 5.32 | 4.38 | 5.10 | 4.18 | 4.75 |
|XGB	                    | 5.50 | 4.57 | 4.84 | 4.70 | 4.90 |
|ADAIN	                 | 5.26 | 4.41 | 4.95 | 4.51 | 4.78 |

   
</details>

<details>
<summary>Mean Absolute Error (lower is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Fold-3 | Mean |
| :- | -:| -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖	 | 3.07 | 2.93 | 3.92 | 3.38 | 3.33 |
|ARD ✔ N ✖ Cat.✖ Per. ✖	 | **2.87** | **2.76** | 3.92 | 3.19 | **3.19** |
|ARD ✔ N ✖ Cat.✖ Per. ✔	 | 2.88 | 2.76 | 3.98 | 3.19 | 3.20 |
|RF	                     | 2.88 | 2.76 | 3.98 | 3.19 | 3.20 |
|IDW	                     | 2.88 | 2.76 | 3.98 | 3.19 | 3.20 |
|KNN	                     | 3.25 | 3.16 | **3.25** | **3.15** | 3.20 |
|XGB	                     | 3.52 | 3.31 | 3.32 | 3.62 | 3.44 |
|ADAIN	                  |      |      |      |      |      |

</details>

<details>
<summary>Mean Absolute Percentage Error (lower is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Fold-3 | Mean |
| :- | -:| -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖	 |34% |29% | 87% |38% |47% |
|ARD ✔ N ✖ Cat.✖ Per. ✖	 | 32% |27% |88% |35% |46% |
|ARD ✔ N ✖ Cat.✖ Per. ✔	 | **32%** | **27%** |89% |**35%** |46% |
|RF	                     |32% |30% |**71%** |36% |**42%**|
|IDW	                     |51% |46% |131% |57% |71% |
|KNN	                     |32% |31% |78% |35% |44% |
|XGB	                     |40% |35% |78% |40% |48% |
|ADAIN	                  |      |      |      |      |      |

</details>

<details>
<summary>R^2 Score (higher is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Fold-3 | Mean |
| :- | -:| -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖ |0.66 | 0.73 | 0.53 | 0.71 | 0.66|
|ARD ✔ N ✖ Cat.✖ Per. ✖ |**0.69** | **0.77** | 0.54 | **0.74** | **0.69**|
|ARD ✔ N ✖ Cat.✖ Per. ✔ |**0.69** | **0.77** | **0.53** | **0.74** | 0.68|
|RF                        |0.64 | 0.69 | 0.66 | 0.74 | 0.68|
|IDW                     |0.19 | 0.20 | -0.10 | 0.02 | 0.08|
|KNN                     |0.62 | 0.73 | 0.61 | 0.74 | 0.68|
|XGB                     |0.60 | 0.70 | 0.65 | 0.67 | 0.66|
|ADAIN	                  |      |      |      |      |      |

</details>

<details>
<summary>Negative Log Predictive Density (lower is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Fold-3 | Mean |
| :- | -:| -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖ |**12,905.88** | 11,909.78 | 10,528.24 | 10,347.43| 11,422.83 |
|ARD ✔ N ✖ Cat.✖ Per. ✖ |13,046.91 | 11,691.14 | **10,261.17** | 9,994.86| **11,248.52** |
|ARD ✔ N ✖ Cat.✖ Per. ✔ |13,226.03 | **11,659.61** | 10,284.22 | **9,926.65**| 11,274.13 |
|RF                        |- | - | - | -|
|IDW                     |- | - | - | -|
|KNN                     |- | - | - | -|
|XGB                     |- | - | - | -|
|ADAIN	                 |- | - | - | -|

</details>

<details>
<summary>Mean Standardized Log Loss (lower is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Fold-3 | Mean |
| :- | -:| -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖ |3.25 | 2.92 | **3.39** | 3.04 | 3.15|
|ARD ✔ N ✖ Cat.✖ Per. ✖ |**3.21** | **2.83** | 3.42 | 2.94 | **3.10**|
|ARD ✔ N ✖ Cat.✖ Per. ✔ |3.28 | 2.84 | 3.42 | **2.93** | 3.12|
|RF                        |- | - | - | -|
|IDW                     |- | - | - | -|
|KNN                     |- | - | - | -|
|XGB                     |- | - | - | -|
|ADAIN                     |- | - | - | -|
</details>

<details>
<summary>Coverage Error (68% or 1 standard deviation) (lower is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Fold-3 | Mean |
| :- | -:| -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖ |0.02 | 0.02 | 0.10 | 0.12 | 0.07|
|ARD ✔ N ✖ Cat.✖ Per. ✖ |0.03 | 0.03 | 0.11 | 0.10 | 0.07|
|ARD ✔ N ✖ Cat.✖ Per. ✔ |**0.01** | **0.02** | **0.10** | **0.10** | **0.06**|
|RF                        |- | - | - | -|
|IDW                     |- | - | - | -|
|KNN                     |- | - | - | -|
|XGB                     |- | - | - | -|
|ADAIN                     |- | - | - | -|
</details>

<details>
<summary>Coverage Error (95% or 2 standard deviation) (lower is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Fold-3 | Mean |
| :- | -:| -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖ |0.04 | 0.03 | 0.09 | 0.09 | 0.06|
|ARD ✔ N ✖ Cat.✖ Per. ✖ |0.04 | 0.03 | 0.12 | 0.07 | 0.06|
|ARD ✔ N ✖ Cat.✖ Per. ✔ |0.05 | 0.03 | 0.12 | 0.07 | 0.07|
|RF                        |- | - | - | -|
|IDW                     |- | - | - | -|
|KNN                     |- | - | - | -|
|XGB                     |- | - | - | -|
|ADAIN                     |- | - | - | -|
</details>

<details>
<summary>Coverage Error (99% or 3 standard deviation) (lower is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Fold-3 | Mean |
| :- | -:| -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖ |0.04 | 0.02 | 0.06 | 0.03 | 0.04|
|ARD ✔ N ✖ Cat.✖ Per. ✖ |0.03 | 0.02 | 0.06 | 0.02 | 0.03|
|ARD ✔ N ✖ Cat.✖ Per. ✔ |0.03 | 0.02 | 0.05 | 0.02 | 0.03|
|RF                        |- | - | - | -|
|IDW                     |- | - | - | -|
|KNN                     |- | - | - | -|
|XGB                     |- | - | - | -|
|ADAIN                     |- | - | - | -|
</details>

### London Data Experiment
#### Preprocessing
In KDD Cup 2018 challenge, London air quality dataset (PM2.5, Latitude and Longitude) is provided along with the grid-wise meteorological data of the London city. To map the meteorological data (Temperature, Pressure, Humidity, Wind direction and Wind speed) with each air quality station, we take a distance-based weighted average (closer the station, higher the weight) of the nearest 4 grid points to the station. 

In our experiments, we choose a month with the least amount of \% missing entries of the AQ variable (PM2.5) (May 2017 consists of 10.8\% missing PM2.5 entries). Note that there are no missing entries in meteorological features. Further, to address the missing entries in the 'May' London air quality dataset, we investigate the stations having a substantial amount of missing values. 2 of the 24 stations ('HR1', 'KF1') from the dataset are having 85\% of the missing PM2.5 values, and thus we remove those stations. After this step, only 1.5% entries are missing in the dataset. To fill in the missing data, we use similar method to the Beijing dataset. Also, unlike Beijing, the London AQ dataset consists only of non-categorical features, thus excluding the (ANCL) configurations with categorical kernel.

#### Experimental Setup
We perform 4-fold cross-validation by splitting the train and test sets based on stations. 

#### Experimental Configuration

| Configuration | ✖ | ✔ |
| :-: | :-: | :-: |
| ARD | ARD disabled | ARD enabled |
| N | Stationary kernel | Non-stationary kernel |
| Cat. | Using RBF/Matern kernel for categorical features | Using Hamming distance kernel for categorical features |
| Per. | Using RBF/Matern kernel for the time feature | Using Local Periodic kernel for the time feature |

#### Results

<details open>
<summary>Root Mean Squared Error (lower is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Fold-3 | Mean |
| :- | -:| -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖	| 5.06 | 4.32 | 5.61	| 4.45 | 4.86 |
|ARD ✔ N ✖ Cat.✖ Per. ✖	| **4.82** | 3.99 | 5.57	| 4.18 | **4.64** |
|ARD ✔ N ✖ Cat.✖ Per. ✔	| 4.83 | **3.99** | 5.60	| **4.16** | 4.65 |
|RF	                    | 5.16 | 4.64 | **4.78** | 4.17 | 4.69 |
|IDW	                    | 7.79 | 7.48 | 8.61 | 8.15 | 8.01 |
|KNN	                    | 5.32 | 4.38 | 5.10 | 4.18 | 4.75 |
|XGB	                    | 5.50 | 4.57 | 4.84 | 4.70 | 4.90 |
|ADAIN	                 | 5.26 | 4.41 | 4.95 | 4.51 | 4.78 |

   
</details>

<details>
<summary>Mean Absolute Error (lower is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Fold-3 | Mean |
| :- | -:| -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖	 | 3.07 | 2.93 | 3.92 | 3.38 | 3.33 |
|ARD ✔ N ✖ Cat.✖ Per. ✖	 | **2.87** | **2.76** | 3.92 | 3.19 | **3.19** |
|ARD ✔ N ✖ Cat.✖ Per. ✔	 | 2.88 | 2.76 | 3.98 | 3.19 | 3.20 |
|RF	                     | 2.88 | 2.76 | 3.98 | 3.19 | 3.20 |
|IDW	                     | 2.88 | 2.76 | 3.98 | 3.19 | 3.20 |
|KNN	                     | 3.25 | 3.16 | **3.25** | **3.15** | 3.20 |
|XGB	                     | 3.52 | 3.31 | 3.32 | 3.62 | 3.44 |
|ADAIN	                  |      |      |      |      |      |

</details>

<details>
<summary>Mean Absolute Percentage Error (lower is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Fold-3 | Mean |
| :- | -:| -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖	 |34% |29% | 87% |38% |47% |
|ARD ✔ N ✖ Cat.✖ Per. ✖	 | 32% |27% |88% |35% |46% |
|ARD ✔ N ✖ Cat.✖ Per. ✔	 | **32%** | **27%** |89% |**35%** |46% |
|RF	                     |32% |30% |**71%** |36% |**42%**|
|IDW	                     |51% |46% |131% |57% |71% |
|KNN	                     |32% |31% |78% |35% |44% |
|XGB	                     |40% |35% |78% |40% |48% |
|ADAIN	                  |      |      |      |      |      |

</details>

<details>
<summary>R^2 Score (higher is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Fold-3 | Mean |
| :- | -:| -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖ |0.66 | 0.73 | 0.53 | 0.71 | 0.66|
|ARD ✔ N ✖ Cat.✖ Per. ✖ |**0.69** | **0.77** | 0.54 | **0.74** | **0.69**|
|ARD ✔ N ✖ Cat.✖ Per. ✔ |**0.69** | **0.77** | **0.53** | **0.74** | 0.68|
|RF                        |0.64 | 0.69 | 0.66 | 0.74 | 0.68|
|IDW                     |0.19 | 0.20 | -0.10 | 0.02 | 0.08|
|KNN                     |0.62 | 0.73 | 0.61 | 0.74 | 0.68|
|XGB                     |0.60 | 0.70 | 0.65 | 0.67 | 0.66|
|ADAIN	                  |      |      |      |      |      |

</details>

<details>
<summary>Negative Log Predictive Density (lower is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Fold-3 | Mean |
| :- | -:| -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖ |**12,905.88** | 11,909.78 | 10,528.24 | 10,347.43| 11,422.83 |
|ARD ✔ N ✖ Cat.✖ Per. ✖ |13,046.91 | 11,691.14 | **10,261.17** | 9,994.86| **11,248.52** |
|ARD ✔ N ✖ Cat.✖ Per. ✔ |13,226.03 | **11,659.61** | 10,284.22 | **9,926.65**| 11,274.13 |
|RF                        |- | - | - | -|
|IDW                     |- | - | - | -|
|KNN                     |- | - | - | -|
|XGB                     |- | - | - | -|
|ADAIN	                 |- | - | - | -|

</details>

<details>
<summary>Mean Standardized Log Loss (lower is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Fold-3 | Mean |
| :- | -:| -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖ |3.25 | 2.92 | **3.39** | 3.04 | 3.15|
|ARD ✔ N ✖ Cat.✖ Per. ✖ |**3.21** | **2.83** | 3.42 | 2.94 | **3.10**|
|ARD ✔ N ✖ Cat.✖ Per. ✔ |3.28 | 2.84 | 3.42 | **2.93** | 3.12|
|RF                        |- | - | - | -|
|IDW                     |- | - | - | -|
|KNN                     |- | - | - | -|
|XGB                     |- | - | - | -|
|ADAIN                     |- | - | - | -|
</details>

<details>
<summary>Coverage Error (68% or 1 standard deviation) (lower is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Fold-3 | Mean |
| :- | -:| -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖ |0.02 | 0.02 | 0.10 | 0.12 | 0.07|
|ARD ✔ N ✖ Cat.✖ Per. ✖ |0.03 | 0.03 | 0.11 | 0.10 | 0.07|
|ARD ✔ N ✖ Cat.✖ Per. ✔ |**0.01** | **0.02** | **0.10** | **0.10** | **0.06**|
|RF                        |- | - | - | -|
|IDW                     |- | - | - | -|
|KNN                     |- | - | - | -|
|XGB                     |- | - | - | -|
|ADAIN                     |- | - | - | -|
</details>

<details>
<summary>Coverage Error (95% or 2 standard deviation) (lower is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Fold-3 | Mean |
| :- | -:| -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖ |0.04 | 0.03 | 0.09 | 0.09 | 0.06|
|ARD ✔ N ✖ Cat.✖ Per. ✖ |0.04 | 0.03 | 0.12 | 0.07 | 0.06|
|ARD ✔ N ✖ Cat.✖ Per. ✔ |0.05 | 0.03 | 0.12 | 0.07 | 0.07|
|RF                        |- | - | - | -|
|IDW                     |- | - | - | -|
|KNN                     |- | - | - | -|
|XGB                     |- | - | - | -|
|ADAIN                     |- | - | - | -|
</details>

<details>
<summary>Coverage Error (99% or 3 standard deviation) (lower is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Fold-3 | Mean |
| :- | -:| -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖ |0.04 | 0.02 | 0.06 | 0.03 | 0.04|
|ARD ✔ N ✖ Cat.✖ Per. ✖ |0.03 | 0.02 | 0.06 | 0.02 | 0.03|
|ARD ✔ N ✖ Cat.✖ Per. ✔ |0.03 | 0.02 | 0.05 | 0.02 | 0.03|
|RF                        |- | - | - | -|
|IDW                     |- | - | - | -|
|KNN                     |- | - | - | -|
|XGB                     |- | - | - | -|
|ADAIN                     |- | - | - | -|
</details>







