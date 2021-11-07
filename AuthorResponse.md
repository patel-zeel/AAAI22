# Extended Author Response

Click on any of the links below to naviagate to corresponding sections.

* [Reviewer 1](https://github.com/ouranonymoussubmission/AAAI22/blob/main/Rebuttal.md#r1)
    * [Q1: Constant Noise](https://github.com/ouranonymoussubmission/AAAI22/blob/main/Rebuttal.md#r1q1)
    * [Q2: Experiment on Non-missing Data](https://github.com/ouranonymoussubmission/AAAI22/blob/main/Rebuttal.md#r1q2)
    * [Q3: Uncertainty Quantification](https://github.com/ouranonymoussubmission/AAAI22/blob/main/Rebuttal.md#r1q3)
    * [Q4: Clarity Needed in Section 4.2, 4.3 and Final Model](https://github.com/ouranonymoussubmission/AAAI22/blob/main/Rebuttal.md#r1q4)
* [Reviewer 2](https://github.com/ouranonymoussubmission/AAAI22/blob/main/Rebuttal.md#r2)
* [Reviewer 3](https://github.com/ouranonymoussubmission/AAAI22/blob/main/Rebuttal.md#r3)
* [Reviewer 4](https://github.com/ouranonymoussubmission/AAAI22/blob/main/Rebuttal.md#r4)
* [London Data Experiment](https://github.com/ouranonymoussubmission/AAAI22/blob/main/Rebuttal.md#london-data-experiment)

## R1
### R1Q1
**Q: Is the assumption that the variance of the noise in (2) or (7) is constant restrictive ? It seems that considering feature-dependent observation variance could provide a more general model.**

A: We  agree  that  constant  noise  can  be  restrictive  based on  nature  of  the  dataset.  However,  a  model  with  input-dependent  noise  is  not *exactly* tractable.  We  are  currently exploring [several approximations](https://icml.cc/Conferences/2011/papers/456_icmlpaper.pdf) that may allow us to model this phenomena efficiently.

### R1Q2
Q: In the data preprocessing the authors highlight the fact that many stations have a lot of missing values, for instance for the pressure feature. They claim that their methodology provides a data-driven, systematic approach to
data filling. Obviously dealing with missing data is required for real-life datasets, but could it be possible to highlight the performance of the proposed method with an additional dataset which does not require to fill so many data (to provide additional guarantees that the preprocessing did not impact the results) ?

A: We totally agree with the comment and thus, we run our models on London Air Quality dataset given in [KDD Cup 2018](https://www.kdd.org/kdd2018/kdd-cup). We found that our model still outperforms all the baselines with multiple metrics. Click [here](https://github.com/ouranonymoussubmission/AAAI22/blob/main/Rebuttal.md#london-data-experiment) to navigate to the further details/results.


---

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







