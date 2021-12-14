# Extended Author Response

Click on any of the links below to naviagate to corresponding sections.

* [Reviewer 1](https://github.com/ouranonymoussubmission/AAAI22/blob/main/AuthorResponse.md#r1)
* [Reviewer 2](https://github.com/ouranonymoussubmission/AAAI22/blob/main/AuthorResponse.md#r2)
* [Reviewer 3](https://github.com/ouranonymoussubmission/AAAI22/blob/main/AuthorResponse.md#r3)
* [Reviewer 4](https://github.com/ouranonymoussubmission/AAAI22/blob/main/AuthorResponse.md#r4)
* [Probalistic metrics](https://github.com/ouranonymoussubmission/AAAI22/blob/main/AuthorResponse.md#probabilistic-metrics)
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

### R1Q4
**Q: The overall description of the kernels is sometimes hard to follow (I believe for instance that length scale in Section 4.2 is not clearly defined, Section 4.3 is a bit sketchy). Maybe after Section 4.6 the authors could also write explicitly the final proposed model ?**

* We agree with this comment and we have revised the confusing notations in the manucript. We have also added a notation table in [extended appendix](https://github.com/ouranonymoussubmission/AAAI22/blob/main/Appendix_v2.pdf). A snapshot of which is given below.

![image](https://user-images.githubusercontent.com/72247818/140640881-dac9a96e-b49c-47f8-957b-9b3eb880b676.png)

* Following is the snapshot of final model description from updated manuscript.

![image](https://user-images.githubusercontent.com/72247818/140642296-5a562899-b0ca-4e15-b770-d2dc8b7ea255.png)

## R2
### R2Q1
**Q: The notations in the paper are very confusing. For example:**

**In the Problem statement section, T is defined as the number of time-stamps. However, it seems that T means the target air quality station as well.**

**The paper misuses bold, non-bold, uppercase, and lowercase symbols without explicit definition. For example, what is the difference between \mathbf{x}_i and x_i. What does f(X) mean in Fig. 2?**

**A possible solution: adding a table to clarify all the used notations to increase the readability.**

* We have updated the notations in Section B in [extended appendix](https://github.com/ouranonymoussubmission/AAAI22/blob/main/Appendix_v2.pdf) as shown [here](https://github.com/ouranonymoussubmission/AAAI22/blob/main/AuthorResponse.md#r1q4)
* We have updated corresponding figure as the following:

|Old figure| Updated Figure|
|:-:|:-:|
|![image](https://user-images.githubusercontent.com/72247818/140643198-256663c7-dbb4-4e17-9138-f576259780ce.png)| ![image](https://user-images.githubusercontent.com/72247818/140643188-c4242b67-cb42-40ec-ad3d-7957d408afd7.png)|

* We have updated Figure 2 as the following:

|Old figure| Updated Figure|
|:-:|:-:|
|![image](https://user-images.githubusercontent.com/72247818/140643311-1583b85d-d20a-4fa8-b200-11294d35affa.png)| ![image](https://user-images.githubusercontent.com/72247818/140643325-2b161bb8-f177-4c24-98a6-94ccd0b537de.png)|


### R2Q2
**Q: This paper omits the derivation of formulas like Eq. 4, 8, 9, 10. Despite the page limit, it would be good to mention them in the appendix.**

We have added the derivations in Sections C and D in the [updated appendix](https://github.com/ouranonymoussubmission/AAAI22/blob/main/Appendix_v2.pdf).

### R2Q3
**Q:  Details of the automatic relevance determination are missing. The paper describes the idea of this module but does not give more details.**

We have added detailed introduction to ARD in Section E in the [updated appendix](https://github.com/ouranonymoussubmission/AAAI22/blob/main/Appendix_v2.pdf).

### R2Q4
**Q: This paper claims that the proposed approach implicitly provides uncertainty estimates along with predictions. However, there is no justification e.g., by either theoretical or experimental evaluation. To me, this claim is not convincing enough.**

Please see the response [here](https://github.com/ouranonymoussubmission/AAAI22/blob/main/AuthorResponse.md#r1q3)

### R2Q5
**Q: The paper utilizes the Beijing air quality dataset for the experiments. It would good to evaluate the proposed model in more datasets, such as London air quality dataset (available at the KDD-18 cup website).**

Please see the response [here](https://github.com/ouranonymoussubmission/AAAI22/blob/main/AuthorResponse.md#r1q2)

### R2Q6
**Q:  The authors are encouraged to add more evaluation metrics. To the best of my knowledge, the common metrics in this application include RMSE, MAE, MAPE, R^2, etc. They help us evaluate the approaches from different views.**

We have provided the results with an extended set of metrics on [Beijing dataset](https://github.com/ouranonymoussubmission/AAAI22/blob/main/AuthorResponse.md#evaluation-of-models-on-beijing-dataset-with-an-extended-set-of-metrics) and [London dataset](https://github.com/ouranonymoussubmission/AAAI22/blob/main/AuthorResponse.md#results).

### R2Q7
**Q: Some important baselines/related works are missing. As far as I know, there are some papers published at top venues also focusing on the spatio-temporal imputation problem. These studies can definitely be applied to the air quality inference task, such as Kriging Convolutional Networks (KCN), Inductive Graph Neural Networks for Spatiotemporal Kriging (IGNNK), and GP-based methods like SGPRN (Scalable Gaussian Process Regression Networks). To improve its significance, comparing the proposed model with them is needed.**

We have compared our model with the suggested KCN (kcn_sage) baseline on the Beijing dataset. With a non-exhaustive but sufficient hyperparameter tuning, we found that KCN achieves train and test RMSE of 25 and 45 compared to 10 and 25 for our models. We have also tried to run Scalable Gaussian Process Regression Networks (SGPRN) on our dataset, but it does not scale for our dataset size. Scalability in SGPRN corresponds to dimensions of multi-output observations. Also, we find that authors have tested their code on 450 datapoints compared to >14K datapoints in our case. Given these concerns with suggested baselines, we do not yet compare with them in the main table but will consult the respective authors to understand these better and potentially include comparisons in the camera-ready.

## R3
### R3Q1
**Q: Can you provide an intuition to the non stationary model under-performing?**

Using probabilistic metrics, we found that non-stationary models perform better than the stationary models on Beijing and London datasets. In the future, we plan to selectively add non-stationarity tobalance between flexibility and model performance.
 
### R3Q2
**Q: Would other applications benefit from your approach? if so which part?**

Our model can be used as a generic regression model but is especially suited for non-stationary environmental phenomena like rainfall, temperature, etc. We highlight a way to handle categorical features in a GP model and leverage batch training which is a recent advancement in GP based methods overcoming a significant hurdle (O(n^3) time-complexity) in using GPs for practical purposes.

## R4
### R4Q1
**Q: How would you address some of the data requirements mentioned in the paper?**

To extend our experiments to larger datasets, we can either omit the baselines requiring contiguous datasets or fill the missing data accurately. We are investigating time-series models to interpolate the missing data accurately. As such, our proposed techniques are general and can be applied immediately to various cities having air quality monitoring.

## Probabilistic metrics

* Negative Log Predictive Density (NLPD)

Log Predictive Density (LPD) is a log probability of sampling the test data given the posterior predictive distribution (higher is better). Negative of LPD is generally known as NLPD metric (lower is better). A set of work uses NLPD as a metric to evaluate probabilistic models [[LLS GP](https://link.springer.com/chapter/10.1007/978-3-540-87481-2_14), [KCN](https://ojs.aaai.org/index.php/AAAI/article/view/5716), [ICML21](https://arxiv.org/pdf/2102.08314.pdf)].

* Mean Standardized Log Loss (MSLL)

Standardized Log Loss (LL) is negative log probability of sampling a test data point given univariate normal distribution with mean as predictive mean and variance as predictive variance for the data point. An average of this metric over all the data points is called as Mean Standardized Log Loss (MSLL).

* [CE (Coverage Error)](https://arxiv.org/pdf/1710.01720v1.pdf)

A predictive distribution is called well-calibrated if x% of samples lie within x% confidence interval. For example, 95% of the test samples should lie within 95% confidence interval. Coverage error for x% confidence interval is an absolute difference between (x/100) and fraction of samples falling within x% confidence interval. For more details, please visit the reference.

## Evaluation of Models on Beijing Dataset with an extended set of metrics

<details open>
<summary>Root Mean Squared Error (lower is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Mean |
| :- | -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖ |37.25 | 39.86 | 36.73 | 37.95|
|ARD ✔ N ✖ Cat.✖ Per. ✖ |23.63 | 25.52 | 25.97 | 25.04|
|ARD ✔ N ✖ Cat.✔ Per. ✖ |22.24 | 24.74 | 25.10 | 24.03|
|ARD ✔ N ✖ Cat.✔ Per. ✔ |22.60 | 24.96 | 25.27 | 24.28|
|ARD ✔ N ✔ Cat.✔ Per. ✖ |24.09|27.83|26.30|26.07|
|ARD ✔ N ✔ Cat.✔ Per. ✔ |23.48|25.18|25.47|24.71|
|RF                        |24.86 | 29.78 | 25.4 | 26.68|
|IDW                     |49.11 | 50 | 45.18 | 48.1|
|KNN                     |38.09 | 38.85 | 37.02 | 37.99|
|XGB                     |34.07 | 34.23 | 33.25 | 33.85|
|ADAIN                     |31.86 | 28.39 | 27.91 | 29.39|

   
</details>

<details>
<summary>Mean Absolute Error (lower is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Mean |
| :- | -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖ |25.85 | 29.51 | 24.61 | 26.66|
|ARD ✔ N ✖ Cat.✖ Per. ✖ |14.28 | 18.26 | 14.88 | 15.81|
|ARD ✔ N ✖ Cat.✔ Per. ✖ |13.06 | 17.35 | 14.69 | 15.03|
|ARD ✔ N ✖ Cat.✔ Per. ✔ |13.37 | 17.48 | 14.85 | 15.23|
|ARD ✔ N ✔ Cat.✔ Per. ✖ | 14.36 | 17.87 | 24.69 | 18.97 |
|ARD ✔ N ✔ Cat.✔ Per. ✔ | 14.91 | 17.09 | 15.31 | 15.77 |
|RF                        |14.16 | 17.96 | 14.64 | 15.59|
|IDW                     |34.79 | 38.35 | 31.24 | 34.79|
|KNN                     |24.57 | 26.41 | 23.42 | 24.8|
|XGB                     |24.07 | 24.4 | 23.36 | 23.94|
|ADAIN                     |19.37 | 18.34 | 18.78 | 18.83|

</details>

<details>
<summary>Mean Absolute Percentage Error (lower is better)</summary>

In range 0 to 1.

| Model | Fold-0 | Fold-1 | Fold-2 | Mean |
| :- | -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖ |0.68 | 1.18 | 0.64 | 0.83|
|ARD ✔ N ✖ Cat.✖ Per. ✖ |0.30 | 0.71 | 0.31 | 0.44|
|ARD ✔ N ✖ Cat.✔ Per. ✖ |0.27 | 0.69 | 0.30 | 0.42|
|ARD ✔ N ✖ Cat.✔ Per. ✔ |0.28 | 0.69 | 0.31 | 0.43|
|ARD ✔ N ✔ Cat.✔ Per. ✖ | 0.30 | 0.67 | 0.93 | 0.64 |
|ARD ✔ N ✔ Cat.✔ Per. ✔ | 0.32 | 0.60 | 0.35 | 0.42 |
|RF                        |0.29 | 0.48 | 0.3 | 0.36|
|IDW                     |1.04 | 1.73 | 1.07 | 1.28|
|KNN                     |0.55 | 0.82 | 0.53 | 0.63|
|XGB                     |0.61 | 0.79 | 0.69 | 0.70|
|ADAIN                     |0.56 | 0.44 | 0.44 | 0.48|

</details>

<details>
<summary>R^2 Score (higher is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Mean |
| :- | -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖ |0.77 | 0.69 | 0.77 | 0.74|
|ARD ✔ N ✖ Cat.✖ Per. ✖ |0.91 | 0.87 | 0.89 | 0.89|
|ARD ✔ N ✖ Cat.✔ Per. ✖ |0.92 | 0.88 | 0.89 | 0.9|
|ARD ✔ N ✖ Cat.✔ Per. ✔ |0.91 | 0.88 | 0.89 | 0.89|
|ARD ✔ N ✔ Cat.✔ Per. ✖  | 0.90 | 0.87 | 0.78 | 0.85 |
|ARD ✔ N ✔ Cat.✔ Per. ✔  | 0.90 | 0.88 | 0.89 | 0.89 |
|RF                        |0.90 | 0.83 | 0.89 | 0.87|
|IDW                     |0.60 | 0.51 | 0.65 | 0.59|
|KNN                     |0.76 | 0.71 | 0.77 | 0.75|
|XGB                     |0.81 | 0.77 | 0.81 | 0.80|
|ADAIN                     |0.48 | 0.58 | 0.61 | 0.56|

</details>

<details>
<summary>Negative Log Predictive Density (lower is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Mean |
| :- | -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖ |363,785.63 | 355,416.47 | 336,014.44 | 351,738.84|
|ARD ✔ N ✖ Cat.✖ Per. ✖ |166,680.38 | 176,821.81 | 167,478.70 | 170,326.96|
|ARD ✔ N ✖ Cat.✔ Per. ✖ |149,838.16 | 155,413.97 | 157,293.14 | 154,181.76|
|ARD ✔ N ✖ Cat.✔ Per. ✔ |134,164.84 | 140,730.25 | 138,787.03 | 137,894.04|
|ARD ✔ N ✔ Cat.✔ Per. ✖ | 57754.82 | 53177.94 | 58549.66 | 56494.14 |
|ARD ✔ N ✔ Cat.✔ Per. ✔ | 51781.70 | 50615.99 | 50824.95 | 51074.22 |
|RF                        |- | - | - | -|
|IDW                     |- | - | - | -|
|KNN                     |- | - | - | -|
|XGB                     |- | - | - | -|
|ADAIN                     |- | - | - | -|

</details>

<details>
<summary>Mean Standardized Log Loss (lower is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Mean |
| :- | -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖ |47.89 | 52.77 | 46.45 | 49.04|
|ARD ✔ N ✖ Cat.✖ Per. ✖ |26.64 | 31.47 | 30.08 | 29.40|
|ARD ✔ N ✖ Cat.✔ Per. ✖ |24.58 | 28.75 | 30.27 | 27.87|
|ARD ✔ N ✖ Cat.✔ Per. ✔ |21.96 | 25.84 | 26.26 | 24.69|
|ARD ✔ N ✔ Cat.✔ Per. ✖ | 9.28 | 8.99 | 13.77 | 10.68 |
|ARD ✔ N ✔ Cat.✔ Per. ✔ | 7.68 | 8.13 | 8.48 | 8.09 |
|RF                        |- | - | - | -|
|IDW                     |- | - | - | -|
|KNN                     |- | - | - | -|
|XGB                     |- | - | - | -|
|ADAIN                     |- | - | - | -|
</details>

<details>
<summary>Coverage Error (68% or 1 standard deviation) (lower is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Mean |
| :- | -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖ |0.57 | 0.59 | 0.55 | 0.57|
|ARD ✔ N ✖ Cat.✖ Per. ✖ |0.46 | 0.53 | 0.45 | 0.48|
|ARD ✔ N ✖ Cat.✔ Per. ✖ |0.45 | 0.52 | 0.45 | 0.47|
|ARD ✔ N ✖ Cat.✔ Per. ✔ |0.43 | 0.51 | 0.44 | 0.46|
|ARD ✔ N ✔ Cat.✔ Per. ✖| 0.3 | 0.37 | 0.48 | 0.38 |
|ARD ✔ N ✔ Cat.✔ Per. ✔| 0.25 | 0.33 | 0.25 | 0.28 |
|RF                        |- | - | - | -|
|IDW                     |- | - | - | -|
|KNN                     |- | - | - | -|
|XGB                     |- | - | - | -|
|ADAIN                     |- | - | - | -|

</details>

<details>
<summary>Coverage Error (95% or 2 standard deviation) (lower is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Mean |
| :- | -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖ |0.73 | 0.77 | 0.69 | 0.73|
|ARD ✔ N ✖ Cat.✖ Per. ✖ |0.54 | 0.67 | 0.54 | 0.58|
|ARD ✔ N ✖ Cat.✔ Per. ✖ |0.51 | 0.65 | 0.53 | 0.57|
|ARD ✔ N ✖ Cat.✔ Per. ✔ |0.50 | 0.64 | 0.51 | 0.55|
 |ARD ✔ N ✔ Cat.✔ Per. ✖| 0.32 | 0.40 | 0.55 | 0.42 |
|ARD ✔ N ✔ Cat.✔ Per. ✔| 0.28 | 0.35 | 0.27 | 0.30 |
|RF                        |- | - | - | -|
|IDW                     |- | - | - | -|
|KNN                     |- | - | - | -|
|XGB                     |- | - | - | -|
|ADAIN                     |- | - | - | -|
</details>

<details>
<summary>Coverage Error (99% or 3 standard deviation) (lower is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Mean |
| :- | -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖ |0.66 | 0.73 | 0.62 | 0.67|
|ARD ✔ N ✖ Cat.✖ Per. ✖ |0.45 | 0.59 | 0.45 | 0.49|
|ARD ✔ N ✖ Cat.✔ Per. ✖ |0.42 | 0.57 | 0.45 | 0.48|
|ARD ✔ N ✖ Cat.✔ Per. ✔ |0.40 | 0.55 | 0.42 | 0.46|
 |ARD ✔ N ✔ Cat.✔ Per. ✖ | 0.22 | 0.28 | 0.40 | 0.30 |
|ARD ✔ N ✔ Cat.✔ Per. ✔ | 0.18 | 0.24 | 0.18 | 0.20 |
|RF                        |- | - | - | -|
|IDW                     |- | - | - | -|
|KNN                     |- | - | - | -|
|XGB                     |- | - | - | -|
|ADAIN                     |- | - | - | -|
</details>

## London Data Experiment
### Preprocessing
In KDD Cup 2018 challenge, London air quality dataset (PM2.5, Latitude and Longitude) is provided along with the grid-wise meteorological data of the London city. To map the meteorological data (Temperature, Pressure, Humidity, Wind direction and Wind speed) with each air quality station, we take a distance-based weighted average (closer the station, higher the weight) of the nearest 4 grid points to the station. 

In our experiments, we choose a month with the least amount of \% missing entries of the AQ variable (PM2.5) (May 2017 consists of 10.8\% missing PM2.5 entries). Note that there are no missing entries in meteorological features. Further, to address the missing entries in the 'May' London air quality dataset, we investigate the stations having a substantial amount of missing values. 2 of the 24 stations ('HR1', 'KF1') from the dataset are having 85\% of the missing PM2.5 values, and thus we remove those stations. After this step, only 1.5% entries are missing in the dataset. To fill in the missing data, we use similar method to the Beijing dataset. Also, unlike Beijing, the London AQ dataset consists only of non-categorical features, thus excluding the (ANCL) configurations with categorical kernel.

### Experimental Setup
We perform 4-fold cross-validation by splitting the train and test sets based on stations. 

### Experimental Configuration

| Configuration | ✖ | ✔ |
| :-: | :-: | :-: |
| ARD | ARD disabled | ARD enabled |
| N | Stationary kernel | Non-stationary kernel |
| Cat. | Using RBF/Matern kernel for categorical features | Using Hamming distance kernel for categorical features |
| Per. | Using RBF/Matern kernel for the time feature | Using Local Periodic kernel for the time feature |

### Results

<details open>
<summary>Root Mean Squared Error (lower is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Fold-3 | Mean |
| :- | -:| -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖	| 5.06 | 4.32 | 5.61	| 4.45 | 4.86 |
|ARD ✔ N ✖ Cat.✖ Per. ✖	| **4.82** | 3.99 | 5.57	| 4.18 | **4.64** |
|ARD ✔ N ✖ Cat.✖ Per. ✔	| 4.83 | **3.99** | 5.60	| **4.16** | 4.65 |
|ARD ✔ N ✔ Cat.✖ Per. ✖| 5.32 | 5.57 | 6.57 | 4.57 | 5.51 |
|ARD ✔ N ✔ Cat.✖ Per. ✔ | 5.14 | 4.44 | 5.00 | 5.03 | 4.90 |
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
 |ARD ✔ N ✔ Cat.✖ Per. ✖| 3.39 | 3.47 | 5.03 | 3.51 | 3.85 |
|ARD ✔ N ✔ Cat.✖ Per. ✔| 3.19 | 3.04 | 3.15 | 3.56 | 3.23 |
|RF	                     | 2.88 | 2.76 | 3.98 | 3.19 | 3.06 |
|IDW	                     | 4.92	| 4.65 |	6.77	| 5.66 | 5.50 |
|KNN	                     | 3.25	| 3.16	| 3.25	| 3.15	| 3.20 |
|XGB	                     | 3.52 | 3.31 | 3.32 | 3.62 | 3.44 |
|ADAIN	                  |3.41|	3.18|	3.38|	3.47|	3.36|

</details>

<details>
<summary>Mean Absolute Percentage Error (lower is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Fold-3 | Mean |
| :- | -:| -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖	 |34% |29% | 87% |38% |47% |
|ARD ✔ N ✖ Cat.✖ Per. ✖	 | 32% |27% |88% |35% |46% |
|ARD ✔ N ✖ Cat.✖ Per. ✔	 | **32%** | **27%** |89% |**35%** |46% |
|ARD ✔ N ✔ Cat.✖ Per. ✖  | 35% | 31% | 107% | 38% | 53% |
|ARD ✔ N ✔ Cat.✖ Per. ✔ | 34% | 30% | 69% | 31% | 41% |
|RF	                     |32% |30% |**71%** |36% |**42%**|
|IDW	                     |51% |46% |131% |57% |71% |
|KNN	                     |32% |31% |78% |35% |44% |
|XGB	                     |40% |35% |78% |40% |48% |
|ADAIN	                   |38%	|31%	|82%	|34%	|46%|

</details>

<details>
<summary>R^2 Score (higher is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Fold-3 | Mean |
| :- | -:| -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖ |0.66 | 0.73 | 0.53 | 0.71 | 0.66|
|ARD ✔ N ✖ Cat.✖ Per. ✖ |**0.69** | **0.77** | 0.54 | **0.74** | **0.69**|
|ARD ✔ N ✖ Cat.✖ Per. ✔ |**0.69** | **0.77** | **0.53** | **0.74** | 0.68|
|ARD ✔ N ✔ Cat.✖ Per. ✖ | 0.62 | 0.56 | 0.36 | 0.69 | 0.56 |
|ARD ✔ N ✔ Cat.✖ Per. ✔ | 0.65 | 0.72 | 0.63 | 0.63 | 0.65 |
|RF                        |0.64 | 0.69 | 0.66 | 0.74 | 0.68|
|IDW                     |0.19 | 0.20 | -0.10 | 0.02 | 0.08|
|KNN                     |0.62 | 0.73 | 0.61 | 0.74 | 0.68|
|XGB                     |0.60 | 0.70 | 0.65 | 0.67 | 0.66|
|ADAIN	                  |0.63|	0.72|	0.64|	0.70|	0.67|

</details>

<details>
<summary>Negative Log Predictive Density (lower is better)</summary>

| Model | Fold-0 | Fold-1 | Fold-2 | Fold-3 | Mean |
| :- | -:| -:| -:| -:| -:|
|ARD ✖ N ✖ Cat.✖ Per. ✖ |12,905.88 | 11,909.78 | 10,528.24 | 10,347.43| 11,422.83 |
|ARD ✔ N ✖ Cat.✖ Per. ✖ |13,046.91 | 11,691.14 | 10,261.17 | 9,994.86| 11,248.52 |
|ARD ✔ N ✖ Cat.✖ Per. ✔ |13,226.03 | 11,659.61 | 10,284.22 | 9,926.65| 11,274.13 |
|ARD ✔ N ✔ Cat.✖ Per. ✖ | 12300.62 | 12201.10 | 10220.89 | 10155.14 | 11219.44 |
|ARD ✔ N ✔ Cat.✖ Per. ✔  | 12186.75 | 11929.85 | 9950.99 | 10285.31 | 11088.23 |
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
|ARD ✖ N ✖ Cat.✖ Per. ✖ |3.25 | 2.92 | 3.39 | 3.04 | 3.15|
|ARD ✔ N ✖ Cat.✖ Per. ✖ |3.21 | 2.83 | 3.42 | 2.94 | 3.10|
|ARD ✔ N ✖ Cat.✖ Per. ✔ |3.28 | 2.84 | 3.42 | 2.93 | 3.12|
 |ARD ✔ N ✔ Cat.✖ Per. ✖ | 3.03 | 3.05 | 3.23 | 2.97 | 3.07 |
|ARD ✔ N ✔ Cat.✖ Per. ✔ | 2.99 | 2.92 | 2.85 | 3.02 | 2.95 |
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
|ARD ✔ N ✖ Cat.✖ Per. ✔ |0.01 | 0.02 | 0.10 | 0.10 | 0.06|
 |ARD ✔ N ✔ Cat.✖ Per. ✖| 0.14 | 0.15 | 0.02 | 0.09 | 0.10 |
|ARD ✔ N ✔ Cat.✖ Per. ✔ | 0.12 | 0.17 | 0.15 | 0.10 | 0.13 |
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
|ARD ✔ N ✔ Cat.✖ Per. ✖ | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
|ARD ✔ N ✔ Cat.✖ Per. ✔| 0.01 | 0.03 | 0.03 | 0.01 | 0.02 |
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
|ARD ✔ N ✔ Cat.✖ Per. ✖ | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 |
|ARD ✔ N ✔ Cat.✖ Per. ✔ | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 |
|RF                        |- | - | - | -|
|IDW                     |- | - | - | -|
|KNN                     |- | - | - | -|
|XGB                     |- | - | - | -|
|ADAIN                     |- | - | - | -|
</details>







