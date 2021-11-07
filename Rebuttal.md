# Extended Feedback

Click on any of the links below to naviagate to a specific question.

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


### London Data Experiment
