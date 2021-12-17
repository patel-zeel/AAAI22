# AAAI22

* [Author Response during the rebuttal](AuthorResponse.md)
* [Appendix](appendix.pdf)

## Main approach configuration

### Encoding
```A``` - ARD enabled

```A_bar``` - ARD disabled

```N``` - Non-stationary kernel  

```N_bar``` - Stationary kernel   

```C``` - Using categorical kernel for categorical features without one-hot-encoding

```C_bar``` - Using RBF/Matern kernel for categorical features with one-hot-encoding

```L``` - Using Local periodic kernel for time feature

```L_bar``` - Using RBF/Matern kernel for time feature

### Example
```AN_barCL_bar``` - GP with ARD enabled stationary kernel with categorical kernel for categorical features and RBF/Matern kernel for time feature

## Folder-wise description

|Folder | Description|
|:------|:-----------|
| data  | data for each baseline and main approach |
| preprocessing | preprocessing pipeline applied to data |
| stat_gp_cat   | Stationary GP with categorical kernel (```C``` fixed, ```L``` variable)|
| stat_gp_no_cat | Stationary GP without categorical kernel (```C_bar``` fixed, ```L``` variable) |
| nonstat_gp_cat | Non-stationary GP with categorical kernel (```C``` fixed, ```L``` variable) |
