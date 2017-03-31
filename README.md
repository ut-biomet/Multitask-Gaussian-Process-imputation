# Multitask-Gaussian-Process-imputation
Codes for multitask Gaussian process imputation and test data are avilable.

"MTGP_imputation" directory are written as the library for python2.7.
"MTGP.py" module contains function to impute missing values with parameters selected by cross validation. Theoretical explanation is written on Hori et.al. (2016).
"MTGP_TPE.py" module selects parameters with Tree Parzen window estimater. 
Required other libraries for the library is "numpy", "scipy", "pandas", "scikit-tensor", "hyperopt", "pymongo", "networkx".  
We checked the library is available on python2.7.10.

Two sample codes are available. "test_MTGP.py" . "matplotlib" is also required to visualize the code.

Test data are available on "tests" directory. 

2017/03/26
Tomoaki Hori
hori@ut-biomet.org

Refenrece
Hori, T., Montcho, D., Agbangla, C., Ebana, K., Futakuchi, K., & Iwata, H. (2016). Multi-task Gaussian process for imputing missing data in multi-trait and multi-environment trials. Theoretical and Applied Genetics, 129(11), 2101-2115.