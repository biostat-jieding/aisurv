# *aisurv Python package*

- **It is still in preparation!!!**

## Auxiliary Information Synthesis in Survival Analysis

This is an R package for analyzing ...
- ...
- *This Python package was contributed by **Jie Ding**, **XXX** and **Xiaoguang Wang**.*

## Package description and included main functions

Installation of this package can be done locally after downloading the package manually from this github website. We will also upload this package to the Comprehensive R Archive Network (CRAN) so that it can be downloaded as a standard R package. Currently, it can be loaded using R command
```R
devtools::install_github("biostat-jieding/aisurv")
library(aisurv)
```

The main function included in our R package is *...()*. It can be called via the following R synopsis:
```R
...(
    yobs,delta,X,
    ...
)
```
We refer to its help page for more detailed explanations of the corresponding arguments (typing *?...*). 

## Illustration via a simulated dataset

An example of the use of this package can be found in the following of this file.

### Data preparation

Set the true underlying quantities:
```R
...
```

Generate the survival dataset:
```R
set.seed(1)
sdata <- ...
```

### Model fitting 

Fit the model via provided function:
```R
...
```

Extract fitted estimates:
```R
...
```

### Visualization

Preparation：
```R
par(mfrow=c(1,3))
```

