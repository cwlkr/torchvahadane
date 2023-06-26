# TorchVahadane

[Vahdane stain normalization](https://ieeexplore.ieee.org/document/7460968) is being used extensively in Digital Pathology workflows to provide better generalization of Deep Learning models in between cohorts.

The StainTools package has been one of the most used and most clear implementation of the Vahadane stain normalization.
Unfortunately, StainTools can be slow when used on large images or on a large number of images.

This repository implements a GPU accelerated version of the Vahadane stain normalization using torch.

This repository provides a fully GPU based stain normalization workflow, useful in combination with cuCIM and a faster workflow using CPU based stain matrix estimation with accelerated stain concentration estimation.

For WSI workflows, a fixed target stain matrix can be set, eliminating the need for recalculating the stain matrix for every new image patch and making the transformation fully GPU based.

![Screenshot]('example_images/fig.png')

Method| fit [s] | transform  [s] | total  [s]
| :--- | :---: | :---: | :---:
**StainTools Vahadane**| 25.1 | 24.3 | 49.4
**TorchVahadane** | 7.3 | 6.5 | 13.8
**TorchVahadane ST**| 3.3 | 1.8 |  ***5.1***


## Installation
torchvahadane can be installed with pip using

```
git clone https://github.com/cwlkr/torchvahadane.git
cd torchvahadane
pip install .
```

or directly

```
pip install git+https://github.com/cwlkr/torchvahadane.git
```

## Usage

TorchVahadane can be employed as a drop-in replacement for StainTools.
Per default, the TorchVahadaneNormalizer uses the cuda device and uses staintools based stain_matrix estimation.
As StainTools is now a read-only repository, StainTools is integrated and not used as a dependency.

```
from torchvahadane import TorchVahadaneNormalizer
normalizer = TorchVahadaneNormalizer(device='cuda', staintools_estimate=True)
normalizer.fit(target)
normalizer.transform(img)

```

## Notes
Spams installation through pip throws more errors then not. Using conda pre-compiled binaries might work best.
Spams is not listed in package requirements

## Acknowledgments

Several lines of code in this repository are directly adapted from the following repositories I would like to credit with their excelent work!

[StainTools](https://github.com/Peter554/StainTools)    
[pytorch-lasso](https://github.com/rfeinman/pytorch-lasso)  
[torchstain](https://github.com/EIDOSLAB/torchstain)
