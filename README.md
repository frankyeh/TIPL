# Template Image Processing Library (TIPL)

## Introduction

Template Image Processing Library (TIPL) is a lightweight C++ template library designed mainly for medical imaging processing. The design paradigm is to provide an "easy-to-use" and also "ready-to-use" library. You need only to include the header files to use it. 

First, get header files from Github

```
!git clone http://github.com/frankyeh/TIPL/
```

Then 

```
#include "TIPL/tipl.hpp"  
```

Now you can use TIPL

## Example

- Notebooks examples:
  - Image IO [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/frankyeh/TIPL-example/main?filepath=/image_io.ipynb)
  - Volume and Slicer Operations [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/frankyeh/TIPL-example/main?filepath=/volume_slice_operations.ipynb)
  - Pixel Operations (Filters) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/frankyeh/TIPL-example/main?filepath=/pixel_operations.ipynb)
  - Morphological Operations [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/frankyeh/TIPL-example/main?filepath=/morphology_operations.ipynb)
  - Matrix and Vector Operations [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/frankyeh/TIPL-example/main?filepath=/matrix_vector.ipynb) 

- Google colab examples:
  - Load NIFTI file [![Colab](https://colab.research.google.com/assets/colab-badge.svg)]("https://colab.research.google.com/github/frankyeh/TIPL-example/blob/main/colab/load_nii.ipynb)
  - Image registration [![Colab](https://colab.research.google.com/assets/colab-badge.svg)]("https://colab.research.google.com/github/frankyeh/TIPL-example/blob/main/colab/spatial_normalization.ipynb)

## Support Jupyter notebook

TIPL can be used in Jupyter notebook with xeus-cling kernel to provide interactive processing. No additional installation is required.

## Design paradigm

A lot of the image processing libraries are designed for experimental/research purpose and do not meet the industrial standard. Consequently, the performance of the codes is not optimal, and the library can be hard to read and use. The design of TIPL follows the several coding guidelines and principles [1-5] that makes it highly efficient and reusable. The following is the main paradigm behind TIPL.

- Decouple image type and image processing method. Most of the image processing libraries are limited to their defined image type. TIPL is not. You may use pointer, or any kind of memory block to as the input. This reduce the unnecessary memory storage and copy.

- Not limited to RGB pixel type. In medical imaging, the most common pixel type is "short" or "float", not the RGB value. TIPL makes no assumption on the pixel type to achieve the best applicability..

- No class inheritance, no hidden functions or interfaces. Class inheritance is known to cause programs in code maintenance  and it is not friendly for library users to customize the library [4]. TIPL combines template-based interfaces with C-style interface to provide a "flat" library structures that is easy to maintain and modify. The connections between header files are thereby minimized. 

## Features

- Headers only, and easy to use. 
- BSD license, free for all purposes        

