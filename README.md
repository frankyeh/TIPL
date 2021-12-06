# Template Image Processing Library (TIPL)

## Introduction

Template Image Processing Library (TIPL) is a header only c++ library for medical imaging processing. 

To use it, include the root header tipl.hpp
```
#include "TIPL/tipl.hpp"  
```

The library supports Linux, MacOS, Windows, and Jupyter Notebook

## Installation with CMake

While the library can be used as is, providing the install location is specified to using projects, we provide 
a CMake system based installation. To install with CMake do the following:

* Download the source to a directory called `./TIPL`
* Configure the installation: 
```bash$
cd TIPL; mkdir build; cd build
cmake .. 
cmake --build . 
cmake --install . --prefix <install_location>
```
or alternatively if using the `make` build system:
```bash$
cd TIPL; mkdir build ; cd build
cmake .. -DCMAKE_INSTALL_PREFIX=<install_location>
make 
make install
```

### Using the installed packge
Afterwards client software building with `CMake` can use this package by specifying
```
find_package(TIPL)
```
in its CMakeLists.txt and linking the client application to the imported target `TIPL::tipl`

A fly in the ointment is that I do not currently know how to make the TIPL libraries C++ standard (14) to 
be propagated. Client applications should also set the `CXX_STANDARD` property of downstream libraries and executables
to 14.

When the client is configured with CMake, it is necessary for CMake to be able to find the installed TIPL
this can be done by adding `<install_location>` for TIPL to the `CMAKE_PREFIX_PATH` or by explicitly specifying `TIPL_DIR`
as
```
cmake -DTIPL_DIR=<install_dir>/lib/cmake/TIPL
```

## Example

- Notebooks examples:
  - Image IO [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/frankyeh/TIPL-example/main?filepath=/image_io.ipynb)
  - Volume and Slicer Operations [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/frankyeh/TIPL-example/main?filepath=/volume_slice_operations.ipynb)
  - Pixel Operations (Filters) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/frankyeh/TIPL-example/main?filepath=/pixel_operations.ipynb)
  - Morphological Operations [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/frankyeh/TIPL-example/main?filepath=/morphology_operations.ipynb)
  - NIFTI file viewer [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/frankyeh/TIPL-example/main?filepath=/nifti_viewer.ipynb)

- Google colab examples:
  - Load NIFTI file [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/frankyeh/TIPL-example/blob/main/colab/load_nii.ipynb)
  - Image registration [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/frankyeh/TIPL-example/blob/main/colab/spatial_normalization.ipynb)


## Design paradigm

A lot of the image processing libraries are designed for experimental/research purpose and do not meet the industrial standard. Consequently, the performance of the codes is not optimal, and the library can be hard to read and use. The design of TIPL follows the several coding guidelines and principles [1-5] that makes it highly efficient and reusable. The following is the main paradigm behind TIPL.

- Decouple image type and image processing method. Most of the image processing libraries are limited to their defined image type. TIPL is not. You may use pointer, or any kind of memory block to as the input. This reduce the unnecessary memory storage and copy.

- Not limited to RGB pixel type. In medical imaging, the most common pixel type is "short" or "float", not the RGB value. TIPL makes no assumption on the pixel type to achieve the best applicability..

- No class inheritance, no hidden functions or interfaces. Class inheritance is known to cause programs in code maintenance  and it is not friendly for library users to customize the library. TIPL combines template-based interfaces with C-style interface to provide a "flat" library structures that is easy to maintain and modify. The connections between header files are thereby minimized. 

## Features

- Headers only, and easy to use. 
- BSD license, free for all purposes        

