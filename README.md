# Template Image Processing Library (TIPL)

**Template Image Processing Library (TIPL)** is a lightweight C++ template library designed mainly for high-performance medical image processing.

TIPL provides image containers, numerical routines, image I/O, filters, registration tools, machine learning utilities, visualization helpers, and optional CUDA support. It is designed to be easy to embed in C++ projects that need direct image access, low dependency overhead, and efficient voxel-wise computation.

## Features

- Header-based C++ image processing library
- Cross-platform support: Linux, macOS, and Windows
- C++17-compatible design
- Medical image I/O support, including NIFTI, DICOM, NRRD, bitmap, MAT, Bruker 2dseq, and AVI
- Numerical operations, interpolation, resampling, FFT, optimization, and statistics
- Image filters, including Gaussian, mean, Sobel, Canny edge, Laplacian, gradient magnitude, and anisotropic diffusion
- Morphology and Otsu thresholding
- Linear and nonlinear registration components
- Machine learning utilities, including classifiers, clustering, CNN, and 3D U-Net components
- Visualization utilities, including marching cubes and color maps
- Optional CUDA support when compiled with CUDA
- Jupyter Notebook support through xeus-cling examples

## Quick start

Clone the repository:

```bash
git clone https://github.com/frankyeh/TIPL.git
````

Include the root header:

```cpp
#include "TIPL/tipl.hpp"
```

A minimal example:

```cpp
#include "TIPL/tipl.hpp"

int main()
{
    tipl::image<3,float> image({64,64,64});
    image = 0.0f;
    image.at(32,32,32) = 1.0f;
    return 0;
}
```

## Installation

TIPL can be used directly as a header-based library, or installed through CMake.

### Option 1: Use directly

Clone the repository and add the TIPL directory to your compiler include path:

```bash
git clone https://github.com/frankyeh/TIPL.git
```

Then include:

```cpp
#include "TIPL/tipl.hpp"
```

### Option 2: Install with CMake

```bash
git clone https://github.com/frankyeh/TIPL.git
cd TIPL
mkdir build
cd build
cmake ..
cmake --build .
cmake --install . --prefix <install_location>
```

For Makefile-based builds:

```bash
git clone https://github.com/frankyeh/TIPL.git
cd TIPL
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=<install_location>
make
make install
```

## Using TIPL in a CMake project

After installation, client projects can use:

```cmake
find_package(TIPL REQUIRED)
target_link_libraries(your_target PRIVATE TIPL::tipl)
```

If CMake cannot locate TIPL, add the installation path to `CMAKE_PREFIX_PATH`:

```bash
cmake .. -DCMAKE_PREFIX_PATH=<install_location>
```

or specify `TIPL_DIR` directly:

```bash
cmake .. -DTIPL_DIR=<install_location>/lib/cmake/TIPL
```

TIPL is intended for C++17-compatible compilers. If needed, set the C++ standard in your downstream project:

```cmake
set_property(TARGET your_target PROPERTY CXX_STANDARD 17)
```

## Examples

Example notebooks and C++ examples are available in the companion repository:

[https://github.com/frankyeh/TIPL-example](https://github.com/frankyeh/TIPL-example)

### Jupyter Notebook examples

* Image I/O
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/frankyeh/TIPL-example/main?filepath=/image_io.ipynb)

* Volume and slice operations
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/frankyeh/TIPL-example/main?filepath=/volume_slice_operations.ipynb)

* Pixel operations and filters
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/frankyeh/TIPL-example/main?filepath=/pixel_operations.ipynb)

* Morphological operations
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/frankyeh/TIPL-example/main?filepath=/morphology_operations.ipynb)

* NIFTI file viewer
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/frankyeh/TIPL-example/main?filepath=/nifti_viewer.ipynb)

### Google Colab examples

* Load a NIFTI file
  [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/frankyeh/TIPL-example/blob/main/colab/load_nii.ipynb)

* Image registration
  [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/frankyeh/TIPL-example/blob/main/colab/spatial_normalization.ipynb)

## Design principles

TIPL was designed for medical image analysis, where images are often stored as scalar volumes rather than RGB images, and where direct memory access and efficient voxel-wise operations are important.

The main design principles are:

### Decouple image storage from image processing

TIPL algorithms are designed to work with different image and memory types. Images can be backed by standard containers, pointers, or device memory. This reduces unnecessary memory copies and allows algorithms to operate on existing data buffers.

### Avoid assumptions about pixel type

Medical images are commonly stored as `short`, `unsigned short`, `float`, or other scalar types. TIPL does not assume RGB pixels and is designed to work with different voxel types.

### Keep the library lightweight

TIPL avoids unnecessary class hierarchy and favors template-based coupling between image types and algorithms. This keeps the code easier to embed, modify, and optimize.

### Reduce unnecessary dependencies

TIPL keeps dependencies minimal and is designed to compile across common C++ compilers, including MSVC, Clang, and GCC.

### Support practical medical imaging workflows

TIPL includes components for image I/O, filtering, morphology, interpolation, resampling, registration, machine learning, and visualization. These components are intended to support practical medical-image processing pipelines.

## Repository structure

```text
TIPL/
├── filter/       Image filters
├── io/           Image I/O formats
├── ml/           Machine learning and neural network utilities
├── numerical/    Numerical operations, interpolation, FFT, optimization
├── reg/          Image registration
├── utility/      Image containers and core utility classes
├── vis/          Visualization utilities
├── tipl.hpp      Root include header
├── def.hpp       Core definitions
├── mt.hpp        Multithreading utilities
├── prog.hpp      Progress reporting utilities
├── po.hpp        Program option utilities
└── CMakeLists.txt
```

## License

Copyright (c) 2010-2026 Fang-Cheng Yeh
All rights reserved.

The TIPL library is dual-licensed. You may use it under either of the following licenses:

1. GNU General Public License v3.0 (GPLv3)
   [https://www.gnu.org/licenses/gpl-3.0.en.html](https://www.gnu.org/licenses/gpl-3.0.en.html)

2. A proprietary license granted by the copyright holder, which permits closed-source use.

For proprietary or closed-source usage, please contact the copyright holder.

