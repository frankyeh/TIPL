/*
Copyright (c) 2010-2026 Fang-Cheng Yeh
All rights reserved.

The TIPL library is dual-licensed. You may use it under one of the following licenses:

1. GNU General Public License v3.0 (GPLv3)
   https://www.gnu.org/licenses/gpl-3.0.en.html
2. A proprietary license granted by the copyright holder, which permits closed-source use.
*/

#include "def.hpp"
#include "cmd.hpp"
#include "mt.hpp"
#include "prog.hpp"
#include "po.hpp"


#include "utility/basic_image.hpp"
#include "utility/pixel_index.hpp"
#include "utility/rgb_image.hpp"


#include "numerical/transformation.hpp"
#include "numerical/index_algorithm.hpp"
#include "numerical/interpolation.hpp"
#include "numerical/window.hpp"
#include "numerical/basic_op.hpp"
#include "numerical/numerical.hpp"
#include "numerical/resampling.hpp"
#include "numerical/fft.hpp"
#include "numerical/optimization.hpp"
#include "numerical/statistics.hpp"
#include "numerical/dif.hpp"
#include "numerical/morphology.hpp"
#include "numerical/otsu.hpp"

#include "io/gz_stream.hpp"
#include "io/nrrd.hpp"
#include "io/dicom.hpp"
#include "io/nifti.hpp"
#include "io/bitmap.hpp"
#include "io/mat.hpp"
#include "io/2dseq.hpp"
#include "io/avi.hpp"

#include "filter/filter_model.hpp"
#include "filter/anisotropic_diffusion.hpp"
#include "filter/gaussian.hpp"
#include "filter/mean.hpp"
#include "filter/sobel.hpp"
#include "filter/canny_edge.hpp"
#include "filter/gradient_magnitude.hpp"
#include "filter/laplacian.hpp"

#include "reg/linear.hpp"
#include "reg/cdm.hpp"
#include "reg/mm_reg.hpp"


#include "ml/utility.hpp"
#include "ml/k_means.hpp"
#include "ml/em.hpp"

#ifndef __CUDACC__
#include "ml/cnn3d_detail.hpp"
#include "ml/cnn3d.hpp"
#include "ml/unet3d.hpp"
#endif

#include "vis/march_cube.hpp"
#include "vis/color_map.hpp"
#include "vis/qt_ext.hpp"



#ifdef __CUDACC__
#include "cu.hpp"
#endif


