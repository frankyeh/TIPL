#ifndef IMAGE_HPP
#define IMAGE_HPP
// Copyright Fang-Cheng Yeh 2010
//
/*
Copyright (c) 2010, Fang-Cheng Yeh
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL FANG-CHENG YEH BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "utility/basic_image.hpp"


#include "morphology/morphology.hpp"
#include "segmentation/segmentation.hpp"

#include "numerical/transformation.hpp"
#include "numerical/index_algorithm.hpp"
#include "numerical/interpolation.hpp"
#include "numerical/window.hpp"
#include "numerical/basic_op.hpp"
#include "numerical/numerical.hpp"
#include "numerical/resampling.hpp"
#include "numerical/slice.hpp"
#include "numerical/fft.hpp"
#include "numerical/optimization.hpp"
#include "numerical/statistics.hpp"

#include "numerical/dif.hpp"


#include "io/io.hpp"
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


#include "reg/linear.hpp"
#include "reg/lddmm.hpp"
#include "reg/cdm.hpp"
#include "reg/bfnorm.hpp"

#include "ml/utility.hpp"
#include "ml/nb.hpp"
#include "ml/lg.hpp"
#include "ml/non_parametric.hpp"
#include "ml/ada_boost.hpp"
#include "ml/decision_tree.hpp"
#include "ml/k_means.hpp"
#include "ml/em.hpp"
#include "ml/hmc.hpp"
#include "ml/svm.hpp"
#include "ml/cnn.hpp"

#include "vis/march_cube.hpp"
#include "vis/color_map.hpp"



#endif//IMAGE_HPP
