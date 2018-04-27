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

#include "tipl/utility/basic_image.hpp"


#include "tipl/morphology/morphology.hpp"
#include "tipl/segmentation/segmentation.hpp"

#include "tipl/numerical/transformation.hpp"
#include "tipl/numerical/index_algorithm.hpp"
#include "tipl/numerical/interpolation.hpp"
#include "tipl/numerical/window.hpp"
#include "tipl/numerical/basic_op.hpp"
#include "tipl/numerical/numerical.hpp"
#include "tipl/numerical/resampling.hpp"
#include "tipl/numerical/slice.hpp"
#include "tipl/numerical/fft.hpp"
#include "tipl/numerical/optimization.hpp"
#include "tipl/numerical/statistics.hpp"


#include "tipl/io/io.hpp"
#include "tipl/io/dicom.hpp"
#include "tipl/io/nifti.hpp"
#include "tipl/io/bitmap.hpp"
#include "tipl/io/mat.hpp"
#include "tipl/io/2dseq.hpp"
#include "tipl/io/avi.hpp"


#include "tipl/filter/filter_model.hpp"
#include "tipl/filter/anisotropic_diffusion.hpp"
#include "tipl/filter/gaussian.hpp"
#include "tipl/filter/mean.hpp"
#include "tipl/filter/sobel.hpp"
#include "tipl/filter/canny_edge.hpp"
#include "tipl/filter/gradient_magnitude.hpp"


#include "tipl/reg/linear.hpp"
#include "tipl/reg/lddmm.hpp"
#include "tipl/reg/cdm.hpp"
#include "tipl/reg/bfnorm.hpp"

#include "tipl/ml/utility.hpp"
#include "tipl/ml/nb.hpp"
#include "tipl/ml/lg.hpp"
#include "tipl/ml/non_parametric.hpp"
#include "tipl/ml/ada_boost.hpp"
#include "tipl/ml/decision_tree.hpp"
#include "tipl/ml/k_means.hpp"
#include "tipl/ml/em.hpp"
#include "tipl/ml/hmc.hpp"
#include "tipl/ml/svm.hpp"
#include "tipl/ml/cnn.hpp"

#include "tipl/vis/march_cube.hpp"
#include "tipl/vis/color_map.hpp"



#endif//IMAGE_HPP
