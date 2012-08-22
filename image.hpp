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

#include "image/utility/basic_image.hpp"

#include "image/morphology/morphology.hpp"
#include "image/segmentation/segmentation.hpp"
#include "image/algo/march_cube.hpp"

#include "image/numerical/transformation.hpp"
#include "image/numerical/index_algorithm.hpp"
#include "image/numerical/interpolation.hpp"
#include "image/numerical/window.hpp"
#include "image/numerical/basic_op.hpp"
#include "image/numerical/numerical.hpp"
#include "image/numerical/resampling.hpp"
#include "image/numerical/slice.hpp"
#include "image/numerical/fft.hpp"
#include "image/numerical/optimization.hpp"
#include "image/numerical/statistics.hpp"


#include "image/io/io.hpp"
#include "image/io/dicom.hpp"
#include "image/io/nifti.hpp"
#include "image/io/bitmap.hpp"
#include "image/io/mat.hpp"
#include "image/io/2dseq.hpp"

#include "image/filter/filter_model.hpp"
#include "image/filter/anisotropic_diffusion.hpp"
#include "image/filter/gaussian.hpp"
#include "image/filter/mean.hpp"
#include "image/filter/sobel.hpp"
#include "image/filter/canny_edge.hpp"
#include "image/filter/gradient_magnitude.hpp"


#include "image/reg/linear.hpp"
#include "image/reg/lddmm.hpp"
#include "image/reg/dmdm.hpp"


#endif//IMAGE_HPP
