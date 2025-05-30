/*
Copyright (c) 2010-2023 Fang-Cheng Yeh
All rights reserved.

TIPL library is shared under one of the following licenses
(1) GPLv4 license (https://www.gnu.org/licenses/gpl-3.0.en.html)
(2) A propriety license allowing close-source usage.

*/

#include "def.hpp"

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
#include "reg/bfnorm.hpp"

#include "cmd.hpp"
#include "prog.hpp"
#include "po.hpp"


#include "ml/utility.hpp"
#include "ml/nb.hpp"
#include "ml/lg.hpp"
#include "ml/non_parametric.hpp"
#include "ml/ada_boost.hpp"
#include "ml/decision_tree.hpp"
#include "ml/k_means.hpp"
#include "ml/em.hpp"
#include "ml/hmc.hpp"
#include "ml/cnn.hpp"

#ifndef __CUDACC__
#include "ml/cnn3d.hpp"
#include "ml/unet3d.hpp"
#endif

#include "vis/march_cube.hpp"
#include "vis/color_map.hpp"
#include "vis/qt_ext.hpp"



#ifdef __CUDACC__
#include "cu.hpp"
#endif

#ifdef USING_XEUS_CLING
// XEUS interface
#include "xtl/xbase64.hpp"
#include "nlohmann/json.hpp"
namespace nl = nlohmann;

namespace tipl
{
    template <typename pixel_type,typename storage_type>
    nl::json mime_bundle_repr(const tipl::image<2,pixel_type,storage_type>& I)
    {
        tipl::io::bitmap bmp;
        bmp << I;
        std::stringstream out;
        bmp.save_to_stream(out);
        auto bundle = nl::json::object();
        bundle["image/png"] = xtl::base64encode(out.str());
        return bundle;
    }
    template <typename pixel_type,typename storage_type>
    nl::json mime_bundle_repr(const tipl::image<3,pixel_type,storage_type>& I)
    {
        return mime_bundle_repr(I.slice_at(I.depth()/2));
    }
    template <int dim>
    nl::json mime_bundle_repr(const tipl::shape<dim>& d)
    {
        std::stringstream out;
        out << d;
        auto bundle = nl::json::object();
        bundle["text/plain"] = out.str();
        return bundle;
    }
    template <int r,int c,typename value_type>
    nl::json mime_bundle_repr(const tipl::matrix<r,c,value_type>& m)
    {
        std::stringstream out;
        out << m;
        auto bundle = nl::json::object();
        bundle["text/plain"] = out.str();
        return bundle;
    }
    template <int dim,typename value_type>
    nl::json mime_bundle_repr(const tipl::vector<dim,value_type>& v)
    {
        std::stringstream out;
        out << v;
        auto bundle = nl::json::object();
        bundle["text/plain"] = out.str();
        return bundle;
    }
    template <typename value_type>
    nl::json mime_bundle_repr(const std::vector<value_type>& vec)
    {
        std::stringstream out;
        for(const auto& i : vec)
            out << i << std::endl;
        auto bundle = nl::json::object();
        bundle["text/plain"] = out.str();
        return bundle;
    }
    namespace io{
        nl::json mime_bundle_repr(const tipl::io::nifti& nii)
        {
            std::stringstream out;
            out << nii;
            auto bundle = nl::json::object();
            bundle["text/plain"] = out.str();
            return bundle;
        }
    }
}

#endif//INCLUDE_NLOHMANN_JSON_HPP_

