#include "image/numerical/numerical.hpp"
#include "image/numerical/basic_op.hpp"
#include "image/morphology/morphology.hpp"
#include <cstdlib>
#include <ctime>

#ifdef TIPL_DEBUG
#include <image/io/nifti.hpp>
#include "image/io/bitmap.hpp"
#include <sstream>
#endif



namespace image
{

namespace segmentation
{

namespace imp
{

template<class pixel_type>
class intensity_likelihood
{
private:
    bool changed;
    double sum;
    double sum2;
    double log_mode;
    double var2;
public:
    unsigned int num;
    double mean;
    double var;

public:
    intensity_likelihood(void):num(0),sum(0.0),sum2(0.0),changed(false) {}
    void update_parameters(void)
    {
        changed = false;
            mean = sum/num;
            var = sum2/num - mean*mean;
            if(var < 0.000000001)
                var = 0.000000001;
        var2 = var * -2.0;
        log_mode = std::log(var) * -0.5;
    }
    void push_back(pixel_type value)
    {
        double temp = value;
        sum += temp;
        sum2 += temp*temp;
        ++num;
        changed = true;
    }
    void pop_back(pixel_type value)
    {
        double temp = value;
        sum -= temp;
        sum2 -= temp*temp;
        --num;
        changed = true;
    }

    double log_likelihood(pixel_type value)
    {
        if(num == 0)
            return -100000000.0;
        if(changed)
            update_parameters();
        double dist = value - mean;
        return log_mode+(dist*dist)/var2;
    }
};



struct intensity_disabled{
    bool operator[](int)const {return false;}
};

struct intensity_enabled{
    bool operator[](int)const {return false;}
};

// initialize model parameters according to the initial contour
template<class ImageType,class LabelImageType,class pixel_type>
void stochastic_competition_init_model_param(
    const ImageType& src,const LabelImageType& label,
    std::vector<imp::intensity_likelihood<pixel_type> >& intensity_model)
{
    for(unsigned int index = 0; index < label.size(); ++index)
    {
        unsigned int labeling = label[index];
        if(labeling >= intensity_model.size())
            intensity_model.resize(labeling + 1);
        intensity_model[labeling].push_back(src[index]);
    }
}

// normalizing gradient field
template<class GradientImageType>
void stochastic_competition_init_gradient(GradientImageType& gre)
{
    double max_gre = 0;
    for(unsigned int index = 0; index < gre.size(); ++index)
        if(max_gre < gre[index].length())
            max_gre = gre[index].length();
    for(unsigned int index = 0; index < gre.size(); ++index)
        gre[index] /= max_gre;
}

// initialize pivot pool
template<class LabelImageType,class PivotMapType>
void stochastic_competition_init_pivots(
    const LabelImageType& label,
    PivotMapType& pivot_map,
    std::vector<unsigned int>& pivot_list)
{
    image::morphology::edge(label,pivot_map);
    pivot_list.clear();
    for(unsigned int index = 0; index < pivot_map.size(); ++index)
        if(pivot_map[index])
        {
            pivot_list.push_back(index);
            pivot_map[index] = pivot_list.size();
        }
}

// update pivot pool
template<class LabelImageType,class PivotMapType,class index_type>
void stochastic_competition_update_pivots(
    const LabelImageType& label,
    const std::vector<index_type>& neighbor_list,
    PivotMapType& pivot_map,
    std::vector<unsigned int>& pivot_list)
{
    for(unsigned int index = 0; index < neighbor_list.size(); ++index)
    {
        index_type cur_index = neighbor_list[index];
        if(image::morphology::is_edge(label,cur_index))
        {
            if(!pivot_map[cur_index.index()])
            {
                pivot_list.push_back(cur_index.index());
                pivot_map[cur_index.index()] = pivot_list.size();
            }
        }
        else
        {
            if(pivot_map[cur_index.index()])
            {
                unsigned int replace_index = pivot_map[cur_index.index()];
                pivot_list[replace_index-1] = pivot_list.back();
                pivot_map[pivot_list.back()] = replace_index;
                pivot_list.resize(pivot_list.size()-1);
                pivot_map[cur_index.index()] = 0;
            }
        }
    }
}


// randomly select a pivot from pool
inline unsigned int stochastic_competition_select_pivot(const std::vector<unsigned int>& pivot_list)
{
    unsigned int rand_32bit = ((unsigned int)std::rand() & 0x0007FFF) |
                        (((unsigned int)std::rand() & 0x0007FFF) << 15);
    double rx = rand_32bit;
    rx /= (0x0007FFF | (0x0007FFF << 15));
    rx *= pivot_list.size();
    rx = std::floor(rx);
    if(rx >= pivot_list.size())
        rx = (double)pivot_list.size()-1;
    return pivot_list[(unsigned int)rx];
}

}

/** initialize the labeling map
    unkown region = 0
    object region = 1
    background region = 2
*/
template<class LabelImageType>
void stochastic_competition_3region(LabelImageType& label,double inner_region_ratio = 0.5,double outer_region_ratio = 0.9)
{
    typedef image::pixel_index<LabelImageType::dimension> index_type;
    std::vector<double> fdim(LabelImageType::dimension);
    for(unsigned int index = 0; index < fdim.size(); ++index)
        fdim[index] = ((double)label.geometry()[index])/2.0;
    std::fill(label.begin(),label.end(),0);
    for(index_type iter; iter.valid(label.geometry()); iter.next(label.geometry()))
    {
        double ratio = 0;
        for(unsigned int index = 0; index < fdim.size(); ++index)
        {
            double dim_r = 1.0-((double)iter[index])/fdim[index];
            dim_r *= dim_r; // dim_r 0~1.0
            ratio += dim_r;
        }
        ratio = std::sqrt(ratio);
        if(ratio < inner_region_ratio)
        {
            label[iter.index()] = 1;
            continue;
        }
        if(ratio > outer_region_ratio)
        {
            label[iter.index()] = 2;
            continue;
        }
    }
}


#ifdef TIPL_DEBUG
#include <sstream>
template<class ImageType,class PitvotList,class LabelImageType>
void stochastic_competition_debug(const ImageType& data,
                                  const PitvotList& pivot_list,
                                  const LabelImageType& label)
{
    static unsigned int total_loop = 0;
    {
        // for debug
        image::basic_image<unsigned char,ImageType::dimension> image_data;
        image::normalize(data,image_data);
        for(unsigned int index = 0; index < pivot_list.size(); ++index)
            if(label[pivot_list[index]])
                image_data[pivot_list[index]] = 0x00FFFFFF;

        image::io::bitmap bitmap_file;
        bitmap_file << image_data;
        std::string file_name = "c:/STRCMP";
        std::ostringstream out;
        out << total_loop;
        file_name += out.str();
        file_name += ".bmp";
        bitmap_file.save_to_file(file_name.c_str());
    }
    ++total_loop;
}
#endif

//  Image label: 0 reserved to unknown label
//
template<class ImageType,class LabelImageType,class LostInfoType>
void stochastic_competition_with_lostinfo(const ImageType& src,
                            LabelImageType& label,
                            const LostInfoType& no_info_map,
                            double Zc = 5.0,
                            double Zr = 5.0)
{
    const double initT = 1.0;
    const double T_cooling_step = 0.02;
    const double termination_ratio = 0.02;
    typedef image::pixel_index<ImageType::dimension> index_type;
    typedef typename LabelImageType::value_type label_type;
    typedef typename ImageType::value_type pixel_type;
    typedef typename image::vector<ImageType::dimension,float> vector_type;


    // initial estimation of model parameters
    std::vector<imp::intensity_likelihood<pixel_type> > intensity_model;
    imp::stochastic_competition_init_model_param(src,label,intensity_model);
    // initialize pivot pool
    image::basic_image<unsigned int,ImageType::dimension> pivot_map;
    std::vector<unsigned int> pivot_list;
    imp::stochastic_competition_init_pivots(label,pivot_map,pivot_list);
    // initialize gradient vector map
    image::basic_image<vector_type,ImageType::dimension> gre;
    image::gradient_multiple_sampling(src,gre);
    imp::stochastic_competition_init_gradient(gre);

#ifdef TIPL_DEBUG
    stochastic_competition_debug(src,pivot_list,label);
#endif
    //stochastic EM
    double T = initT;
    const unsigned int clique_radius = 2;
    const unsigned int clique_size = 25;
    Zr *= 2.0;
    Zc /= (double)clique_radius * std::sqrt(2.0f);
    Zr /= clique_size;
    Zc /= clique_size;
    long t = std::clock()+CLOCKS_PER_SEC*5;
    for(unsigned int iteration = 0,success_pivot = 0;
        !pivot_list.empty() && std::clock() < t; ++iteration)
    {
        // E-step
        // 1: randomly select a pivot
        unsigned int pivot_index = imp::stochastic_competition_select_pivot(pivot_list);
        index_type pivot_full_index = index_type(pivot_index,label.geometry());
        pixel_type pivot_intensity = src[pivot_index];
        label_type cur_label = label[pivot_index];

        std::vector<index_type> neighbor_list;
        std::vector<index_type> supporting_neighbors;
        label_type expected_label;
        bool pivot_without_info = no_info_map[pivot_index];
        bool neighbor_without_info = pivot_without_info;
        int clique_potential = 0;

        // 2: select an expected labeling
        {
            image::get_neighbors(pivot_full_index,label.geometry(),2,neighbor_list);

            std::vector<label_type> other_label;
            for(unsigned int j = 0; j < neighbor_list.size(); ++j)
            {
                unsigned int neighbor_index = neighbor_list[j].index();
                if(no_info_map[neighbor_index])
                    neighbor_without_info = true;
                if(label[neighbor_index] != cur_label)
                    other_label.push_back(label[neighbor_index]);
                else
                {
                    supporting_neighbors.push_back(neighbor_list[j]);
                    --clique_potential;
                }
            }
            // don't include pivot it self
            ++clique_potential;
            if(other_label.empty())
                continue;
            // choose a random label
            std::random_shuffle(other_label.begin(),other_label.end());
            expected_label = other_label.front();
            if(expected_label == 0)//unkwnown_label
                continue;
            clique_potential += std::count(other_label.begin(),other_label.end(),expected_label);
        }

        // 3 compute the log prob difference

        double log_dif = 0;
        // intensity likelihood
        if(!pivot_without_info)
            log_dif += intensity_model[expected_label].log_likelihood(pivot_intensity)-
                       intensity_model[cur_label].log_likelihood(pivot_intensity);
        // gradient likelihood
        if(!neighbor_without_info)
        {
            unsigned int low_intensity_label;
            if(intensity_model[expected_label].mean > intensity_model[cur_label].mean)
                low_intensity_label = cur_label;
            else
                low_intensity_label = expected_label;
            double log_gre_dif = 0.0;
            for(unsigned int j = 0; j < neighbor_list.size(); ++j)
            {
                unsigned int neighbor_index = neighbor_list[j].index();
                // ignore high intensity pixels
                if(label[neighbor_index] != low_intensity_label || neighbor_list[j] == pivot_full_index)
                    continue;
                vector_type pivot_gre;
                if(low_intensity_label == cur_label)
                    pivot_gre = gre[pivot_full_index.index()]-gre[neighbor_list[j].index()];
                else
                    pivot_gre = gre[neighbor_list[j].index()]-gre[pivot_full_index.index()];
                vector_type cur_dist(neighbor_list[j]);
                cur_dist -= vector_type(pivot_full_index);
                log_gre_dif += cur_dist * pivot_gre;
            }
            log_dif += Zc*log_gre_dif;
        }

        // spatial prior
        log_dif += Zr*(double)clique_potential;

        // simulated annealing
        if(log_dif < 0 && std::exp(log_dif/T)*RAND_MAX < std::rand())
            continue;

        // change the label of the pivot
        label[pivot_index] = expected_label;
        ++success_pivot;
        // update pivot pool
        imp::stochastic_competition_update_pivots(label,neighbor_list,pivot_map,pivot_list);

        // M-step
        // update model parameters
        if(!pivot_without_info)
        {
            intensity_model[cur_label].pop_back(pivot_intensity);
            intensity_model[expected_label].push_back(pivot_intensity);
        }

        if(iteration >= (pivot_list.size() << 2))
        {
            // termination criteria
            if((double)(pivot_list.size() << 2)*termination_ratio > success_pivot ||
                T <= T_cooling_step)
            {
                if(intensity_model[0].num == 0)
                    break;
                for(unsigned int index = 0; index < label.size(); ++index)
                    if(label[index] == 0)
                    {
                        intensity_model[2].push_back(src[index]);
                        label[index] = 2;
                    }
                intensity_model[0].num = 0;
                imp::stochastic_competition_init_pivots(label,pivot_map,pivot_list);
                T = initT/4;
            }
#ifdef TIPL_DEBUG
            stochastic_competition_debug(src,pivot_list,label);
#endif
            if(T > T_cooling_step)
                T -= T_cooling_step;
            iteration = success_pivot = 0;
        }
    }
#ifdef TIPL_DEBUG
    for(unsigned int index = 0; index < pivot_list.size(); ++index)
        if(label[pivot_list[index]] == 2)
            label[pivot_list[index]] = 0;
    stochastic_competition_debug(src,pivot_list,label);
#endif

}
/*
initial_contour has 1 insid the contour and 0 elsewhere.
*/
template<class ImageType,class LabelImageType>
void stochastic_competition(const ImageType& src,
                            LabelImageType& initial_contour,
                            double Zc = 30.0,
                            double Zr = 5.0,
                            bool consider_region_intensity = true)
{
    image::basic_image<unsigned char,LabelImageType::dimension> outter_contour(initial_contour);
    image::geometry<ImageType::dimension> range_max,range_min,new_geo;

    image::bounding_box(initial_contour,range_min,range_max,0);

    image::morphology::dilation2(outter_contour,(range_max[0]-range_min[0])/5);
    image::morphology::erosion2(initial_contour,(range_max[0]-range_min[0])/5);

    image::bounding_box(outter_contour,range_min,range_max,0);

    for(int dim = 0;dim < ImageType::dimension;++dim)
    {
        int min_value  = std::max((int)0,(int)range_min[dim]*5/4-(int)range_max[dim]/4);
        range_max[dim] = std::min((int)src.geometry()[dim],(int)range_max[dim]*5/4-(int)range_min[dim]/4);
        range_min[dim] = min_value;
        new_geo[dim] = range_max[dim]-range_min[dim];
    }

    for(unsigned int index = 0;index < initial_contour.size();++index)
    {
        if(outter_contour[index] && initial_contour[index])
            outter_contour[index] = 1;
        else
            if(outter_contour[index] && !initial_contour[index])
                outter_contour[index] = 0;
            else
                outter_contour[index] = 2;
    }
    image::basic_image<class ImageType::value_type,ImageType::dimension> crop_image(src);

    image::crop(outter_contour,range_min,range_max);
    image::crop(crop_image,range_min,range_max);

    if(consider_region_intensity)
        stochastic_competition_with_lostinfo(crop_image,outter_contour,imp::intensity_enabled(),Zc,Zr);
    else
        stochastic_competition_with_lostinfo(crop_image,outter_contour,imp::intensity_disabled(),Zc,Zr);

    std::replace(outter_contour.begin(),outter_contour.end(),0,1);
    std::replace(outter_contour.begin(),outter_contour.end(),2,0);
    std::fill(initial_contour.begin(),initial_contour.end(),0);
    image::draw(outter_contour,initial_contour,range_min);
\
}


}
}
