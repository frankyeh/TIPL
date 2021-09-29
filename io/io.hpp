#ifndef IO_HPP
#define IO_HPP
#include <vector>
#include <string>
#include <fstream>
#include "nifti.hpp"
#include "dicom.hpp"
#include "../utility/basic_image.hpp"
#include "../numerical/basic_op.hpp"

namespace tipl
{

namespace io
{



class volume{
public:
    std::vector<std::shared_ptr<dicom> > dicom_reader;
    float orientation_matrix[9];
    tipl::shape<3> dim;
    tipl::vector<3> vs;
    uint8_t dim_order[3]; // used to rotate the volume to axial view
    uint8_t flip[3];        // used to rotate the volume to axial view
    std::string error_msg;

    void free_all(void)
    {
        dicom_reader.clear();
    }
    void change_orientation(bool x,bool y,bool z)
    {
        bool xyz[3];
        xyz[0] = x;
        xyz[1] = y;
        xyz[2] = z;
        for(int index = 0;index < 3;++index)
            if(xyz[dim_order[index]])
                flip[index] = !flip[index];
    }
public:
    ~volume(void){free_all();}
    const std::shared_ptr<dicom> get_dicom(unsigned int index) const{return dicom_reader[index];}
    const shape<3>& shape(void) const{return dim;}
    void get_voxel_size(tipl::vector<3,float>& voxel_size) const
    {
        voxel_size = vs;
    }
    template<typename vector_type>
    void get_image_row_orientation(vector_type image_row_orientation) const
    {
        std::copy(orientation_matrix,orientation_matrix+9,image_row_orientation);
    }
    bool load_from_files(const std::vector<std::string>& files)
    {
        if(files.empty())
            return false;
        free_all();
        std::vector<int> image_num;
        float r0[9];
        for (unsigned int index = 0;index < files.size();++index)
        {
            std::shared_ptr<dicom> d(new dicom);
            if (!d->load_from_file(files[index]))
            {
                error_msg = "failed to read ";
                error_msg += files[index];
                return false;
            }
            if(index)
            {
                float rn[9];
                d->get_image_orientation(rn);
                if(rn[0] != r0[0])
                {
                    error_msg = "inconsistent image orientation at ";
                    error_msg += files[index];
                    return false;
                }
            }
            else
                d->get_image_orientation(r0);
            // get image sequence
            image_num.push_back(0);
            std::istringstream(d->get_image_num()) >> image_num.back();
            dicom_reader.push_back(d);
        }
        // sort dicom according to the image num
        {
            auto order = tipl::arg_sort(image_num.size(),[&](int i,int j){return image_num[i] < image_num[j];});
            std::vector<std::shared_ptr<dicom> > new_dicom_reader(order.size());
            for(size_t i = 0;i < order.size();++i)
                new_dicom_reader[i] = dicom_reader[order[i]];
            new_dicom_reader.swap(dicom_reader);
        }

        {
            dim = tipl::shape<3>(dicom_reader.front()->width(),
                                 dicom_reader.front()->height(),
                                 dicom_reader.size());
            dicom_reader.front()->get_voxel_size(vs);
            dicom_reader.front()->get_image_orientation(orientation_matrix);
            if(vs[2] == 0.0f)
                vs[2] = std::fabs(dicom_reader[1]->get_slice_location()-
                                                  dicom_reader[0]->get_slice_location());
            // the last row of the orientation matrix should be derived from slice location
            // otherwise, could be flipped in the saggital slices
            {
                tipl::vector<3> pos1,pos2;
                dicom_reader[0]->get_left_upper_pos(pos1.begin());
                dicom_reader[1]->get_left_upper_pos(pos2.begin());
                orientation_matrix[6] = pos2[0]-pos1[0];
                orientation_matrix[7] = pos2[1]-pos1[1];
                orientation_matrix[8] = pos2[2]-pos1[2];
            }
            tipl::get_orientation(3,orientation_matrix,dim_order,flip);

        }
        tipl::reorient_vector(vs,dim_order);
        tipl::reorient_matrix(orientation_matrix,dim_order,flip);
        return true;
    }
    template<typename image_type>
    void get_untouched_image(image_type& source) const
    {
        if(dicom_reader.empty())
            return;
        if(dicom_reader.size() == 1)
            *dicom_reader.front() >> source;
        else
        {
            source.resize(dim);
            for(size_t index = 0;index < dicom_reader.size();++index)
                dicom_reader[index]->save_to_buffer(&*source.begin()+index*dim.plane_size(),dim.plane_size());
        }
    }

    template<typename image_type>
    void save_to_image(image_type& I) const
    {
        tipl::image<typename image_type::value_type,3> buffer;
        get_untouched_image(buffer);
        tipl::reorder(buffer,I,dim_order,flip); // to LPS
    }

    template<typename image_type>
    const volume& operator>>(image_type& I) const
    {
        save_to_image(I);
        return *this;
    }
};

}// io

}// image
#endif
