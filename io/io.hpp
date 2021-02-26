#ifndef IO_HPP
#define IO_HPP
#include <vector>
#include <string>
#include <fstream>
#include "nifti.hpp"
#include "dicom.hpp"
#include "tipl/utility/basic_image.hpp"
#include "tipl/numerical/basic_op.hpp"

namespace tipl
{

namespace io
{



class volume{
public:
    std::vector<std::shared_ptr<dicom> > dicom_reader;
    std::vector<std::shared_ptr<nifti> > nifti_reader;
    float orientation_matrix[9];
    tipl::geometry<3> dim;
    tipl::vector<3,float> vs;
    uint8_t dim_order[3]; // used to rotate the volume to axial view
    uint8_t flip[3];        // used to rotate the volume to axial view

    void free_all(void)
    {
        dicom_reader.clear();
        nifti_reader.clear();
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
    const std::shared_ptr<nifti> get_nifti(unsigned int index) const{return nifti_reader[index];}
    const geometry<3>& geo(void) const{return dim;}
    void get_voxel_size(tipl::vector<3,float>& voxel_size) const
    {
        voxel_size = vs;
    }

    template<class vector_type>
    void get_image_row_orientation(vector_type image_row_orientation) const
    {
        std::copy(orientation_matrix,orientation_matrix+9,image_row_orientation);
    }


    template<class file_name_type>
    bool load_from_file(const file_name_type& file_name)
    {
        std::shared_ptr<dicom> dicom_header(new dicom);
        if (dicom_header->load_from_file(file_name))
        {
            dicom_header->get_image_dimension(dim);
            dicom_header->get_voxel_size(vs);
            dicom_header->get_image_orientation(orientation_matrix);
            free_all();
            dicom_reader.push_back(dicom_header);
            tipl::get_orientation(3,orientation_matrix,dim_order,flip);
            tipl::reorient_vector(vs,dim_order);
            tipl::reorient_matrix(orientation_matrix,dim_order,flip);
            return true;
        }
        std::shared_ptr<nifti> nifti_header(new nifti);
        if (nifti_header->load_from_file(file_name))
        {
            nifti_header->get_image_dimension(dim);
            nifti_header->get_voxel_size(vs);
            nifti_header->get_image_orientation(orientation_matrix);
            free_all();
            nifti_reader.push_back(nifti_header);
            tipl::get_inverse_orientation(3,orientation_matrix,dim_order,flip);
            change_orientation(true,true,false);
            // from +x = Right  +y = Anterior +z = Superior
            // to +x = Left  +y = Posterior +z = Superior
            tipl::reorient_vector(vs,dim_order);
            tipl::reorient_matrix(orientation_matrix,dim_order,flip);
            return true;
        }
        return false;
    }
    template<class string_list_type>
    bool load_from_files(const string_list_type& files,unsigned int count)
    {
        if(count == 1)
            return load_from_file(files[0]);
        free_all();
        for (unsigned int index = 0;index < count;++index)
        {
            std::shared_ptr<dicom> d(new dicom);
            if (d->load_from_file(files[index]))
            {
                dicom_reader.push_back(d);
                continue;
            }
            std::shared_ptr<nifti> n(new nifti);
            if (n->load_from_file(files[index].c_str()))
            {
                nifti_reader.push_back(n);
            }
        }
        if(!dicom_reader.empty())
        {
            // remove first slice if it has different imag orientation
            {
                float r1[9],r2[9];
                dicom_reader[0]->get_image_orientation(r1);
                dicom_reader[1]->get_image_orientation(r2);
                if(r1[0] != r2[0])
                    dicom_reader.erase(dicom_reader.begin());
            }
            if(dicom_reader.size() < 2)
                return false;
            dim = tipl::geometry<3>(dicom_reader.front()->width(),
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
        else
            if(!nifti_reader.empty())
            {
                dim = tipl::geometry<3>(nifti_reader.front()->width(),
                                        nifti_reader.front()->height(),
                                        nifti_reader.size());
                nifti_reader.front()->get_voxel_size(vs);
                nifti_reader.front()->get_image_orientation(orientation_matrix);
                tipl::get_inverse_orientation(3,orientation_matrix,dim_order,flip);
                change_orientation(true,true,false);
                // from +x = Right  +y = Anterior +z = Superior
                // to +x = Left  +y = Posterior +z = Superior
            }
            else
                return false;
        tipl::reorient_vector(vs,dim_order);
        tipl::reorient_matrix(orientation_matrix,dim_order,flip);
        return true;
    }
    template<class image_type>
    void save_to_image(image_type& source) const
    {
        typedef typename image_type::value_type value_type;
        tipl::image<value_type,3> buffer;
        if(!dicom_reader.empty())
        {
            if(dicom_reader.size() == 1)
                *dicom_reader.front() >> buffer;
            else
            {
                buffer.resize(dim);
                for(size_t index = 0;index < dicom_reader.size();++index)
                    dicom_reader[index]->save_to_buffer(&*buffer.begin()+index*dim.plane_size(),dim.plane_size());
            }
            tipl::reorder(buffer,source,dim_order,flip);
            return;
        }
        if(!nifti_reader.empty())
        {
            if(nifti_reader.size() == 1)
                *nifti_reader.front() >> buffer;
            else
            {
                buffer.resize(dim);
                for(size_t index = 0;index < nifti_reader.size();++index)
                    nifti_reader[index]->save_to_buffer(&*buffer.begin()+index*dim.plane_size(),dim.plane_size());
            }
            tipl::reorder(buffer,source,dim_order,flip);
            return;
        }
    }

    template<class image_type>
    const volume& operator>>(image_type& source) const
    {
        save_to_image(source);
        return *this;
    }
};

}// io

}// image
#endif
