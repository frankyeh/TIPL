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
private:
    std::vector<std::shared_ptr<dicom> > dicom_reader;
    std::vector<std::shared_ptr<nifti> > nifti_reader;
    float orientation_matrix[9];
    float spatial_resolution[3];
    char dim_order[3]; // used to rotate the volume to axial view
    char flip[3];        // used to rotate the volume to axial view

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

    template<class voxel_size_type>
    void get_voxel_size(voxel_size_type voxel_size) const
    {
        std::copy(spatial_resolution,spatial_resolution+3,voxel_size);
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
            dicom_header->get_voxel_size(spatial_resolution);
            dicom_header->get_image_orientation(orientation_matrix);
            free_all();
            dicom_reader.push_back(dicom_header);
            tipl::get_orientation(3,orientation_matrix,dim_order,flip);
            tipl::reorient_vector(spatial_resolution,dim_order);
            tipl::reorient_matrix(orientation_matrix,dim_order,flip);
            return true;
        }
        std::shared_ptr<nifti> nifti_header(new nifti);
        if (nifti_header->load_from_file(file_name))
        {
            nifti_header->get_voxel_size(spatial_resolution);
            nifti_header->get_image_orientation(orientation_matrix);
            free_all();
            nifti_reader.push_back(nifti_header);
            tipl::get_inverse_orientation(3,orientation_matrix,dim_order,flip);
            change_orientation(true,true,false);
            // from +x = Right  +y = Anterior +z = Superior
            // to +x = Left  +y = Posterior +z = Superior
            tipl::reorient_vector(spatial_resolution,dim_order);
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
            std::shared_ptr<dicom> dicom_header(new dicom);
            if (dicom_header->load_from_file(files[index]))
            {
                if(dicom_reader.empty())
                {
                    dicom_header->get_voxel_size(spatial_resolution);
                    dicom_header->get_image_orientation(orientation_matrix);
                }
                dicom_reader.push_back(dicom_header);
                continue;
            }
            std::shared_ptr<nifti> nifti_header(new nifti);
            if (nifti_header->load_from_file(files[index].c_str()))
            {
                if(nifti_reader.empty())
                {
                    nifti_header->get_voxel_size(spatial_resolution);
                    nifti_header->get_image_orientation(orientation_matrix);
                }
                nifti_reader.push_back(nifti_header);
            }
        }
        if(!dicom_reader.empty())
        {
            tipl::vector<3> pos1,pos2;
            dicom_reader[0]->get_left_upper_pos(pos1.begin());
            dicom_reader[1]->get_left_upper_pos(pos2.begin());
            orientation_matrix[6] = pos2[0]-pos1[0];
            orientation_matrix[7] = pos2[1]-pos1[1];
            orientation_matrix[8] = pos2[2]-pos1[2];
            tipl::get_orientation(3,orientation_matrix,dim_order,flip);
        }
        else
            if(!nifti_reader.empty())
            {
                tipl::get_inverse_orientation(3,orientation_matrix,dim_order,flip);
                change_orientation(true,true,false);
                // from +x = Right  +y = Anterior +z = Superior
                // to +x = Left  +y = Posterior +z = Superior
            }
            else
                return false;
        tipl::reorient_vector(spatial_resolution,dim_order);
        tipl::reorient_matrix(orientation_matrix,dim_order,flip);
        return true;
    }
    unsigned int width(void) const{return dicom_reader.empty() ? (nifti_reader.empty() ? 0 : nifti_reader.front()->width()):dicom_reader.front()->width();}
    unsigned int height(void) const{return dicom_reader.empty() ? (nifti_reader.empty() ? 0 : nifti_reader.front()->height()):dicom_reader.front()->height();}
    unsigned int depth(void) const{return (unsigned int)(dicom_reader.size()+nifti_reader.size());}
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
                buffer.resize(tipl::geometry<3>(dicom_reader.front()->width(),
                                             dicom_reader.front()->height(),
                                             dicom_reader.size()));
                for(unsigned int index = 0;index < dicom_reader.size();++index)
					dicom_reader[index]->save_to_buffer(&*buffer.begin()+index*buffer.plane_size(),buffer.plane_size());
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
                buffer.resize(tipl::geometry<3>(nifti_reader.front()->width(),
                                             nifti_reader.front()->height(),
                                             nifti_reader.size()));
                for(unsigned int index = 0;index < nifti_reader.size();++index)
                    nifti_reader[index]->save_to_buffer(&*buffer.begin()+index*buffer.plane_size(),buffer.plane_size());
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
