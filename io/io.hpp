#ifndef IO_HPP
#define IO_HPP
#include <vector>
#include <string>
#include "nifti.hpp"
#include "dicom.hpp"
#include "image/utility/basic_image.hpp"
#include "image/numerical/basic_op.hpp"

namespace image
{

namespace io
{

class volume{
private:
    std::vector<dicom*> dicom_reader;
    std::vector<nifti*> nifti_reader;
    float orientation_matrix[9];
    float spatial_resolution[3];
    char dim_order[3]; // used to rotate the volume to axial view
    char flip[3];        // used to rotate the volume to axial view

    void free_all(void)
    {
        for(int index = 0;index < dicom_reader.size();++index)
            delete dicom_reader[index];
        for(int index = 0;index < nifti_reader.size();++index)
            delete nifti_reader[index];
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

    void reorientation(void)
    {
        // now reordering the spatial information
        {
            float sr[3];
            std::copy(spatial_resolution,spatial_resolution+3,sr);
            for(unsigned int index = 0;index < 3;++index)
                spatial_resolution[dim_order[index]] = sr[index];
        }

                // now reordering the orientation information
        {
            float orientation_matrix_[9];
            std::copy(orientation_matrix,orientation_matrix+9,orientation_matrix_);
            for(unsigned int index = 0,ptr = 0;index < 3;++index,ptr += 3)
                if(flip[index])
                {
                    orientation_matrix_[ptr] = -orientation_matrix_[ptr];
                    orientation_matrix_[ptr+1] = -orientation_matrix_[ptr+1];
                    orientation_matrix_[ptr+2] = -orientation_matrix_[ptr+2];
                }
            for(unsigned int index = 0;index < 3;++index)
            std::copy(orientation_matrix_+index*3,
                      orientation_matrix_+index*3+3,
                      orientation_matrix+dim_order[index]*3);
        }
    }

public:
    ~volume(void){free_all();}

    template<typename voxel_size_type>
    void get_voxel_size(voxel_size_type voxel_size) const
    {
        std::copy(spatial_resolution,spatial_resolution+3,voxel_size);
    }

    template<typename vector_type>
    void get_image_row_orientation(vector_type image_row_orientation) const
    {
        std::copy(orientation_matrix,orientation_matrix+9,image_row_orientation);
    }


    template<typename file_name_type>
    bool load_from_file(const file_name_type& file_name)
    {
        std::auto_ptr<dicom> dicom_header(new dicom);
        if (dicom_header->load_from_file(file_name))
        {
            dicom_header->get_voxel_size(spatial_resolution);
            dicom_header->get_image_orientation(orientation_matrix);
            free_all();
            dicom_reader.push_back(dicom_header.release());
            image::get_orientation(3,orientation_matrix,dim_order,flip);
            reorientation();
            return true;
        }
        std::auto_ptr<nifti> nifti_header(new nifti);
        if (nifti_header->load_from_file(file_name))
        {
            nifti_header->get_voxel_size(spatial_resolution);
            nifti_header->get_image_orientation(orientation_matrix);
            free_all();
            nifti_reader.push_back(nifti_header.release());
            image::get_inverse_orientation(3,orientation_matrix,dim_order,flip);
            change_orientation(true,true,false);
            // from +x = Right  +y = Anterior +z = Superior
            // to +x = Left  +y = Posterior +z = Superior
            reorientation();
            return true;
        }
        return false;
    }
    template<typename string_list_type>
    bool load_from_files(const string_list_type& files,unsigned int count)
    {
        if(count == 1)
            return load_from_file(files[0]);
        free_all();
        for (unsigned int index = 0;index < count;++index)
        {
            std::auto_ptr<dicom> dicom_header(new dicom);
            if (dicom_header->load_from_file(files[index]))
            {
                if(dicom_reader.empty())
                {
                    dicom_header->get_voxel_size(spatial_resolution);
                    dicom_header->get_image_orientation(orientation_matrix);
                }
                dicom_reader.push_back(dicom_header.release());
                continue;
            }
            std::auto_ptr<nifti> nifti_header(new nifti);
            if (nifti_header->load_from_file(files[index].c_str()))
            {
                if(nifti_reader.empty())
                {
                    nifti_header->get_voxel_size(spatial_resolution);
                    nifti_header->get_image_orientation(orientation_matrix);
                }
                nifti_reader.push_back(nifti_header.release());
            }
        }
        if(!dicom_reader.empty())
            image::get_orientation(3,orientation_matrix,dim_order,flip);
        else
            if(!nifti_reader.empty())
            {
                image::get_inverse_orientation(3,orientation_matrix,dim_order,flip);
                change_orientation(true,true,false);
                // from +x = Right  +y = Anterior +z = Superior
                // to +x = Left  +y = Posterior +z = Superior
            }
            else
                return false;
        reorientation();
        return true;
    }

    template<typename image_type>
    void save_to_image(image_type& source) const
    {
        typedef typename image_type::value_type value_type;
        image::basic_image<value_type,3> buffer;
        if(!dicom_reader.empty())
        {
            if(dicom_reader.size() == 1)
                *dicom_reader.front() >> buffer;
            else
            {
                buffer.resize(image::geometry<3>(dicom_reader.front()->width(),
                                             dicom_reader.front()->height(),
                                             dicom_reader.size()));
                for(unsigned int index = 0;index < dicom_reader.size();++index)
					dicom_reader[index]->save_to_buffer(&*buffer.begin()+index*buffer.plane_size(),buffer.plane_size());
            }
            image::reorder(buffer,source,dim_order,flip);
            return;
        }
        if(!nifti_reader.empty())
        {
            if(nifti_reader.size() == 1)
                *nifti_reader.front() >> buffer;
            else
            {
                buffer.resize(image::geometry<3>(nifti_reader.front()->width(),
                                             nifti_reader.front()->height(),
                                             nifti_reader.size()));
                for(unsigned int index = 0;index < nifti_reader.size();++index)
                    nifti_reader[index]->save_to_buffer(&*buffer.begin()+index*buffer.plane_size(),buffer.plane_size());
            }
            image::reorder(buffer,source,dim_order,flip);
            return;
        }
    }

    template<typename image_type>
    const volume& operator>>(image_type& source) const
    {
        save_to_image(source);
        return *this;
    }
};

}// io

}// image
#endif
