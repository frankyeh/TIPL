#ifndef ML_K_MEANS_HPP
#define ML_K_MEANS_HPP

namespace image{

namespace ml{


template<class attribute_type,class classifications_iterator_type>
void k_means_clustering(const normalized_attributes<attribute_type>& attributes,
                        classifications_iterator_type classification,size_t k)
{
    size_t sample_size = attributes.size();
    size_t attribute_dimension = attributes.attribute_dimension();
    for (size_t index = 0;index < sample_size;++index)
        classification[index] = index%k;
    size_t change_cluster = 0;
    do
    {
        // E-step
        std::vector<std::vector<attribute_type> > means(k);
        for (size_t index = 0;index < k;++index)
        {
            std::vector<attribute_type> cur_mean(attribute_dimension);
            size_t count = 0;
            for (size_t j = 0;j < sample_size;++j)
                if (classification[j] == index)
                {
                    for (size_t k = 0;k < attribute_dimension;++k)
                        cur_mean[k] += attributes[j][k];
                    ++count;
                }
            if (count)
                for (size_t k = 0;k < attribute_dimension;++k)
                    cur_mean[k] /= (attribute_type)count;
            means[index].swap(cur_mean);
        }
        // M-step
        change_cluster = 0;
        for (size_t j = 0;j < sample_size;++j)
        {
            attribute_type min_dis = std::numeric_limits<attribute_type>::max();
            size_t min_cluster = 0;
            for (size_t index = 0;index < k;++index)
            {
                attribute_type dis2 = 0;
                for (size_t i = 0;i < attribute_dimension;++i)
                {
                    attribute_type d = attributes[j][i] - means[index][i];
                    dis2 += d*d;
                    if (dis2 > min_dis)
                        break;
                }
                if (dis2 < min_dis)
                {
                    min_dis = dis2;
                    min_cluster = index;
                }
            }
            if (classification[j] != min_cluster)
            {
                classification[j] = min_cluster;
                ++change_cluster;
            }
        }
    }
    while (change_cluster);
}


template<class attribute_type,class classification_type>
class k_means
{
protected:
    size_t k;
public:

public:
    k_means(size_t k_):k(k_) {}

    template<class attributes_iterator_type,class classifications_iterator_type>
    void operator()(attributes_iterator_type attributes_from,
                    attributes_iterator_type attributes_to,
                    size_t attribute_dimension,
                    classifications_iterator_type classifications_from)
    {
        normalized_attributes<attribute_type> attributes(attributes_from,attributes_to,attribute_dimension);
        k_means_clustering(attributes,classifications_from,k);
    }

    template<class attributes_iterator_type,class classifications_iterator_type>
    void operator()(attributes_iterator_type attributes_from,
                    attributes_iterator_type attributes_to,
                    classifications_iterator_type classifications_from)
    {
        normalized_attributes<attribute_type> attributes(attributes_from,attributes_to);
        k_means_clustering(attributes,classifications_from,k);
    }
};

}// ml

}// image


#endif//ML_K_MEANS_HPP
