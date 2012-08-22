#include <vector>

namespace image{
struct disjoint_set{

    std::vector<unsigned int> rank;
    std::vector<unsigned int> label;

    unsigned int find_set(unsigned int pos)
    {
        unsigned int set = pos;
        if(set != label[set])
        {
            do{set = label[set];}
            while(set != label[set]);
            label[pos] = set;
        }
        return set;
    }
    unsigned int join_set(unsigned int set1,unsigned int set2)
    {
        if(set1 == set2)
            return set1;
        if(rank[set1] > rank[set2])
            std::swap(set1,set2);
        label[set1] = set2;
        if(rank[set1] == rank[set2])
            ++rank[set2];
        return set2;
    }
    void flatten(void)
    {
        for(unsigned int index = 0;index < label.size();++index)
            find_set(index);

    }
    
};

}
