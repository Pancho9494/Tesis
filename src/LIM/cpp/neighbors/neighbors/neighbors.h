

#include "../../utils/cloud/cloud.h"
#include "../../utils/nanoflann/nanoflann.hpp"

#include <set>
#include <cstdint>

void ordered_neighbors(std::vector<PointXYZ>& queries,
                        std::vector<PointXYZ>& supports,
                        std::vector<int>& neighbors_indices,
                        float radius);

void batch_ordered_neighbors(std::vector<PointXYZ>& queries,
                                std::vector<PointXYZ>& supports,
                                std::vector<int>& q_batches,
                                std::vector<int>& s_batches,
                                std::vector<int>& neighbors_indices,
                                float radius);

void batch_nanoflann_neighbors(std::vector<PointXYZ>& queries,
                                std::vector<PointXYZ>& supports,
                                std::vector<int>& q_batches,
                                std::vector<int>& s_batches,
                                std::vector<int>& neighbors_indices,
                                float radius);
