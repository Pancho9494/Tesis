#include "neighbors.h"

void brute_neighbors(std::vector<PointXYZ>& queries, std::vector<PointXYZ>& supports, std::vector<int>& neighbors_indices, float radius, int verbose) {
	int max_count = 0, i0 = 0, i = 0;
	float radius_squared = radius * radius;
	std::vector<std::vector<int>> tmp(queries.size());

	for (PointXYZ& query : queries) {
		i = 0;
		for (PointXYZ& support : supports) {
			if ((query - support).sq_norm() < radius_squared) {
				tmp[i0].push_back(i);
				if (tmp[i0].size() > max_count) {
					max_count = tmp[i0].size();
				}
			}
			i++;
		}
		i0++;
	}

	neighbors_indices.resize(queries.size() * max_count);
	i0 = 0;
	for (std::vector<int>& inds : tmp) {
		for (int j = 0; j < max_count; j++) {
			if (j < inds.size()) {
				neighbors_indices[i0 * max_count + j] = inds[j];
			}
			else {
				neighbors_indices[i0 * max_count + j] = -1;
			}
		}
		i0++;
	}
}

void ordered_neighbors(std::vector<PointXYZ>& queries, std::vector<PointXYZ>& supports, std::vector<int>& neighbors_indices, float radius) {
	float radius_squared = radius * radius;
	int max_count = 0, i0 = 0;
	std::vector<std::vector<int>> tmp(queries.size());
	std::vector<std::vector<float>> dists(queries.size());

	int i, index;
	float d2;
	for (PointXYZ& query : queries) {
		i = 0;
		for (PointXYZ& support : supports) {
		    d2 = (query - support).sq_norm();
			if (d2 < radius_squared) {
			    // Find order of the new point
			    auto it = std::upper_bound(dists[i0].begin(), dists[i0].end(), d2);
			    index = std::distance(dists[i0].begin(), it);

			    // Insert element
                dists[i0].insert(it, d2);
                tmp[i0].insert(tmp[i0].begin() + index, i);

			    // Update max count
				if (tmp[i0].size() > max_count) {
					max_count = tmp[i0].size();
				}
			}
			i++;
		}
		i0++;
	}

	// Reserve the memory
	neighbors_indices.resize(queries.size() * max_count);
	i0 = 0;
	for (std::vector<int>& inds : tmp) {
		for (int j = 0; j < max_count; j++) {
			if (j < inds.size()) {
				neighbors_indices[i0 * max_count + j] = inds[j];
			}
			else {
				neighbors_indices[i0 * max_count + j] = -1;
			}
		}
		i0++;
	}
}

void batch_ordered_neighbors(std::vector<PointXYZ>& queries,
                                std::vector<PointXYZ>& supports,
                                std::vector<int>& q_batches,
                                std::vector<int>& s_batches,
                                std::vector<int>& neighbors_indices,
                                float radius) {
	std::vector<std::vector<int>> tmp(queries.size());
	std::vector<std::vector<float>> dists(queries.size());
	float radius_squared = radius * radius;
	int max_count = 0, i0 = 0;
	int b = 0, sum_qb = 0, sum_sb = 0;
	float d2;
	int i, index;
	std::vector<PointXYZ>::iterator support_it;
	for (PointXYZ& query : queries) {
	    // Check if we changed batch
	    if (i0 == sum_qb + q_batches[b]) {
	        sum_qb += q_batches[b];
	        sum_sb += s_batches[b];
	        b++;
	    }

	    // Loop only over the supports of current batch
		i = 0;
        for(support_it = supports.begin() + sum_sb; support_it < supports.begin() + sum_sb + s_batches[b]; support_it++ ) {
		    d2 = (query - *support_it).sq_norm();
			if (d2 < radius_squared) {
			    // Find order of the new point
			    auto it = std::upper_bound(dists[i0].begin(), dists[i0].end(), d2);
			    index = std::distance(dists[i0].begin(), it);

			    // Insert element
                dists[i0].insert(it, d2);
                tmp[i0].insert(tmp[i0].begin() + index, sum_sb + i);

			    // Update max count
				if (tmp[i0].size() > max_count) {
					max_count = tmp[i0].size();
				}
			}
			i++;
		}
		i0++;
	}

	neighbors_indices.resize(queries.size() * max_count);
	i0 = 0;
	for (std::vector<int>& inds : tmp) {
		for (int j = 0; j < max_count; j++) {
			if (j < inds.size()) {
				neighbors_indices[i0 * max_count + j] = inds[j];
			}
			else {
				neighbors_indices[i0 * max_count + j] = supports.size();
			}
		}
		i0++;
	}
}


void batch_nanoflann_neighbors(std::vector<PointXYZ>& queries,
                                std::vector<PointXYZ>& supports,
                                std::vector<int>& q_batches,
                                std::vector<int>& s_batches,
                                std::vector<int>& neighbors_indices,
                                float radius) {

	std::vector<std::vector<std::pair<std::size_t, float>>> all_inds_dists(queries.size());
	float radius_squared = radius * radius;
	int i0 = 0, max_count = 0, b = 0, sum_qb = 0, sum_sb = 0;
	float d2;

	PointCloud current_cloud;
	nanoflann::KDTreeSingleIndexAdaptorParams tree_params(10 /* max leaf */);
    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud >,PointCloud, 3> kd_tree;


    kd_tree* index;
    // Build KDTree for the first batch element
    current_cloud.pts = std::vector<PointXYZ>(supports.begin() + sum_sb, supports.begin() + sum_sb + s_batches[b]);
    index = new kd_tree(3, current_cloud, tree_params);
    index->buildIndex();


    nanoflann::SearchParams search_params;
    search_params.sorted = true;
	float query_pt[3];
	std::size_t n_matches;
	for (PointXYZ& query : queries) {

	    if (i0 == sum_qb + q_batches[b]) {
	        sum_qb += q_batches[b];
	        sum_sb += s_batches[b];
	        b++;

	        // Change the points
	        current_cloud.pts.clear();
            current_cloud.pts = std::vector<PointXYZ>(supports.begin() + sum_sb, supports.begin() + sum_sb + s_batches[b]);

	        // Build KDTree of the current element of the batch
            delete index;
            index = new kd_tree(3, current_cloud, tree_params);
            index->buildIndex();
	    }

	    // Initial guess of neighbors size
        all_inds_dists[i0].reserve(max_count);

	    // Find neighbors
		query_pt[0] = query.x; query_pt[1] = query.y; query_pt[2] = query.z;
		n_matches = index->radiusSearch(query_pt, radius_squared, all_inds_dists[i0], search_params);

        // Update max count
        if (n_matches > max_count) {
            max_count = n_matches;
		}

        // Increment query idx
		i0++;
	}

	// Reserve the memory
	neighbors_indices.resize(queries.size() * max_count);
	i0 = 0;
	sum_sb = 0;
	sum_qb = 0;
	b = 0;
	for (std::vector<std::pair<std::size_t, float>>& inds_dists : all_inds_dists) {
	    if (i0 == sum_qb + q_batches[b]) {
	        sum_qb += q_batches[b];
	        sum_sb += s_batches[b];
	        b++;
	    }

		for (int j = 0; j < max_count; j++) {
			if (j < inds_dists.size()) {
				neighbors_indices[i0 * max_count + j] = inds_dists[j].first + sum_sb;
			}
			else {
				neighbors_indices[i0 * max_count + j] = supports.size();
			}
		}
		i0++;
	}

	delete index;

	return;
}

