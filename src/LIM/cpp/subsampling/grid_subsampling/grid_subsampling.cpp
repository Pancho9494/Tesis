
#include "grid_subsampling.h"


void grid_subsampling(std::vector<PointXYZ>& original_points,
                      std::vector<PointXYZ>& subsampled_points,
                      std::vector<float>& original_features,
                      std::vector<float>& subsampled_features,
                      std::vector<int>& original_classes,
                      std::vector<int>& subsampled_classes,
                      float sampleDl,
                      int verbose) {

	// Initialize variables
	// ******************

	// Number of points in the cloud
	std::size_t N = original_points.size();

	// Dimension of the features
	std::size_t fdim = original_features.size() / N;
	std::size_t ldim = original_classes.size() / N;

	// Limits of the cloud
	PointXYZ minCorner = min_point(original_points);
	PointXYZ maxCorner = max_point(original_points);
	PointXYZ originCorner = floor(minCorner * (1/sampleDl)) * sampleDl;

	// Dimensions of the grid
	std::size_t sampleNX = (std::size_t)floor((maxCorner.x - originCorner.x) / sampleDl) + 1;
	std::size_t sampleNY = (std::size_t)floor((maxCorner.y - originCorner.y) / sampleDl) + 1;
	//std::size_t sampleNZ = (std::size_t)floor((maxCorner.z - originCorner.z) / sampleDl) + 1;

	// Check if features and classes need to be processed
	bool use_feature = original_features.size() > 0;
	bool use_classes = original_classes.size() > 0;


	// Create the sampled map
	// **********************

	// Verbose parameters
	int i = 0;
	int nDisp = N / 100;

	// Initialize variables
	std::size_t iX, iY, iZ, mapIdx;
	std::unordered_map<std::size_t, SampledData> data;

	for (auto& p : original_points) {
		// Position of point in sample map
		iX = (std::size_t)floor((p.x - originCorner.x) / sampleDl);
		iY = (std::size_t)floor((p.y - originCorner.y) / sampleDl);
		iZ = (std::size_t)floor((p.z - originCorner.z) / sampleDl);
		mapIdx = iX + sampleNX*iY + sampleNX*sampleNY*iZ;

		// If not already created, create key
		if (data.count(mapIdx) < 1)
			data.emplace(mapIdx, SampledData(fdim, ldim));

		// Fill the sample map
		if (use_feature && use_classes)
			data[mapIdx].update_all(p, original_features.begin() + i * fdim, original_classes.begin() + i * ldim);
		else if (use_feature)
			data[mapIdx].update_features(p, original_features.begin() + i * fdim);
		else if (use_classes)
			data[mapIdx].update_classes(p, original_classes.begin() + i * ldim);
		else
			data[mapIdx].update_points(p);

		// Display
		i++;
		if (verbose > 1 && i%nDisp == 0)
			std::cout << "\rSampled Map : " << std::setw(3) << i / nDisp << "%";

	}

	// Divide for barycentre and transfer to a vector
	subsampled_points.reserve(data.size());
	if (use_feature)
		subsampled_features.reserve(data.size() * fdim);
	if (use_classes)
		subsampled_classes.reserve(data.size() * ldim);
	for (auto& v : data)
	{
		subsampled_points.push_back(v.second.point * (1.0 / v.second.count));
		if (use_feature)
		{
		    float count = (float)v.second.count;
		    transform(v.second.features.begin(),
                      v.second.features.end(),
                      v.second.features.begin(),
                      [count](float f) { return f / count;});
            subsampled_features.insert(subsampled_features.end(),v.second.features.begin(),v.second.features.end());
		}
		if (use_classes)
		{
		    for (std::size_t i = 0; i < ldim; i++)
		        subsampled_classes.push_back(max_element(v.second.labels[i].begin(), v.second.labels[i].end(),
		        [](const std::pair<int, int>&a, const std::pair<int, int>&b){return a.second < b.second;})->first);
		}
	}

	return;
}


void batch_grid_subsampling(std::vector<PointXYZ>& original_points,
                              std::vector<PointXYZ>& subsampled_points,
                              std::vector<float>& original_features,
                              std::vector<float>& subsampled_features,
                              std::vector<int>& original_classes,
                              std::vector<int>& subsampled_classes,
                              std::vector<int>& original_batches,
                              std::vector<int>& subsampled_batches,
                              float sampleDl,
                              int max_p) {
	// Initialize variables
	// ******************

	int b = 0;
	int sum_b = 0;

	// Number of points in the cloud
	std::size_t N = original_points.size();

	// Dimension of the features
	std::size_t fdim = original_features.size() / N;
	std::size_t ldim = original_classes.size() / N;

	// Handle max_p = 0
	if (max_p < 1)
	    max_p = N;

	// Loop over batches
	// *****************


	for (b = 0; b < original_batches.size(); b++)
	{

	    // Extract batch points features and labels
	    std::vector<PointXYZ> b_o_points = std::vector<PointXYZ>(original_points.begin () + sum_b,
	                                                   original_points.begin () + sum_b + original_batches[b]);

        std::vector<float> b_o_features;
        if (original_features.size() > 0)
        {
            b_o_features = std::vector<float>(original_features.begin () + sum_b * fdim,
                                         original_features.begin () + (sum_b + original_batches[b]) * fdim);
	    }

	    std::vector<int> b_o_classes;
        if (original_classes.size() > 0)
        {
            b_o_classes = std::vector<int>(original_classes.begin () + sum_b * ldim,
                                      original_classes.begin () + sum_b + original_batches[b] * ldim);
	    }


        // Create result containers
        std::vector<PointXYZ> b_s_points;
        std::vector<float> b_s_features;
        std::vector<int> b_s_classes;

        // Compute subsampling on current batch
        grid_subsampling(b_o_points,
                         b_s_points,
                         b_o_features,
                         b_s_features,
                         b_o_classes,
                         b_s_classes,
                         sampleDl,
						 0);

        // Stack batches points features and labels
        // ****************************************

        // If too many points remove some
        if (b_s_points.size() <= max_p)
        {
            subsampled_points.insert(subsampled_points.end(), b_s_points.begin(), b_s_points.end());

            if (original_features.size() > 0)
                subsampled_features.insert(subsampled_features.end(), b_s_features.begin(), b_s_features.end());

            if (original_classes.size() > 0)
                subsampled_classes.insert(subsampled_classes.end(), b_s_classes.begin(), b_s_classes.end());

            subsampled_batches.push_back(b_s_points.size());
        }
        else
        {
            subsampled_points.insert(subsampled_points.end(), b_s_points.begin(), b_s_points.begin() + max_p);

            if (original_features.size() > 0)
                subsampled_features.insert(subsampled_features.end(), b_s_features.begin(), b_s_features.begin() + max_p * fdim);

            if (original_classes.size() > 0)
                subsampled_classes.insert(subsampled_classes.end(), b_s_classes.begin(), b_s_classes.begin() + max_p * ldim);

            subsampled_batches.push_back(max_p);
        }

        // Stack new batch lengths
        sum_b += original_batches[b];
	}

	return;
}