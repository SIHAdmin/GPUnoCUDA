// This is the Kernel function for executing the MeanShift computations on OpenCL devices
// Copyright 2015 Atomic Object (MIT)

kernel void mean_shift_point(global float2 *points,
                             global float2 *original_points,
                             size_t num_points,
                             float bandwidth,
                             global float2 *shifted_points)

// The MeanShift algorithm bascially finds, for each sample point, the weighted sum of its neighbours.
// The weighting function is a kernel, in our case a gaussian function with sigma given by the BANDWIDTH parameter.
/* The algorithm works as follows:
 1. For each point, find a weighted sum of its neighbours within a local region, defined by the kernel functon (> weight)
    Normalise this by the sum of all weights.
 
 2. Replace each point by this weighted sum, which essectially shifts the point a small way towards the centre of mass of its current neighbourhood.
 
 3. Continue shifting the point, until it reaches the local centrwe of mass -- i.e., it stops moving.
 
 4. This limiting point will be a local maximum in the data (sample) distribution, ie a point of maximum density.
*/
{
    float base_weight = 1. / (bandwidth * sqrt(2. * M_PI_F));  // The Gaussian dist normalisation factor
    float2 shift = { 0, 0 };
    float scale = 0;

    // --> For each point i ..
    size_t i = get_global_id(0);

    // .. add the weighted positions of its neighbours
    for (size_t j = 0; j < num_points; j++) {
        float dist = distance(points[i], original_points[j]);
        float weight = base_weight * exp(-0.5f * pow(dist / bandwidth, 2.f)); // Weight factor for the given neghbour

        // Sum the numerator and denominator of the normalised weighted sum (ie sum / normalisation)
        shift += original_points[j] * weight;  // Weighted neighbour contribution
        scale += weight;                       // Weighting contribution
    }

    shifted_points[i] = shift / scale;         // numerator / denominator
}
