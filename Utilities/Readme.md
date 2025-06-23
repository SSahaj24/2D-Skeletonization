## The Codebase for Post-Processing the Skeletonised Data - A few utilities used in the Manuscript

- `LengthWisePlot.m` - Given all linelengths, region-wise, per category, this code is used to calculate the sorted top LineLength in each region, per category (WSI/STP/Single Neurons), the Top common regions with the highest line lengths
- `calc_lineLength_inRAF.m` - Given a 20 micron labelled volume, line strings in each category, this code is used to calculate  LineLength in each region, per category
- `supriseIndex.m` - Given line densities in each category, this code is used to calculate the Surprise Index of each Neuron from the Mouselight or ION dataset.
- `wiNN_3D.m` - Weighted Nearest Neighbour algorithm

*Add `load_np.m` in the MATLAB execution path*
