# TVB_C -- A fast implementation of The Virtual Brain's simulation core in C

The code implements a brain network model composed of connected ReducedWongWang neural mass models (Wong & Wang, 2006) with feedback inhibition control (FIC). For more information on the model and FIC please see Deco et al. (2014), Schirner et al. (2018) and Shen et al. (2019).

For more information on The Virtual Brain (TVB) please see 
www.thevirtualbrain.org

For questions and other enquiries please contact 
Michael Schirner (m.schirner@fu-berlin.de) or 
Petra Ritter (petra.ritter@charite.de).

# Usage

```
./tvbii <parameter_file> <subject_id>
```

Example
```
./tvbii param_set_1 UE_20120803
```

• The first argument specifies a text file that contains parameters (see the file 'param_set_1' for an example and description of paramters)

• The second argument specifies the subject-id for the input files contained in the folder _'input'_. Each of the three input files must have <subject_id> as prefix and as suffix either _"_SC_strengths.txt"_, _"_SC_distances.txt"_ or, _"_SC_regionids.txt"_. The three files specify structural connectivity of the brain network model in a sparse matrix format. The enclosed Matlab script _generate_input_SC.m_ generates these input files from Matlab's standard matrix format.

• Results are written into folder _'output'_. File-schema: BOLD_<parameter_file>.txt; the first n lines (n=number of regions) contain two columns each that contain the sum of all input strengths for that region and the J_i value found during FIC tuning, respectively. The following t lines contain n columns each and contain simulated fMRI BOLD activity for t time points.

• The relative folder structure, i.e., the location of the folders 'input' and 'output' relative to the program binary needs to remain stable, otherwise the program won't be able to read or write data.

• Due to optimization reasons, the number of nodes must be divisible by four. If the number of nodes is not divisible by four, "fake" regions must be added that contain zero coupling to other nodes (all zeros in strength matrix).
  
  
# Compilation
  
  Compile the C code with a compiler with support for SSE instructions enabled. I found the following combination of flags yields good performance (in terms of simulation speed) with the GNU C compiler

```
gcc -Wall -std=c99 -msse2 -O3 -ftree-vectorize -ffast-math -funroll-loops -fomit-frame-pointer -m64 -lm main.c -o tvb
```
  
# References
  
  Deco, G., Ponce-Alvarez, A., Hagmann, P., Romani, G. L., Mantini, D., & Corbetta, M. (2014). How local excitation–inhibition ratio impacts the whole brain dynamics. Journal of Neuroscience, 34(23), 7886-7898.
  
  Schirner, M., McIntosh, A. R., Jirsa, V., Deco, G., & Ritter, P. (2018). Inferring multi-scale neural mechanisms with brain network modelling. Elife, 7, e28927.
  
  Shen, K., Bezgin, G., Schirner, M., Ritter, P., Everling, S., McIntosh, A. R. (2019) A macaque connectome for large-scale network simulations in TheVirtualBrain. (under review)
  
  Wong, K. F., & Wang, X. J. (2006). A recurrent network mechanism of time integration in perceptual decisions. Journal of Neuroscience, 26(4), 1314-1328.
