# AJBI-based--_color_blending
code for paper "An Adaptive Joint Bilateral Interpolation-Based Color Blending Method for Stitched UAV Images"

## usage

#### How to Adjust Program Parameters

In "main.cpp", line 21, 22, 23 change parameters used in algorithm. 

#### How to enable parallel processing feature

In "main.cpp", comment out line 111 and uncomment line 113 will enable the parallel processing feature.






## changelog

1. 2022 March 6: 
   * first commit (wrong default setting of our code, so the executed configuration is superpixel-based method, which led to inconsistency between the experimental result in our paper and the code generated result.)
2. 2022 April 17: 
   * Change the default setting of our code to pixel-based, so the code generated result is the same as the experimental result shown in our paper.

3. 2022 May 1: 
   * Comment out  the superpixel based method related code. 
4.  2022 May 18: 
   * Add the parallel processing feature to our code, so the execution time can be greatly reduced. (from around 250 seconds to 30 seconds, and with the compilation toolset set to LLVM in Visual Studio 2019, the execution time can be further reduced to around  3 seconds.)
   * Add high resolution images from experimental result.

