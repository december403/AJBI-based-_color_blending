# AJBI-based--_color_blending
code for paper "An Adaptive Joint Bilateral Interpolation-Based Color Blending Method for Stitched UAV Images"

## usage

#### How to Adjust Program Parameters

In "main.cpp", line 21, 22, 23 change parameters used in algorithm. 

#### How to disable parallel processing feature

In "main.cpp", comment out line 113 and uncomment line 111 will disable the parallel processing feature.

## Enviroment
* Windos 10 64-bit
* Visual Studio 2019
* Visual Studio platform toolset LLVM-clang
* ISO C++ 17 

## High resolution images of blended results
https://drive.google.com/drive/folders/1J-deo3aMMBK72u9mcZAB0r9rmCSAd8gR?usp=sharing



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

5. 2022 June 8:
   * Set the parallel processing feature enable as default.
   * Add five original resolution color corrected results.
   * Add the description of execution enviroment.

6. 2022 August 25:
   * Add high resolution images of blended results
   
