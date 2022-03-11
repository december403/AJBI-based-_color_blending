#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <opencv2/ximgproc.hpp>
#include <opencv2/core/types.hpp>
#include <stack>
#include <limits>
#include <queue>
#include <algorithm>

using namespace cv;
using namespace std;
extern Mat seam_mask, warped_tar_img, warped_ref_img, result_from_tar_mask, color_comp_map, propagation_map, discovered_map, test_wavefront_marching, discovered_time_stamp_map, range_map;
extern int height, width, sup_pixel_size, propagation_coefficient;
extern vector<Point> seam_pixel_lst, target_pixel_lst, refined_seam_pixel_lst, discarded_seam_pixel_lst, sorted_seam_pixel_lst;
extern double min_sigma_color, sigma_color, sigma_dist, cost_threshold, sigma_dist_aligned, sigma_dist_misaligned, sigma_dist_hybrid, sigma_color_aligned, sigma_color_misaligned, sigma_color_hybrid;
extern vector<double> average_color;
extern float ruler;
extern vector<Mat> propagate_maps, debug_propagate_maps;
extern Mat anomaly_mask;
extern Mat test_mask;

//extern vector<vector<Point>> sup_pxl_lst;
enum PIXEL_STATE { ALIGNED = 60, MISALIGNED = 120, HYBRID = 255 };

vector<Point> build_seam_pixel_lst();
void init_color_comp_map(vector<Point>& seam_pixel_lst);
void update_color_comp_map(vector<vector<Point>>& sup_pxl_lst);
void update_color_comp_map_range(vector<vector<Point>>& sup_pxl_lst);
vector<double> get_color_comp_value_range(int x, int y);
void update_color_comp_map_avg(vector<vector<Point>>& sup_pxl_lst);
vector<double> get_color_comp_value(int x, int y);
vector<double> get_color_comp_value_avg(int x, int y, int r, int g, int b);
void build_range_map_plus_side_propagation(vector<Point> sorted_seam_pixel_lst);

void build_final_result();
vector<cv::Point> build_target_pixel_lst();
vector<vector<Point>> build_superpixel_lst();
void refine_seam_pixel_lst(vector<Point> seam_pixel_lst, vector<Point>& refined_seam_pixel_lst);
vector<Point> find_next_wavefront(vector<Point> current_wavefront);
void refine_seam_pixel_lst(vector<Point> seam_pixel_lst, vector<Point>& refined_seam_pixel_lst, vector<Point>& discarded_seam_pixel_lst);

void build_range_map(vector<Point> sorted_seam_pixel_lst);


void sort_seam_pixel_lst(vector<Point> seam_pixel_lst, vector<Point>& sorted_seam_pixel_lst);



// discarded functions
//void build_propagate_map_v2();
//vector<double> get_color_comp_value_range_v2(int x, int y);
//void update_color_comp_map_range_v2(vector<vector<Point>>& sup_pxl_lst);

vector<double> get_color_comp_value_my_attempt(int x, int y);
void update_color_comp_map_my_attempt(vector<vector<Point>>& sup_pxl_lst);
void home_made_median_filter_at_end_ver();
void build_range_map_plus_side_propagation_median_filter(vector<Point> sorted_seam_pixel_lst);
void home_made_median_filter_at_end_ver_length_ver();
void build_range_map_plus_side_propagation_mean_filter_progressive(vector<Point> sorted_seam_pixel_lst);
void build_range_map_with_side_addition(vector<Point> sorted_seam_pixel_lst);

void build_range_map_plus_side_propagation_median_filter_progressive(vector<Point> sorted_seam_pixel_lst);


void update_color_comp_map_range_color_diff_only(vector<vector<Point>>& sup_pxl_lst);
vector<double> get_color_comp_value_range_color_diff_only(int x, int y);

void update_color_comp_map_range_anomaly_ver(vector<vector<Point>>& sup_pxl_lst);
vector<double> get_color_comp_value_range_anomaly_ver(int x, int y);


// discarded
void refine_seam_pixel_lst_based_abs_RGB_diff(vector<Point> seam_pixel_lst, vector<Point>& refined_seam_pixel_lst, vector<Point>& discarded_seam_pixel_lst);
