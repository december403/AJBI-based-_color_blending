
#include "utili.h"

Mat seam_mask, warped_tar_img, warped_ref_img, result_from_tar_mask, color_comp_map, propagation_map, discovered_map, test_wavefront_marching, discovered_time_stamp_map, range_map;
int height, width, sup_pixel_size, propagation_coefficient;
vector<Point> seam_pixel_lst, target_pixel_lst, refined_seam_pixel_lst, discarded_seam_pixel_lst, sorted_seam_pixel_lst;
double min_sigma_color, sigma_color, sigma_dist, cost_threshold, sigma_dist_aligned, sigma_dist_misaligned, sigma_dist_hybrid, sigma_color_aligned, sigma_color_misaligned, sigma_color_hybrid;
vector<Mat> propagate_maps, debug_propagate_maps;
Mat anomaly_mask;
float ruler;
Mat test_mask;
int main()
{	
	// Our's wavefront propagation method
	clock_t start, end;
	double cpu_time_used;
	start = clock();



	sigma_dist = 0.5;
	sigma_color = 3;
	min_sigma_color = 0.1;
	//sup_pixel_size = 14;
	cost_threshold = 500.0;
	ruler = 5.0f;
	propagation_coefficient = 20;



	result_from_tar_mask = imread("image/result_from_target.png", IMREAD_GRAYSCALE);
	seam_mask = imread("image/seam_mask.png", IMREAD_GRAYSCALE);
	warped_tar_img = imread("image/warped_target.png", IMREAD_COLOR);
	warped_ref_img = imread("image/warped_reference.png", IMREAD_COLOR);
	Mat result = imread("image/result.png", IMREAD_COLOR);
	height = result_from_tar_mask.rows;
	width = result_from_tar_mask.cols;

	color_comp_map = Mat::zeros(warped_tar_img.rows, warped_tar_img.cols, CV_64FC3);
	propagation_map = Mat::zeros(warped_tar_img.rows, warped_tar_img.cols, CV_8U);
	discovered_map = Mat::zeros(warped_tar_img.rows, warped_tar_img.cols, CV_8U);
	test_wavefront_marching = Mat::zeros(warped_tar_img.rows, warped_tar_img.cols, CV_8U);
	discovered_time_stamp_map = Mat::zeros(warped_tar_img.rows, warped_tar_img.cols, CV_16U);
	range_map = Mat::zeros(warped_tar_img.rows, warped_tar_img.cols, CV_16UC2);



	// generate a list consists of points on stitching line.
	seam_pixel_lst = build_seam_pixel_lst();


	// split and merge approach to classified points on stitching line into two classes. (If the stitching line is slightly missaligned, there might be no misaligned class. )
	//refine_seam_pixel_lst(seam_pixel_lst, refined_seam_pixel_lst, discarded_seam_pixel_lst);
	refine_seam_pixel_lst_based_abs_RGB_diff(seam_pixel_lst, refined_seam_pixel_lst, discarded_seam_pixel_lst);

	anomaly_mask = Mat::zeros(warped_tar_img.rows, warped_tar_img.cols, CV_8U);
	for (Point p : discarded_seam_pixel_lst)
	{
		anomaly_mask.at<uchar>(p) = 255;
	}




	// sort the points on stitching line from one end point to another end point.
	sort_seam_pixel_lst(seam_pixel_lst, sorted_seam_pixel_lst);



	// build the cite range for each in target image. consider the previous wavefront and its neighbor.

	build_range_map_with_side_addition(sorted_seam_pixel_lst);
	//end = clock();
	//cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
	//printf("Time = %f\n", cpu_time_used);
	//imwrite("test_range.png", test_wavefront_marching);

	//return 0;
	// build the cite range for each in target image. consider the previous wavefront only.
	//build_range_map(sorted_seam_pixel_lst);



	// the func. build_range_map and func. build_range_map_plus_side_propagation generates debug image for wavefront marching.
	// in other func. there are also many debugging using the test_wavefront_marching.
	//imwrite("test_range.png", test_wavefront_marching);


	// generate a list consists of points on target images.
	target_pixel_lst = build_target_pixel_lst();


	// generate a list of lists consists of superpixels and lts pixels.
	//vector<vector<Point>> sup_pxl_lst = build_superpixel_lst();
	

	// initialize the color compensation map difference of the seam_pixel_lst. 
	init_color_comp_map(seam_pixel_lst);


	// generate for the pxiel-based color blending.
	vector<vector<Point>> pixel_wise_lst;
	for (Point p : target_pixel_lst)
	{
		vector<Point> a;
		a.push_back(p);
		pixel_wise_lst.push_back(a);
	}

	// to calculate the color for all pixels in target image.
	// update_color_comp_map_range_anomaly_ver(pixel_wise_lst);
	//update_color_comp_map_range_anomaly_ver(sup_pxl_lst);
	update_color_comp_map_range_anomaly_and_parallel_ver(pixel_wise_lst);





	// just add the color of target image with color compensation map.
	build_final_result();
	warped_ref_img.copyTo(warped_tar_img, result_from_tar_mask == 0);
	imwrite("result.png", warped_tar_img);

	end = clock();
	cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf("Time = %f\n", cpu_time_used);

}