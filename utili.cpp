#include <math.h>
#include <iostream>
#include "utili.h"
using namespace cv;
using namespace std;


vector<cv::Point> build_seam_pixel_lst()
{
	vector<Point> locations;
	cv::findNonZero(seam_mask, locations);
	return locations;
}

vector<cv::Point> build_target_pixel_lst()
{
	vector<Point> locations;
	cv::findNonZero(result_from_tar_mask - seam_mask, locations);
	return locations;
}

void init_color_comp_map(vector<Point> &seam_pixel_lst)
{
	for (int i = 0; i < seam_pixel_lst.size(); i++)
	{
		int x = seam_pixel_lst[i].x;
		int y = seam_pixel_lst[i].y;
		color_comp_map.at<Vec3d>(y, x)[0] = double(warped_ref_img.at<Vec3b>(y, x)[0]) - double(warped_tar_img.at<Vec3b>(y, x)[0]);
		color_comp_map.at<Vec3d>(y, x)[1] = double(warped_ref_img.at<Vec3b>(y, x)[1]) - double(warped_tar_img.at<Vec3b>(y, x)[1]);
		color_comp_map.at<Vec3d>(y, x)[2] = double(warped_ref_img.at<Vec3b>(y, x)[2]) - double(warped_tar_img.at<Vec3b>(y, x)[2]);
	}
}

void update_color_comp_map(vector<vector<Point>>& sup_pxl_lst)
{

	for (vector<Point> lst : sup_pxl_lst)
	{
		int x = lst[0].x;
		int y = lst[0].y;
		vector<double> color_comp = get_color_comp_value(x, y);
		color_comp_map.at<Vec3d>(y, x)[0] = color_comp[0];
		color_comp_map.at<Vec3d>(y, x)[1] = color_comp[1];
		color_comp_map.at<Vec3d>(y, x)[2] = color_comp[2];
	}
//#pragma omp parallel for 
	for (vector<Point> lst : sup_pxl_lst)
	{
		int x = lst[0].x;
		int y = lst[0].y;
		double aa = color_comp_map.at<Vec3d>(y, x)[0];
		double bb = color_comp_map.at<Vec3d>(y, x)[1];
		double cc = color_comp_map.at<Vec3d>(y, x)[2];
		for (Point pt : lst)
		{
			int x = pt.x;
			int y = pt.y;
			color_comp_map.at<Vec3d>(y, x)[0] = aa;
			color_comp_map.at<Vec3d>(y, x)[1] = bb;
			color_comp_map.at<Vec3d>(y, x)[2] = cc;
		}
	}
}

void update_color_comp_map_my_attempt(vector<vector<Point>>& sup_pxl_lst)
{
	for (vector<Point> lst : sup_pxl_lst)
	{
		int x = lst[0].x;
		int y = lst[0].y;
		vector<double> color_comp = get_color_comp_value_my_attempt(x, y);
		color_comp_map.at<Vec3d>(y, x)[0] = color_comp[0];
		color_comp_map.at<Vec3d>(y, x)[1] = color_comp[1];
		color_comp_map.at<Vec3d>(y, x)[2] = color_comp[2];
	}
	for (vector<Point> lst : sup_pxl_lst)
	{
		int x = lst[0].x;
		int y = lst[0].y;
		double aa = color_comp_map.at<Vec3d>(y, x)[0];
		double bb = color_comp_map.at<Vec3d>(y, x)[1];
		double cc = color_comp_map.at<Vec3d>(y, x)[2];
		for (Point pt : lst)
		{
			int x = pt.x;
			int y = pt.y;
			color_comp_map.at<Vec3d>(y, x)[0] = aa;
			color_comp_map.at<Vec3d>(y, x)[1] = bb;
			color_comp_map.at<Vec3d>(y, x)[2] = cc;
		}
	}
}

vector<double> get_color_comp_value_my_attempt(int x, int y)
{
	vector<double> color_comp = { 0.0, 0.0, 0.0 };
	double weight_sum = 0.0;
	double a, b;
	int wavefront_num = discovered_time_stamp_map.at<ushort>(y, x);

	for (int i = 0; i < refined_seam_pixel_lst.size(); i++)
	{
		int x_seam = refined_seam_pixel_lst[i].x;
		int y_seam = refined_seam_pixel_lst[i].y;

		double weight;
		double dist_diff = pow((x - x_seam), 2) + pow((y - y_seam), 2);
		double color_diff = (pow((double(warped_tar_img.at<Vec3b>(y, x)[0]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[0])), 2)
			+ pow((double(warped_tar_img.at<Vec3b>(y, x)[1]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[1])), 2)
			+ pow((double(warped_tar_img.at<Vec3b>(y, x)[2]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[2])), 2)) / 65025;

		// x^2
		//a = color_diff / sigma_color / sigma_color * -1.0 / wavefront_num / wavefront_num / wavefront_num / wavefront_num;
		//b = dist_diff / sigma_dist / sigma_dist / width / width * -1.0 / wavefront_num / wavefront_num / wavefront_num / wavefront_num;

		// x
		double ration = 1;
		a = color_diff / sigma_color / sigma_color * -1.0 / wavefront_num / wavefront_num / ration/ ration;
		b = dist_diff / sigma_dist / sigma_dist / width / width * -1.0 / wavefront_num / wavefront_num / ration / ration;
		//b = dist_diff / sigma_dist / sigma_dist / width / width * -1.0 / wavefront_num / wavefront_num ;

		weight = exp(a) * exp(b);

		color_comp[0] += color_comp_map.at<Vec3d>(y_seam, x_seam)[0] * weight;
		color_comp[1] += color_comp_map.at<Vec3d>(y_seam, x_seam)[1] * weight;
		color_comp[2] += color_comp_map.at<Vec3d>(y_seam, x_seam)[2] * weight;

		weight_sum += weight;
	}
	if (weight_sum == 0.0)
	{
		color_comp[0] = 0.0;
		color_comp[1] = 0.0;
		color_comp[2] = 0.0;
	}
	else
	{
		color_comp[0] = color_comp[0] / weight_sum;
		color_comp[1] = color_comp[1] / weight_sum;
		color_comp[2] = color_comp[2] / weight_sum;
	}

	return color_comp;
}


void update_color_comp_map_range(vector<vector<Point>>& sup_pxl_lst)
{
	for (vector<Point> lst : sup_pxl_lst)
	{
		int x = lst[0].x;
		int y = lst[0].y;
		vector<double> color_comp = get_color_comp_value_range(x, y);
		color_comp_map.at<Vec3d>(y, x)[0] = color_comp[0];
		color_comp_map.at<Vec3d>(y, x)[1] = color_comp[1];
		color_comp_map.at<Vec3d>(y, x)[2] = color_comp[2];
	}
	for (vector<Point> lst : sup_pxl_lst)
	{
		int x = lst[0].x;
		int y = lst[0].y;
		double aa = color_comp_map.at<Vec3d>(y, x)[0];
		double bb = color_comp_map.at<Vec3d>(y, x)[1];
		double cc = color_comp_map.at<Vec3d>(y, x)[2];
		for (Point pt : lst)
		{
			int x = pt.x;
			int y = pt.y;
			color_comp_map.at<Vec3d>(y, x)[0] = aa;
			color_comp_map.at<Vec3d>(y, x)[1] = bb;
			color_comp_map.at<Vec3d>(y, x)[2] = cc;
		}
	}
}

vector<double> get_color_comp_value_range(int x, int y)
{
	vector<double> color_comp = { 0.0, 0.0, 0.0 };
	double weight_sum = 0.0;
	double a, b;
	int low_bound = range_map.at<Vec2w>(y, x)[0];
	int high_bound = range_map.at<Vec2w>(y, x)[1];
	int range_num = high_bound - low_bound + 1;

	for (int i = 0; i < range_num; i++)
	{
		int x_seam = sorted_seam_pixel_lst[i+low_bound].x;
		int y_seam = sorted_seam_pixel_lst[i+low_bound].y;

		double weight;
		double dist_diff = pow((x - x_seam), 2) + pow((y - y_seam), 2);
		double color_diff = (pow((double(warped_tar_img.at<Vec3b>(y, x)[0]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[0])), 2)
			+ pow((double(warped_tar_img.at<Vec3b>(y, x)[1]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[1])), 2)
			+ pow((double(warped_tar_img.at<Vec3b>(y, x)[2]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[2])), 2)) / 65025;

		a = color_diff / sigma_color / sigma_color * -1.0;
		b = dist_diff / sigma_dist / sigma_dist / width / width * -1.0;
		weight = exp(a) * exp(b);

		color_comp[0] += color_comp_map.at<Vec3d>(y_seam, x_seam)[0] * weight;
		color_comp[1] += color_comp_map.at<Vec3d>(y_seam, x_seam)[1] * weight;
		color_comp[2] += color_comp_map.at<Vec3d>(y_seam, x_seam)[2] * weight;

		weight_sum += weight;
	}
	if (weight_sum == 0.0)
	{
		color_comp[0] = 0.0;
		color_comp[1] = 0.0;
		color_comp[2] = 0.0;
	}
	else
	{
		color_comp[0] = color_comp[0] / weight_sum;
		color_comp[1] = color_comp[1] / weight_sum;
		color_comp[2] = color_comp[2] / weight_sum;
	}

	return color_comp;
}

void update_color_comp_map_range_anomaly_and_parallel_ver(vector<vector<Point>>& sup_pxl_lst)
{

	//for (vector<Point> lst : sup_pxl_lst)
	//{
	//	
	//	int x = lst[0].x;
	//	int y = lst[0].y;
	//	vector<double> color_comp = get_color_comp_value_range_anomaly_and_parallel_ver(x, y);
	//	color_comp_map.at<Vec3d>(y, x)[0] = color_comp[0];
	//	color_comp_map.at<Vec3d>(y, x)[1] = color_comp[1];
	//	color_comp_map.at<Vec3d>(y, x)[2] = color_comp[2];
	//}


	std::for_each(
	std::execution::par,
	sup_pxl_lst.begin(),
	sup_pxl_lst.end(),
	[&](vector<Point> lst)
	{   
		int x = lst[0].x;
		int y = lst[0].y;
		vector<double> color_comp = get_color_comp_value_range_anomaly_and_parallel_ver(x, y);
		color_comp_map.at<Vec3d>(y, x)[0] = color_comp[0];
		color_comp_map.at<Vec3d>(y, x)[1] = color_comp[1];
		color_comp_map.at<Vec3d>(y, x)[2] = color_comp[2];
	});


	for (vector<Point> lst : sup_pxl_lst)
	{
		int x = lst[0].x;
		int y = lst[0].y;
		double aa = color_comp_map.at<Vec3d>(y, x)[0];
		double bb = color_comp_map.at<Vec3d>(y, x)[1];
		double cc = color_comp_map.at<Vec3d>(y, x)[2];
		for (Point pt : lst)
		{
			int x = pt.x;
			int y = pt.y;
			color_comp_map.at<Vec3d>(y, x)[0] = aa;
			color_comp_map.at<Vec3d>(y, x)[1] = bb;
			color_comp_map.at<Vec3d>(y, x)[2] = cc;
		}
	}
}

vector<double> get_color_comp_value_range_anomaly_and_parallel_ver(int x, int y)
{
	vector<double> color_comp = { 0.0, 0.0, 0.0 };
	double weight_sum = 0.0;
	double a, b;
	int low_bound = range_map.at<Vec2w>(y, x)[0];
	int high_bound = range_map.at<Vec2w>(y, x)[1];
	int range_num = high_bound - low_bound + 1;
	int total_anomaly_cnt = 0;
	int max_anomaly_cnt = discarded_seam_pixel_lst.size();


	for (int i = 0; i < range_num; i++)
	{
		//cout << i + low_bound<<endl;
		int x_seam = sorted_seam_pixel_lst[i + low_bound].x;
		int y_seam = sorted_seam_pixel_lst[i + low_bound].y;
		if (anomaly_mask.at<uchar>(y_seam, x_seam) == 255) total_anomaly_cnt++;
	}

	double anomaly_ratio = (double)total_anomaly_cnt / (double)range_num;

	//std::vector<int> v(range_num);
	//std::iota(v.begin(), v.end(), 0);
	//std::for_each(
	//	std::execution::par,
	//	v.begin(),
	//	v.end(),
	//	[&](int i)
	//{   
	//	int x_seam = sorted_seam_pixel_lst[i + low_bound].x;
	//	int y_seam = sorted_seam_pixel_lst[i + low_bound].y;
	//	double weight;
	//	double dist_diff = pow((x - x_seam), 2) + pow((y - y_seam), 2);
	//	double color_diff = (pow((double(warped_tar_img.at<Vec3b>(y, x)[0]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[0])), 2)
	//		+ pow((double(warped_tar_img.at<Vec3b>(y, x)[1]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[1])), 2)
	//		+ pow((double(warped_tar_img.at<Vec3b>(y, x)[2]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[2])), 2)) / 65025;
	//	//double new_color_sigma = sigma_color * anomaly_ratio;
	//	double new_color_sigma = max(sigma_color * anomaly_ratio, min_sigma_color);
	//	//double new_color_sigma = anomaly_ratio * 5 + sigma_color * (1 - anomaly_ratio);
	//	a = color_diff / new_color_sigma / new_color_sigma * -1.0;
	//	b = dist_diff / sigma_dist / sigma_dist / width / width * -1.0;
	//	weight = exp(a) * exp(b);
	//	color_comp[0] += color_comp_map.at<Vec3d>(y_seam, x_seam)[0] * weight;
	//	color_comp[1] += color_comp_map.at<Vec3d>(y_seam, x_seam)[1] * weight;
	//	color_comp[2] += color_comp_map.at<Vec3d>(y_seam, x_seam)[2] * weight;
	//	weight_sum += weight;
	//});






	for (int i = 0; i < range_num; i++)
	{
		int x_seam = sorted_seam_pixel_lst[i + low_bound].x;
		int y_seam = sorted_seam_pixel_lst[i + low_bound].y;
		double weight;
		double dist_diff = pow((x - x_seam), 2) + pow((y - y_seam), 2);
		double color_diff = (pow((double(warped_tar_img.at<Vec3b>(y, x)[0]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[0])), 2)
			+ pow((double(warped_tar_img.at<Vec3b>(y, x)[1]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[1])), 2)
			+ pow((double(warped_tar_img.at<Vec3b>(y, x)[2]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[2])), 2)) / 65025;
		//double new_color_sigma = sigma_color * anomaly_ratio;
		double new_color_sigma = max(sigma_color * anomaly_ratio, min_sigma_color);
		//double new_color_sigma = anomaly_ratio * 5 + sigma_color * (1 - anomaly_ratio);
		a = color_diff / new_color_sigma / new_color_sigma * -1.0;
		b = dist_diff / sigma_dist / sigma_dist / width / width * -1.0;
		weight = exp(a) * exp(b);
		color_comp[0] += color_comp_map.at<Vec3d>(y_seam, x_seam)[0] * weight;
		color_comp[1] += color_comp_map.at<Vec3d>(y_seam, x_seam)[1] * weight;
		color_comp[2] += color_comp_map.at<Vec3d>(y_seam, x_seam)[2] * weight;
		weight_sum += weight;
	}


	if (weight_sum == 0.0)
	{
		color_comp[0] = 0.0;
		color_comp[1] = 0.0;
		color_comp[2] = 0.0;
	}
	else
	{
		color_comp[0] = color_comp[0] / weight_sum;
		color_comp[1] = color_comp[1] / weight_sum;
		color_comp[2] = color_comp[2] / weight_sum;
	}

	return color_comp;
}


void update_color_comp_map_range_anomaly_ver(vector<vector<Point>>& sup_pxl_lst)
{
//#pragma omp parallel for
	for (vector<Point> lst : sup_pxl_lst)
	{
		//int nthreads = omp_get_num_threads();
		//cout << nthreads << endl;
		int x = lst[0].x;
		int y = lst[0].y;
		vector<double> color_comp = get_color_comp_value_range_anomaly_ver(x, y);
		color_comp_map.at<Vec3d>(y, x)[0] = color_comp[0];
		color_comp_map.at<Vec3d>(y, x)[1] = color_comp[1];
		color_comp_map.at<Vec3d>(y, x)[2] = color_comp[2];
	}
	for (vector<Point> lst : sup_pxl_lst)
	{
		int x = lst[0].x;
		int y = lst[0].y;
		double aa = color_comp_map.at<Vec3d>(y, x)[0];
		double bb = color_comp_map.at<Vec3d>(y, x)[1];
		double cc = color_comp_map.at<Vec3d>(y, x)[2];
		for (Point pt : lst)
		{
			int x = pt.x;
			int y = pt.y;
			color_comp_map.at<Vec3d>(y, x)[0] = aa;
			color_comp_map.at<Vec3d>(y, x)[1] = bb;
			color_comp_map.at<Vec3d>(y, x)[2] = cc;
		}
	}
}

vector<double> get_color_comp_value_range_anomaly_ver(int x, int y)
{
	vector<double> color_comp = { 0.0, 0.0, 0.0 };
	double weight_sum = 0.0;
	double a, b;
	int low_bound = range_map.at<Vec2w>(y, x)[0];
	int high_bound = range_map.at<Vec2w>(y, x)[1];
	int range_num = high_bound - low_bound + 1;
	int total_anomaly_cnt = 0;
	int max_anomaly_cnt = discarded_seam_pixel_lst.size();


	for (int i = 0; i < range_num; i++)
	{
		//cout << i + low_bound<<endl;
		int x_seam = sorted_seam_pixel_lst[i + low_bound].x;
		int y_seam = sorted_seam_pixel_lst[i + low_bound].y;
		if (anomaly_mask.at<uchar>(y_seam, x_seam) == 255) total_anomaly_cnt++;
	}
	//if ((double)total_anomaly_cnt / (double)range_num > 0.5)
	//{
	//	cout << "Total: " << range_num << "      anomaly: " << total_anomaly_cnt << "     ration: " << \
	//		(double)total_anomaly_cnt / (double)range_num << endl;
	//}
	
	//test_mask.at<uchar>(y, x) = (int)((double)total_anomaly_cnt / (double)range_num * 100);
	//double anomaly_ratio = (double)total_anomaly_cnt / (double)range_num - (double)discarded_seam_pixel_lst.size() / (double)range_num;
	double anomaly_ratio = (double)total_anomaly_cnt / (double)range_num;
	for (int i = 0; i < range_num; i++)
	{
		int x_seam = sorted_seam_pixel_lst[i + low_bound].x;
		int y_seam = sorted_seam_pixel_lst[i + low_bound].y;

		double weight;
		double dist_diff = pow((x - x_seam), 2) + pow((y - y_seam), 2);
		double color_diff = (pow((double(warped_tar_img.at<Vec3b>(y, x)[0]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[0])), 2)
			+ pow((double(warped_tar_img.at<Vec3b>(y, x)[1]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[1])), 2)
			+ pow((double(warped_tar_img.at<Vec3b>(y, x)[2]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[2])), 2)) / 65025;
		//double new_color_sigma = sigma_color * anomaly_ratio;
		double new_color_sigma = max( sigma_color * anomaly_ratio, min_sigma_color);

		//double new_color_sigma = anomaly_ratio * 5 + sigma_color * (1 - anomaly_ratio);
		a = color_diff / new_color_sigma / new_color_sigma * -1.0;
		b = dist_diff / sigma_dist / sigma_dist / width / width * -1.0;
		weight = exp(a) * exp(b);

		color_comp[0] += color_comp_map.at<Vec3d>(y_seam, x_seam)[0] * weight;
		color_comp[1] += color_comp_map.at<Vec3d>(y_seam, x_seam)[1] * weight;
		color_comp[2] += color_comp_map.at<Vec3d>(y_seam, x_seam)[2] * weight;

		weight_sum += weight;
	}
	if (weight_sum == 0.0)
	{
		color_comp[0] = 0.0;
		color_comp[1] = 0.0;
		color_comp[2] = 0.0;
	}
	else
	{
		color_comp[0] = color_comp[0] / weight_sum;
		color_comp[1] = color_comp[1] / weight_sum;
		color_comp[2] = color_comp[2] / weight_sum;
	}

	return color_comp;
}



void update_color_comp_map_range_color_diff_only(vector<vector<Point>>& sup_pxl_lst)
{
	for (vector<Point> lst : sup_pxl_lst)
	{
		int x = lst[0].x;
		int y = lst[0].y;
		vector<double> color_comp = get_color_comp_value_range_color_diff_only(x, y);
		color_comp_map.at<Vec3d>(y, x)[0] = color_comp[0];
		color_comp_map.at<Vec3d>(y, x)[1] = color_comp[1];
		color_comp_map.at<Vec3d>(y, x)[2] = color_comp[2];
	}
	for (vector<Point> lst : sup_pxl_lst)
	{
		int x = lst[0].x;
		int y = lst[0].y;
		double aa = color_comp_map.at<Vec3d>(y, x)[0];
		double bb = color_comp_map.at<Vec3d>(y, x)[1];
		double cc = color_comp_map.at<Vec3d>(y, x)[2];
		for (Point pt : lst)
		{
			int x = pt.x;
			int y = pt.y;
			color_comp_map.at<Vec3d>(y, x)[0] = aa;
			color_comp_map.at<Vec3d>(y, x)[1] = bb;
			color_comp_map.at<Vec3d>(y, x)[2] = cc;
		}
	}
}

vector<double> get_color_comp_value_range_color_diff_only(int x, int y)
{
	vector<double> color_comp = { 0.0, 0.0, 0.0 };
	double weight_sum = 0.0;
	double a, b;
	int low_bound = range_map.at<Vec2w>(y, x)[0];
	int high_bound = range_map.at<Vec2w>(y, x)[1];
	int range_num = high_bound - low_bound + 1;

	for (int i = 0; i < range_num; i++)
	{
		int x_seam = sorted_seam_pixel_lst[i + low_bound].x;
		int y_seam = sorted_seam_pixel_lst[i + low_bound].y;

		double weight;
		//double dist_diff = pow((x - x_seam), 2) + pow((y - y_seam), 2);
		double color_diff = (pow((double(warped_tar_img.at<Vec3b>(y, x)[0]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[0])), 2)
			+ pow((double(warped_tar_img.at<Vec3b>(y, x)[1]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[1])), 2)
			+ pow((double(warped_tar_img.at<Vec3b>(y, x)[2]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[2])), 2)) / 65025;

		a = color_diff / sigma_color / sigma_color * -1.0;
		//b = dist_diff / sigma_dist / sigma_dist / width / width * -1.0;
		//weight = exp(a) * exp(b);
		weight = exp(a);


		color_comp[0] += color_comp_map.at<Vec3d>(y_seam, x_seam)[0] * weight;
		color_comp[1] += color_comp_map.at<Vec3d>(y_seam, x_seam)[1] * weight;
		color_comp[2] += color_comp_map.at<Vec3d>(y_seam, x_seam)[2] * weight;

		weight_sum += weight;
	}
	if (weight_sum == 0.0)
	{
		color_comp[0] = 0.0;
		color_comp[1] = 0.0;
		color_comp[2] = 0.0;
	}
	else
	{
		color_comp[0] = color_comp[0] / weight_sum;
		color_comp[1] = color_comp[1] / weight_sum;
		color_comp[2] = color_comp[2] / weight_sum;
	}

	return color_comp;
}




void update_color_comp_map_avg(vector<vector<Point>>& sup_pxl_lst)
{
	for (vector<Point> lst : sup_pxl_lst)
	{
		int x = lst[0].x;
		int y = lst[0].y;
		int sup_px_size = lst.size();
		int rr = 0;
		int gg = 0;
		int bb = 0;
		for (Point p : lst)
		{
			bb += warped_tar_img.at<Vec3b>(p)[0];
			gg += warped_tar_img.at<Vec3b>(p)[1];
			rr += warped_tar_img.at<Vec3b>(p)[2];
		}
		bb /= sup_px_size;
		gg /= sup_px_size;
		rr /= sup_px_size;
		vector<double> color_comp = get_color_comp_value_avg(x, y, rr, gg, bb);
		color_comp_map.at<Vec3d>(y, x)[0] = color_comp[0];
		color_comp_map.at<Vec3d>(y, x)[1] = color_comp[1];
		color_comp_map.at<Vec3d>(y, x)[2] = color_comp[2];
	}
	for (vector<Point> lst : sup_pxl_lst)
	{
		int x = lst[0].x;
		int y = lst[0].y;
		double aa = color_comp_map.at<Vec3d>(y, x)[0];
		double bb = color_comp_map.at<Vec3d>(y, x)[1];
		double cc = color_comp_map.at<Vec3d>(y, x)[2];
		for (Point pt : lst)
		{
			int x = pt.x;
			int y = pt.y;
			color_comp_map.at<Vec3d>(y, x)[0] = aa;
			color_comp_map.at<Vec3d>(y, x)[1] = bb;
			color_comp_map.at<Vec3d>(y, x)[2] = cc;
		}
	}
}

vector<double> get_color_comp_value_avg(int x, int y, int rr, int gg, int bb)
{
	vector<double> color_comp = { 0.0, 0.0, 0.0 };
	double weight_sum = 0.0;
	double a, b;

	for (int i = 0; i < refined_seam_pixel_lst.size(); i++)
	{
		int x_seam = refined_seam_pixel_lst[i].x;
		int y_seam = refined_seam_pixel_lst[i].y;

		double weight;
		double dist_diff = pow((x - x_seam), 2) + pow((y - y_seam), 2);
		double color_diff = (pow((double(bb) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[0])), 2)
			               + pow((double(gg) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[1])), 2)
			               + pow((double(rr) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[2])), 2)) / 65025;

		a = color_diff / sigma_color / sigma_color * -1.0;
		b = dist_diff / sigma_dist / sigma_dist / width / width * -1.0;
		weight = exp(a) * exp(b);

		color_comp[0] += color_comp_map.at<Vec3d>(y_seam, x_seam)[0] * weight;
		color_comp[1] += color_comp_map.at<Vec3d>(y_seam, x_seam)[1] * weight;
		color_comp[2] += color_comp_map.at<Vec3d>(y_seam, x_seam)[2] * weight;

		weight_sum += weight;
	}
	if (weight_sum == 0.0)
	{
		color_comp[0] = 0.0;
		color_comp[1] = 0.0;
		color_comp[2] = 0.0;
	}
	else
	{
		color_comp[0] = color_comp[0] / weight_sum;
		color_comp[1] = color_comp[1] / weight_sum;
		color_comp[2] = color_comp[2] / weight_sum;
	}

	return color_comp;
}

void build_final_result()
{
	warped_tar_img.convertTo(warped_tar_img, CV_64FC3);
	warped_tar_img = warped_tar_img + color_comp_map;
	warped_tar_img.convertTo(warped_tar_img, CV_8UC3);
	color_comp_map.convertTo(color_comp_map, CV_8UC3);
}

vector<double> get_color_comp_value(int x, int y)
{
	vector<double> color_comp = { 0.0, 0.0, 0.0 };
	double weight_sum = 0.0;
	double a, b;
	
	for (int i = 0; i < refined_seam_pixel_lst.size(); i++)
	{	
		int x_seam = refined_seam_pixel_lst[i].x;
		int y_seam = refined_seam_pixel_lst[i].y;

		double weight;
		double dist_diff = pow((x - x_seam), 2) + pow((y - y_seam), 2);
		double color_diff = (pow((double(warped_tar_img.at<Vec3b>(y, x)[0]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[0])), 2) 
						  + pow( (double(warped_tar_img.at<Vec3b>(y, x)[1]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[1])), 2)
			              + pow( (double(warped_tar_img.at<Vec3b>(y, x)[2]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[2])), 2)) / 65025;

		a = color_diff / sigma_color / sigma_color * -1.0;
		b = dist_diff  / sigma_dist / sigma_dist / width / width * -1.0;
		weight = exp(a) * exp(b);

		color_comp[0] += color_comp_map.at<Vec3d>(y_seam, x_seam)[0] * weight;
		color_comp[1] += color_comp_map.at<Vec3d>(y_seam, x_seam)[1] * weight;
		color_comp[2] += color_comp_map.at<Vec3d>(y_seam, x_seam)[2] * weight;

		weight_sum += weight;
	}
	if (weight_sum == 0.0)
	{
		color_comp[0] = 0.0;
		color_comp[1] = 0.0;
		color_comp[2] = 0.0;
	}
	else
	{
		color_comp[0] = color_comp[0] / weight_sum;
		color_comp[1] = color_comp[1] / weight_sum;
		color_comp[2] = color_comp[2] / weight_sum;
	}

	return color_comp;
}







//vector<vector<Point>> build_superpixel_lst()
//{	
//
//	//clock_t start, end;
//	//double cpu_time_used;
//	//start = clock();
//
//
//	Mat  mask, lab_tar_img, blured_lab_tar_img;
//	cvtColor(warped_tar_img, lab_tar_img, COLOR_BGR2Lab);
//	Ptr<cv::ximgproc::SuperpixelSLIC> slic = cv::ximgproc::createSuperpixelSLIC(lab_tar_img, cv::ximgproc::SLIC, sup_pixel_size, ruler);
//
//	slic->iterate();
//	//cout << "hey guys, the segementation is finished!" << endl;
//	//end = clock();
//	//cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//	//printf("Time = %f\n", cpu_time_used);
//
//
//	Mat labels;
//	int num_sup_pxl = slic->getNumberOfSuperpixels();
//	vector<vector<Point>> sup_pxl_lst(num_sup_pxl);
//	vector<bool> sup_pxl_in_tar_img(num_sup_pxl, false);
//	slic->getLabels(labels);
//
//	//For dubug purpose
//	//Mat contour_mask, temp;
//	//slic->getLabelContourMask(contour_mask);
//	//warped_tar_img.copyTo(temp);
//	//temp.setTo(Scalar(0, 0, 255), contour_mask);
//	//imwrite("slic_contour.png", temp);
//
//	for (Point pt : target_pixel_lst)
//	{	
//		if (result_from_tar_mask.at<uchar>(pt))
//		{
//			int label = labels.at<int>(pt);
//			sup_pxl_in_tar_img[label] = true;
//			sup_pxl_lst[label].push_back(pt);
//		}	
//	}
//	vector<vector<Point>> refined_sup_pxl_lst;
//	for (vector<Point> a : sup_pxl_lst)
//	{
//		if (!a.empty())
//		{
//			refined_sup_pxl_lst.push_back(a);
//		}
//	}
//	//cout << "hey guys, building list is finished!" << endl;
//	//end = clock();
//	//cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//	//printf("Time = %f\n", cpu_time_used);
//	return refined_sup_pxl_lst;
//}


void refine_seam_pixel_lst(vector<Point> seam_pixel_lst, vector<Point>& refined_seam_pixel_lst, vector<Point>& discarded_seam_pixel_lst)
{
	int count = seam_pixel_lst.size();
	Mat points(count, 1, CV_32F);
	Mat label, centers, out;
	int anomaly_count = 0;
	int ordinary_count = 0;

	for (int i=0; i<count; i++)
	{
		Point p = seam_pixel_lst[i];
		float distance = sqrt(
			              (pow((float(warped_ref_img.at<Vec3b>(p)[0]) - float(warped_tar_img.at<Vec3b>(p)[0])), 2)
						 + pow((float(warped_ref_img.at<Vec3b>(p)[1]) - float(warped_tar_img.at<Vec3b>(p)[1])), 2)
						 + pow((float(warped_ref_img.at<Vec3b>(p)[2]) - float(warped_tar_img.at<Vec3b>(p)[2])), 2)));
		//float distance = 
		//	pow((float(warped_ref_img.at<Vec3b>(p)[0]) - float(warped_tar_img.at<Vec3b>(p)[0])), 2)
		//		+ pow((float(warped_ref_img.at<Vec3b>(p)[1]) - float(warped_tar_img.at<Vec3b>(p)[1])), 2)
		//		+ pow((float(warped_ref_img.at<Vec3b>(p)[2]) - float(warped_tar_img.at<Vec3b>(p)[2])), 2);
		points.at<float>(i) = distance;
	}
	kmeans(points, 2, label, TermCriteria(TermCriteria::Type::COUNT, 50, 0.01), 5, KMEANS_PP_CENTERS, out);
	int target = (out.at<float>(0) <= out.at<float>(1)) ? 0 : 1;
	for (int i = 0; i < count; i++)
	{
		if (label.at<int>(i) == target)
		{
			refined_seam_pixel_lst.push_back(seam_pixel_lst[i]);
		}
		else
		{
			discarded_seam_pixel_lst.push_back(seam_pixel_lst[i]);
			anomaly_count++;
		}
	}

	ordinary_count = count - anomaly_count;
	double cost = double(ordinary_count * anomaly_count) * pow(out.at<float>(0) - out.at<float>(1), 2) / double(count * count);
	//cout << "The cost is " << cost << endl;

	if (cost < cost_threshold)
	{
		refined_seam_pixel_lst = seam_pixel_lst;
		discarded_seam_pixel_lst.clear();
	}
}




void refine_seam_pixel_lst_based_abs_RGB_diff(vector<Point> seam_pixel_lst, vector<Point>& refined_seam_pixel_lst, vector<Point>& discarded_seam_pixel_lst)
{
	int count = seam_pixel_lst.size();
	Mat points1(count, 1, CV_32F);
	Mat points2(count, 1, CV_32F);
	Mat points3(count, 1, CV_32F);
	Mat label1, label2, label3, centers, out1, out2, out3;
	int anomaly_count = 0;
	int ordinary_count = 0;

	for (int i = 0; i < count; i++)
	{
		Point p = seam_pixel_lst[i];
		
		int distance1 = (int)warped_ref_img.at<Vec3b>(p)[0] - (int)warped_tar_img.at<Vec3b>(p)[0];
		distance1 = abs(distance1);
		int distance2 = (int)warped_ref_img.at<Vec3b>(p)[1] - (int)warped_tar_img.at<Vec3b>(p)[1];
		distance2 = abs(distance2);
		int distance3 = (int)warped_ref_img.at<Vec3b>(p)[2] - (int)warped_tar_img.at<Vec3b>(p)[2];
		distance3 = abs(distance3);


		points1.at<float>(i, 0) = (float)distance1;
		points2.at<float>(i, 0) = (float)distance2;
		points3.at<float>(i, 0) = (float)distance3;
	}
	kmeans(points1, 2, label1, TermCriteria(TermCriteria::Type::COUNT, 50, 0.01), 5, KMEANS_PP_CENTERS, out1);
	kmeans(points2, 2, label2, TermCriteria(TermCriteria::Type::COUNT, 50, 0.01), 5, KMEANS_PP_CENTERS, out2);
	kmeans(points3, 2, label3, TermCriteria(TermCriteria::Type::COUNT, 50, 0.01), 5, KMEANS_PP_CENTERS, out3);
	int sum1 = sum(label1)[0];
	int sum2 = sum(label2)[0];
	int sum3 = sum(label3)[0];
	int target1 = sum1 < count / 2 ? 0 : 1;
	int target2 = sum2 < count / 2 ? 0 : 1;
	int target3 = sum3 < count / 2 ? 0 : 1;

	
	for (int i = 0; i < count; i++)
	{
		if (label1.at<int>(i) == target1 && label2.at<int>(i) == target2 && label3.at<int>(i) == target3)
		{
			refined_seam_pixel_lst.push_back(seam_pixel_lst[i]);
		}
		else
		{
			anomaly_count++;
			discarded_seam_pixel_lst.push_back(seam_pixel_lst[i]);
		}
	}

	ordinary_count = count - anomaly_count;
	double cost1 = double(ordinary_count * anomaly_count) * pow(out1.at<float>(0) - out1.at<float>(1), 2) / double(count * count);
	double cost2 = double(ordinary_count * anomaly_count) * pow(out2.at<float>(0) - out2.at<float>(1), 2) / double(count * count);
	double cost3 = double(ordinary_count * anomaly_count) * pow(out3.at<float>(0) - out3.at<float>(1), 2) / double(count * count);
	//cout << "The cost1 is " << cost1 << endl;
	//cout << "The cost2 is " << cost2 << endl;
	//cout << "The cost3 is " << cost3 << endl;

	if (cost1 < cost_threshold && cost2 < cost_threshold && cost3 < cost_threshold)
	{
		refined_seam_pixel_lst = seam_pixel_lst;
	}
}

vector<Point> find_next_wavefront(vector<Point> current_wavefront)
{
	static uint8_t gray = 0;
	static int x_offset[] = { -1, -1, -1,  0,  0,  1, 1, 1 };
	static int y_offset[] = { -1,  0,  1, -1,  1, -1, 0, 1 };
	vector<Point> next_wavefront;
	next_wavefront.reserve(10000);
	for (Point point : current_wavefront) discovered_map.at<uchar>(point) = 255;
	for (Point point : current_wavefront)
	{
		for (int i = 0; i < 8; i++)
		{
			int y_n = point.y + y_offset[i];
			int x_n = point.x + x_offset[i];
			if (x_n == -1 || y_n == -1 || x_n == width || y_n == height) continue;

			bool is_not_discovered = discovered_map.at<uchar>(y_n, x_n) == 0 && result_from_tar_mask.at<uchar>(y_n, x_n) == 255;

			if (is_not_discovered)
			{
				next_wavefront.push_back(Point(x_n, y_n));
				discovered_map.at<uchar>(y_n, x_n) = 127;
				test_wavefront_marching.at<uchar>(y_n, x_n) = gray;
			}
		}
	}
	gray += 10;
	return next_wavefront;

}












void build_range_map_with_side_addition(vector<Point> sorted_seam_pixel_lst)
{
	for (int idx = 0; idx < sorted_seam_pixel_lst.size(); idx++)
	{
		range_map.at<Vec2w>(sorted_seam_pixel_lst[idx])[0] = idx;
		range_map.at<Vec2w>(sorted_seam_pixel_lst[idx])[1] = idx;
	}
	int max_cite_range = sorted_seam_pixel_lst.size() - 1;
	int time_stamp = 1;
	static uint8_t gray = 255;
	static int x_offset[] = { -1, -1, -1,  0,  0,  1, 1, 1 };
	static int y_offset[] = { -1,  0,  1, -1,  1, -1, 0, 1 };
	bool flag_early_termination = false;
	vector<Point> next_wavefront;
	next_wavefront.reserve(10000);
	vector<Point> current_wavefront = sorted_seam_pixel_lst;
	for (Point point : sorted_seam_pixel_lst) discovered_map.at<uchar>(point) = 255;
	for (Point point : sorted_seam_pixel_lst) discovered_time_stamp_map.at<ushort>(point) = time_stamp;
	//Mat chosen_one = imread("result.png");
	//Mat chosen_one_2 = imread("test_range.png");
	//for (Point point : sorted_seam_pixel_lst) chosen_one.at<Vec3b>(point) = 255;

	clock_t start, end;
	double cpu_time_used;


	// march through whole target image.
	while (true)
	{




		//start = clock();



		// find next wavefront
		time_stamp++;
		next_wavefront.clear();
		for (Point point : current_wavefront)
		{
			for (int i = 0; i < 8; i++)
			{
				int y_n = point.y + y_offset[i];
				int x_n = point.x + x_offset[i];
				if (x_n == -1 || y_n == -1 || x_n == width || y_n == height) continue;
				bool is_not_discovered = discovered_map.at<uchar>(y_n, x_n) == 0 && result_from_tar_mask.at<uchar>(y_n, x_n) == 255;
				if (is_not_discovered)
				{
					next_wavefront.push_back(Point(x_n, y_n));
					discovered_map.at<uchar>(y_n, x_n) = 255;
					discovered_time_stamp_map.at<ushort>(y_n, x_n) = time_stamp;
					//test_wavefront_marching.at<uchar>(y_n, x_n) = gray;
				}
			}
		}




		//end = clock();
		//cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
		//printf("find next Time = %f\n", cpu_time_used);
		//start = clock();
		
		
		
		
		
		
		
		// break from the while loop if there is no next wavefront.
		if (next_wavefront.size() == 0) break;
		int reference_all_range_cnt = 0;
		// propagate the citation range from previous wavefront to current wavefront.
		for (Point point : next_wavefront)
		{
			int min_range = std::numeric_limits<int>::max();
			int max_range = std::numeric_limits<int>::min();
			
			if (!flag_early_termination)
			{
				for (int i = 0; i < 8; i++)
				{
					int y_n = point.y + y_offset[i];
					int x_n = point.x + x_offset[i];
					if (x_n == -1 || y_n == -1 || x_n >= width || y_n >= height) continue;
					if (discovered_time_stamp_map.at<ushort>(y_n, x_n) == time_stamp - 1)
					{
						if (min_range > range_map.at<Vec2w>(y_n, x_n)[0]) min_range = range_map.at<Vec2w>(y_n, x_n)[0];
						if (max_range < range_map.at<Vec2w>(y_n, x_n)[1]) max_range = range_map.at<Vec2w>(y_n, x_n)[1];
					}
				}

				range_map.at<Vec2w>(point)[0] = max(min_range - propagation_coefficient, 0);
				range_map.at<Vec2w>(point)[1] = min(max_range + propagation_coefficient, max_cite_range);
				if (range_map.at<Vec2w>(point)[0] == 0 && range_map.at<Vec2w>(point)[1] == max_cite_range) reference_all_range_cnt++;
			}
			else
			{
				range_map.at<Vec2w>(point)[0] = 0;
				range_map.at<Vec2w>(point)[1] =max_cite_range;
			}
		}
		if (next_wavefront.size() == reference_all_range_cnt) flag_early_termination = true;
		//cout << reference_all_range_cnt << " asdasdasdasdasd " << next_wavefront.size() << endl;


		//end = clock();
		//cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
		//printf("side propagation Time = %f\n", cpu_time_used);
		//start = clock();
		

		// propagate the citation range from neighbors in current wavefront.
		//Mat tmp_range_map;
		//range_map.copyTo(tmp_range_map);

		//end = clock();
		//cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
		//printf("copy Time = %f\n", cpu_time_used);
		//start = clock();

		//vector<Point> pending_update_pts;
		vector<int> pending_update_min_ranges, pending_update_max_ranges;

		for (Point point : next_wavefront)
		{
			int min_range = std::numeric_limits<int>::max();
			int max_range = std::numeric_limits<int>::min();

			for (int i = 0; i < 8; i++)
			{
				int y_n = point.y + y_offset[i];
				int x_n = point.x + x_offset[i];
				if (x_n == -1 || y_n == -1 || x_n >= width || y_n >= height) continue;
				if (discovered_time_stamp_map.at<ushort>(y_n, x_n) == time_stamp)
				{
					if (min_range > range_map.at<Vec2w>(y_n, x_n)[0]) min_range = range_map.at<Vec2w>(y_n, x_n)[0];
					if (max_range < range_map.at<Vec2w>(y_n, x_n)[1]) max_range = range_map.at<Vec2w>(y_n, x_n)[1];
				}
			}
			//pending_update_pts.push_back(point)
			//range_map.at<Vec2w>(point)[0] = min_range;
			//range_map.at<Vec2w>(point)[1] = max_range;
			pending_update_min_ranges.push_back(min_range);
			pending_update_max_ranges.push_back(max_range);
		}
		
		for (int idx = 0; idx < next_wavefront.size(); idx++)
		{
			range_map.at<Vec2w>(next_wavefront[idx])[0] = pending_update_min_ranges[idx];
			range_map.at<Vec2w>(next_wavefront[idx])[1] = pending_update_max_ranges[idx];
		}









		//// propagate the citation range from neighbors in current wavefront.
		//Mat tmp_range_map;
		//range_map.copyTo(tmp_range_map);

		//end = clock();
		//cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
		//printf("copy Time = %f\n", cpu_time_used);
		//start = clock();

		////vector<Point> pending_update_pts;
		//vector<int> pending_update_min_ranges, pending_update_max_ranges;

		//for (Point point : next_wavefront)
		//{
		//	int min_range = std::numeric_limits<int>::max();
		//	int max_range = std::numeric_limits<int>::min();

		//	for (int i = 0; i < 8; i++)
		//	{
		//		int y_n = point.y + y_offset[i];
		//		int x_n = point.x + x_offset[i];
		//		if (x_n == -1 || y_n == -1 || x_n >= width || y_n >= height) continue;
		//		if (discovered_time_stamp_map.at<ushort>(y_n, x_n) == time_stamp)
		//		{
		//			if (min_range > tmp_range_map.at<Vec2w>(y_n, x_n)[0]) min_range = tmp_range_map.at<Vec2w>(y_n, x_n)[0];
		//			if (max_range < tmp_range_map.at<Vec2w>(y_n, x_n)[1]) max_range = tmp_range_map.at<Vec2w>(y_n, x_n)[1];
		//		}
		//	}
		//	//pending_update_pts.push_back(point)
		//	//range_map.at<Vec2w>(point)[0] = min_range;
		//	//range_map.at<Vec2w>(point)[1] = max_range;
		//	pending_update_min_ranges.push_back(min_range);
		//	pending_update_max_ranges.push_back(max_range);
		//}

		//for (int idx = 0; idx < next_wavefront.size(); idx++)
		//{
		//	range_map.at<Vec2w>(next_wavefront[idx])[0] = pending_update_min_ranges[idx];
		//	range_map.at<Vec2w>(next_wavefront[idx])[1] = pending_update_max_ranges[idx];
		//}









		//end = clock();
		//cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
		//printf("find range Time = %f\n", cpu_time_used);
		


		//cv::Rect myROI(500, 500, 400, 200);
		////cv::Rect myROI(0, 0, 1300, 1200);
		//if (time_stamp < 20 && time_stamp > 10)
		//	//if(0)
		//	//if (time_stamp %500 ==1 )
		//	//	////if (time_stamp % 5 == 1 && time_stamp < 40 && time_stamp > 3)
		//	//	////if (time_stamp % 5 == 1 )
		//{
		//	int count = 0;
		//	for (Point point : next_wavefront)
		//	{
		//		count++;
		//		//if (1)
		//		if (count < 900 && count >700)
		//			//			//if (count % 20 == 0)
		//		{
		//			Mat test_range;
		//			Mat test_range_2;
		//			//			//test_wavefront_marching.copyTo(test_range);
		//			chosen_one.copyTo(test_range);
		//			chosen_one_2.copyTo(test_range_2);
		//			//cout << "the min idx: " << range_map.at<Vec2w>(point)[0] << endl;
		//			//cout << "the max idx: " << range_map.at<Vec2w>(point)[1] << endl << endl;
		//			for (Point p : sorted_seam_pixel_lst) test_range.at<uchar>(p) = 127;
		//			line(test_range, point, sorted_seam_pixel_lst[range_map.at<Vec2w>(point)[0]], Scalar(50, 50, 50));
		//			line(test_range, point, sorted_seam_pixel_lst[range_map.at<Vec2w>(point)[1]], Scalar(50, 50, 50));
		//			//line(test_range, point, sorted_seam_pixel_lst[range_map.at<Vec2w>(point)[1]], Scalar(200, 200, 200));
		//			test_range.at<Vec3b>(point)[2] = 255;
		//			for (Point p : sorted_seam_pixel_lst) test_range_2.at<uchar>(p) = 127;
		//			line(test_range_2, point, sorted_seam_pixel_lst[range_map.at<Vec2w>(point)[0]], Scalar(50, 50, 50));
		//			line(test_range_2, point, sorted_seam_pixel_lst[range_map.at<Vec2w>(point)[1]], Scalar(50, 50, 50));
		//			//line(test_range, point, sorted_seam_pixel_lst[range_map.at<Vec2w>(point)[1]], Scalar(200, 200, 200));
		//			test_range_2.at<Vec3b>(point)[2] = 255;
		//			//stringstream frame_name;
		//			//frame_name << cv::format("range/%.5d-%.5d", time_stamp, count) << ".png";
		//			//imwrite(frame_name.str(), test_range(myROI));
		//			stringstream frame_name_2;
		//			//frame_name_2 << cv::format("range_2/%.5d-%.5d", time_stamp, count) << ".png";
		//			//frame_name_2 << cv::format("mean_range/%.5d-%.5d", time_stamp, count) << ".png";
		//			frame_name_2 << cv::format("median_range_length_previous/%.5d-%.5d", time_stamp, count) << ".png";
		//			//imwrite(frame_name_2.str(), test_range_2(myROI));
		//			imwrite(frame_name_2.str(), test_range(myROI));
		//		}
		//	}
		//}
		gray -= 10;

		current_wavefront = next_wavefront;
	}

}


void sort_seam_pixel_lst(vector<Point> seam_pixel_lst, vector<Point>& sorted_seam_pixel_lst)
{
	Mat seam_discover_map;
	seam_mask.copyTo(seam_discover_map);
	vector<Point> endpoints_lst;
	int x_off[4] = { -1, 1, 0, 0 };
	int y_off[4] = {0, 0, -1, 1};

	for (Point p : seam_pixel_lst)
	{
		int x = p.x;
		int y = p.y;
		int count = 0;

		for (int i = 0; i < 4; i++)
		{
			int nx = x + x_off[i];
			int ny = y + y_off[i];
			
			if (seam_mask.at<uchar>(ny, nx) == 255) count++;
		}
		if (count == 1)
		{
			endpoints_lst.push_back(p);
		}
	}



	stack<Point> discover_stack;
	discover_stack.push(endpoints_lst[0]);
	seam_discover_map.at<uchar>(endpoints_lst[0]) = 0;
	sorted_seam_pixel_lst.push_back(endpoints_lst[0]);

	while (!discover_stack.empty())
	{
		Point p = discover_stack.top();
		discover_stack.pop();
		int x = p.x;
		int y = p.y;

		for (int i=0; i<4; i++)
		{
			int nx = x + x_off[i];
			int ny = y + y_off[i];

			if (seam_discover_map.at<uchar>(ny, nx) == 255)
			{
				Point new_p = Point(nx, ny);
				seam_discover_map.at<uchar>(new_p) = 0;
				sorted_seam_pixel_lst.push_back(new_p);
				discover_stack.push(new_p);
			}

		}
	}
}




