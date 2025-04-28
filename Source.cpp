#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Header.h"
#include <stdio.h>
#include <conio.h>
#include <iostream> 
#include <vector>
#include <iterator>
#include <string>
#include <math.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <fstream>

//#include <gsl>
#include <algorithm>
#include <chrono>
#include <thrust\pair.h>
#include <thrust\device_vector.h>
#include <thrust\extrema.h>
#include <limits>
#include <cstdlib> // for rand() and srand()
#include <ctime>   // for time()
#include <deque>
#include <utility>


#include <cuda_runtime.h>

#include <random>   //for random
#include <chrono>


using namespace std;
using namespace cv;

int var = 90;


/*start_index
345353
start_index
236669
start_index
93553
start_index
105595
start_index
9372
start_index
10201
start_index
37597
start_index
17904
16454
start_index
10141
27220
start_index
2042
6082
start_index
1032
3366
start_index
1097
138
start_index
256
554
start_index
490
2147
start_index
8
13*/

void visualize_nmap(const cv::Mat& nmap){
	//void visualize_nmap(const cv::Mat & nmap)
	
	// Create a grayscale image to visualize nmap
	cv::Mat nmap_color(nmap.size(), CV_8UC1);
	// Assign values to each pixel in nmap_color
	for (int i = 0; i < nmap.rows; i++) {
		for (int j = 0; j < nmap.cols; j++) {
			nmap_color.at<uchar>(i, j) = nmap.at<int>(i, j) * 255 / 7;
		}
	}

	// Show the nmap_color image
	cv::imshow("Neighborhood Map", nmap_color);
	cv::waitKey(0);	
}

//load the image
//load the depth map
//resize them using size_limit careful with this part, limitations of the computer
std::pair<cv::Mat, cv::Mat> load_image_and_depth_map(const std::string& img_file_name, const std::string& depths_file_name, int size_limit = 1024) {
	cv::Mat img = cv::imread(img_file_name, cv::IMREAD_COLOR);  //se usa img y no float img
	cv::Mat depths = cv::imread(depths_file_name, IMREAD_GRAYSCALE);
	if (img.rows > size_limit || img.cols > size_limit) {
		cv::resize(img, img, cv::Size(size_limit, size_limit));
		cv::resize(depths, depths, cv::Size(size_limit, size_limit));}
	cv::Mat img_float, depths_float;
	img.convertTo(img_float, CV_32FC3, 1.0 / 255.0);
	depths.convertTo(depths_float, CV_32FC1);
	depths_float = depths;
	return std::make_pair(img, depths);
}

//make an escalation of the depth map
//for that it tries to normalize the process
//function
//cv::Mat 
std::tuple<cv::Mat, std::vector<std::vector<float>>> preprocess_monodepth_map(cv::Mat depths, float additive_depths, float multiply_depths) {
	cv::Mat depths_n;   //depths_normaliced
	cv::Mat Depths_float = depths.clone(); //floated depth used for preprocess
	double z_max, z_min;
	cv::minMaxLoc(depths, &z_min, &z_max);
	int H = depths.rows;
	int W = depths.cols;
	std::cout << "preprocessed map: " << z_max << " min: " << z_min << endl;
	std::cout << "preprocessed map: " << H << " W: " << W << endl;
	cv::normalize(depths, depths_n, 0, 1.0, cv::NORM_MINMAX, CV_32FC1);
	cv::Mat depths_processed = (multiply_depths * (1.0 - depths_n)) + additive_depths;
	//used to fill the float versions of the code
	// Normalize the depths matrix and store the result in a vector of vectors
	std::vector<std::vector<float>> normalizedDepths(depths.rows, std::vector<float>(depths.cols));
	// Normalize the depths matrix
	for (int i = 0; i < depths.rows; ++i) {
		for (int j = 0; j < depths.cols; ++j) {
			float value = depths.at<uchar>(i, j);  // Get the value at position (i, j)
			value = (value - z_min) / (z_max - z_min);  // Perform normalization
			normalizedDepths[i][j] = value;  // Update the value in the vector
			// depths = (multiply_depth * (1.0 - depths)) + additive_depth
			normalizedDepths[i][j] = (multiply_depths * (1.0 - normalizedDepths[i][j])) + additive_depths;
			//std::wcout << " normalized: " << normalizedDepths[i][j] << " size:rows " << normalizedDepths.size() << " size cols: " << normalizedDepths[i].size()<<" z_max: "<<z_max<< " z_min "<<z_min << endl;
		}
	}
	// Find the maximum and minimum values in normalizedDepths
	//float minVal = std::numeric_limits<float>::max();
	//float maxVal = std::numeric_limits<float>::lowest();
	
	//required formula
	
	//for (const auto& row : normalizedDepths) {
	//	for (float value : row) {
	//		if (value < minVal) {
	//			minVal = value;
	//		}
	//		if (value > maxVal) {
	//			maxVal = value;
	//		}
	//	}
	//}

	//std::cout << "Preprocessed minVal " << minVal << " maxVal: " << maxVal << endl;

	//return depths_processed;
	return std::make_tuple(depths_processed, normalizedDepths);
}

// Calculate the minimum value
size_t minimum_value(size_t a, size_t b) {
	return a < b ? a : b;
}

//Dar una explicacion de lo que realiza la funcion
//entender cuanto demora cada parte y cuando demora en simplificar, anotarlo en el codigo
//hacer una caja negra por funcion
//poner en un archivo de texto todos los pasos para que se entiendan
//ver si es posible si se puede precalcular
//std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point2f>, std::vector<cv::Point2f>> 
//std::tuple<std::vector<cv::Point2f>&, std::vector<cv::Point2f>&, std::vector<cv::Point2f>&> 
std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point2f>, std::vector<cv::Point2f>>  find_backscatter_estimation_points(cv::Mat img, cv::Mat depths, int num_bins = 10, float fraction = 0.01, int max_vals = 20, float min_depth_percent = 0.0) {
	// Convert min_depth_percent to float
	float min_depth_percent_f = static_cast<float>(min_depth_percent);                      //min_depth_percent = float(min_depth_percent)
	// Get maximum and minimum depth values
	double z_max, z_min;
	cv::minMaxLoc(depths, &z_min, &z_max);                                                  //z_max, z_min = np.max(depths), np.min(depths)
	//std::cout << "z_min: " << z_min << " z_max: " << z_max << endl;
	// Compute minimum depth value to consider
	float min_depth = z_min + (min_depth_percent_f * (z_max - z_min));                      //min_depth = z_min + (min_depth_percent * (z_max - z_min))
	//std::cout << "min_depth: " << min_depth << endl;
	
	//std::vector<float> z_ranges(num_bins + 1);                                              //z_ranges = np.linspace(z_min, z_max, num_bins + 1)
	//float z_range_step = (z_max - z_min) / num_bins;
	//for (int i = 0; i <= num_bins; i++) {
		//z_ranges[i] = z_min + (i * z_range_step);
		//std::cout << z_ranges[i] << endl;
	//}
	//0.0, 1.2, 2.4, 3.599999, 4.8, 6.0, 7.1999999, 8.4, 9.6, 10.7999999, 12.0
	std::vector<double> z_ranges = { 0.0, 1.2, 2.4, 3.599999, 4.8, 6.0, 7.1999999, 8.4, 9.6, 10.7999999, 12.0 };  //z_ranges = np.linspace(z_min, z_max, num_bins + 1)
	//salia mucho mas facil hacerlo de esta manera

	//std::cout << " z_ranges: " << z_ranges.size() << endl;
	//for (int i = 0; i < z_ranges.size(); i++){
	//	std::cout << "z_ranges: " << i << endl;
	//}
	// Split the image into its channels
	// Convert img to floating point type and divide by 255
	
	cv::Mat img_float;                                                                     //img_norms = np.mean(img, axis=2)
	img.convertTo(img_float, CV_32F, 1.0 / 255.0);

	// Compute the mean of each three-channel pixel along the third axis
	cv::Mat img_norms;
	cv::cvtColor(img_float, img_norms, cv::COLOR_BGR2GRAY);
	//std::cout<< "img_norms size: "<<img_norms.cols <<" rows: "<<img_norms.rows <<" channels: "<<img_norms.channels() << std::endl;
	//std::cout << "depths size: " << depths.cols << " rows: " << depths.rows << " channels: " << depths.channels() << std::endl;
	
	// Print the first 10 values of img_norms
	//for (int i = 0; i < 10; i++) {
	//	for (int j = 0; j < 10; j++) {
	//		std::cout << (int)img_norms.at<uchar>(i, j) << " ";
	//	}
	//	std::cout << std::endl;
	//}
	// Print the first 10 values of img
	//for (int i = 0; i < 10; i++) {
	//	for (int j = 0; j < 10; j++) {
	//		std::cout << "(" << (int)img.at<cv::Vec3b>(i, j)[0] << "," << (int)img.at<cv::Vec3b>(i, j)[1] << "," << (int)img.at<cv::Vec3b>(i, j)[2] << ") ";
	//	}
	//	std::cout << std::endl;
	//}

	// Compute points for each color channel
	std::vector<cv::Point2f> points_r;
	std::vector<cv::Point2f> points_g;
	std::vector<cv::Point2f> points_b;
	for (int i = 0; i < z_ranges.size() - 1; i++) {
		double a = z_ranges.at(i);                                                                  //a, b = z_ranges[i], z_ranges[i+1]
		double b = z_ranges.at(i + 1);
		
		cv::Mat mask = cv::Mat::zeros(depths.size(), CV_8UC1); // Inicializar la matriz con 0s      //locs = np.where(np.logical_and(depths > min_depth, np.logical_and(depths >= a, depths <= b)))
		cv::Mat depth_mask = depths > min_depth;
		cv::Mat range_mask = (depths >= a) & (depths <= b);
		cv::bitwise_and(depth_mask, range_mask, mask); // Aplicar la operaci�n bitwise_and para combinar las m�scaras
		cv::Mat nonZeroCoords;
		cv::findNonZero(mask, nonZeroCoords);
		std::vector<int> y_positions, x_positions;
		//std::cout << "x_position 290: " << x_positions.size() << endl;
		for (int y = 0; y < mask.rows; y++) {
			for (int x = 0; x < mask.cols; x++) {
				if (mask.at<uint8_t>(y, x)) {  // Si el valor es true (255)
					y_positions.push_back(y);
					x_positions.push_back(x);
				}
			}
		}
		std::vector<std::vector<int>> locs = { y_positions, x_positions };

		//std::cout << "y_position: " << y_positions.size() << endl;
		//std::cout << "locs: " << locs[0].size() << endl;
		//std::cout << "mask: " << std::endl;
		//for (int row = 0; row < mask.rows; row++) {
		//	for (int col = 0; col < mask.cols; col++) {
		//		std::cout << (int)mask.at<uchar>(row, col) << " ";
		//	}
		//	std::cout << std::endl;
		//}
		// 
		// Assuming locs is already defined as std::vector<cv::Point>
		std::vector<int> norms_in_range;                                                          //norms_in_range, px_in_range, depths_in_range = img_norms[locs], img[locs], depths[locs]
		std::vector<int> depths_in_range;
		std::vector<cv::Vec3b> px_in_range;
		//crear un area de memoria
		for (int j = 0; j < locs[0].size(); j++) {
			int y = locs[0][j];
			int x = locs[1][j];
			//verlo de una dimension
			//conversion lineal
			//en cuda hacerlo explicitamente
			int img_norm = img_norms.at<uchar>(y, x);
			int depth = depths.at<uchar>(y, x);
			//std::wcout << "	std::vector<int> depths_in_range: " << depths_in_range.capacity() << endl;
			//no confundir i con la j
			norms_in_range.push_back(img_norm);
			cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);
			px_in_range.push_back(pixel);
			depths_in_range.push_back(depth);
		}	
		//for (const auto& pixel : px_in_range) {
		//	std::cout << "B: " << static_cast<int>(pixel[0]) << ", ";
		//	std::cout << "G: " << static_cast<int>(pixel[1]) << ", ";
		//	std::cout << "R: " << static_cast<int>(pixel[2]) << std::endl;
		//}
		
		//std::cout << "norms_in_range: " << norms_in_range.size() << endl;

		// Zip the three vectors together into a single vector of tuples     
		std::vector<std::tuple<int, cv::Vec3b, int>> zipped_data;                                     //arr = sorted(zip(norms_in_range, px_in_range, depths_in_range), key=lambda x: x[0])
		for (size_t i = 0; i < norms_in_range.size(); i++) {
			zipped_data.emplace_back(norms_in_range[i], px_in_range[i], depths_in_range[i]);
		}
		std::wcout << "zipped data: " << zipped_data.size() << endl;
		// Sort the zipped data based on the first element of each tuple
		std::sort(zipped_data.begin(), zipped_data.end(),
			[](const std::tuple<int, cv::Vec3b, int>& a, const std::tuple<int, cv::Vec3b, int>& b) {
				return std::get<0>(a) < std::get<0>(b);
			}
		);
		
		// Create the 'arr' vector as a sorted combination of the three vectors
		std::vector<std::tuple<int, cv::Vec3b, int>> arr(zipped_data.begin(), zipped_data.end());

		// Calculate the number of elements to extract
		size_t num_points = minimum_value(std::ceil(fraction * arr.size()), max_vals);                //points = arr[:min(math.ceil(fraction * len(arr)), max_vals)]
		
		//std::cout << "num_points: " << num_points << endl;
		// Extract the desired portion of the arr vector
		std::vector<std::tuple<int, cv::Vec3b, int>> points(arr.begin(), arr.begin() + num_points);
		//std::cout << "points: " << points.size() << endl;
		//for (const auto& point : points) {
		//	int z = std::get<0>(point);
		//	cv::Vec3b p = std::get<1>(point);
		//	int depth = std::get<2>(point);
		//
		//	std::cout << "z: " << z << ", p: (" << static_cast<int>(p[0]) << ", " << static_cast<int>(p[1]) << ", " << static_cast<int>(p[2]) << "), depth: " << depth << std::endl;
		//}
		for (const auto& point : points) {               
			int z = std::get<0>(point);
			cv::Vec3b p = std::get<1>(point);

			points_r.emplace_back(z, p[0]);            //points_r.extend([(z, p[0]) for n, p, z in points])
			points_g.emplace_back(z, p[1]);            //points_g.extend([(z, p[1]) for n, p, z in points])
			points_b.emplace_back(z, p[2]);            //points_b.extend([(z, p[2]) for n, p, z in points])
		}
		//std::wcout << "point_r: " << points_r.size() << endl;
		//for (const auto& point : points_b) {
		//	std::cout << "Point: (" << point.x << ", " << point.y << ")" << std::endl;
		//}
		//std::wcout << "point_r: " << points_r.size() << endl;
	
	}
	//return std::make_tuple(points_r, points_g, points_b);
	return std::tie(points_r, points_g, points_b);
}


//void find_backscatter_values(B_pts, depths, restarts = 10, max_mean_loss_fraction = 0.1)
//def estimate(depths, B_inf, beta_B, J_prime, beta_D_prime) :
//	val = (B_inf * (1 - np.exp(-1 * beta_B * depths))) + (J_prime * np.exp(-1 * beta_D_prime * depths))
//	return val
//def loss(B_inf, beta_B, J_prime, beta_D_prime) :
//	val = np.mean(np.abs(B_vals - estimate(B_depths, B_inf, beta_B, J_prime, beta_D_prime)))
//	return val
void find_backscatter_values(const std::vector<std::pair<double, cv::Point>>& B_pts, cv::Mat depths, int restarts = 10, float max_mean_loss_fraction = 0.1) {
	std::cout << "entro a find backscatter values" << endl;
	// Extract B_vals and B_depths from B_pts
	int num_pts = B_pts.size();                                                                                             //B_vals, B_depths = B_pts[:, 1], B_pts[:, 0]
	std::vector<double> B_vals(num_pts), B_depths(num_pts);  
	for (int i = 0; i < num_pts; i++) {
		B_vals[i] = B_pts[i].first;
		B_depths[i] = B_pts[i].second.y;   //corregir parece que es al revez
	}
	double z_max, z_min;                                                                                                    //z_max, z_min = np.max(depths), np.min(depths)
	cv::minMaxLoc(depths, &z_min, &z_max);
	// Calculate the maximum mean loss
	double max_mean_loss = max_mean_loss_fraction * (z_max - z_min);                                                        //max_mean_loss = max_mean_loss_fraction * (z_max - z_min)
	double best_loss = 1/0.00000000000000000000000000000000001; //REVISAR ESTA PARTE                                        //best_loss = np.inf
	std::vector<double> coefs;                                                                                              //coefs = None

	// Define the estimate function
	auto estimate = [](std::vector<double> depths, double B_inf, double beta_B, double J_prime, double beta_D_prime) {       //def estimate(depths, B_inf, beta_B, J_prime, beta_D_prime):
		std::vector<double> val(depths.size());                                                                              //val = (B_inf * (1 - np.exp(-1 * beta_B * depths))) + (J_prime * np.exp(-1 * beta_D_prime * depths))
		for (int i = 0; i < depths.size(); i++) {
			val[i] = B_inf * (1 - std::exp(-beta_B * depths[i])) + J_prime * std::exp(-beta_D_prime * depths[i]);
		}
		return val;                                                                                                           //return val
	};

	//corregir esta funcion fase de prueba
	// Define the loss function
	auto loss = [&](double B_inf, double beta_B, double J_prime, double beta_D_prime) {                                       //def loss(B_inf, beta_B, J_prime, beta_D_prime):
		double sum = 0.0;                                                                                                     //val = np.mean(np.abs(B_vals - estimate(B_depths, B_inf, beta_B, J_prime, beta_D_prime)))
		for (int i = 0; i < num_pts; i++) {
			double estimate_val = B_inf * (1 - std::exp(-beta_B * B_depths[i])) + J_prime * std::exp(-beta_D_prime * B_depths[i]);
			sum += std::abs(B_vals[i] - estimate_val);
		}
		double val = sum / static_cast<double>(num_pts);
		return val;                                                                                                           //return val
	};
	// Set the bounds for the optimization
	std::vector<double> bounds_lower = { 0, 0, 0, 0 };
	std::vector<double> bounds_upper = { 1, 5, 1, 5 };



	// Define the optimization problem
	// Run the optimization multiple times to avoid local minima
	

	// Define the learning rate and number of iterations for gradient descent
	double learning_rate = 0.001;             //revisar  esta parte del codigo
	int num_iterations = 10000;

	// Initialize the parameters with random values within the bounds
	std::vector<double> p0(4);
	for (int i = 0; i < 4; i++) {
		p0[i] = bounds_lower[i] + static_cast<double>(rand()) / RAND_MAX * (bounds_upper[i] - bounds_lower[i]);
	}

	// Perform gradient descent
	//double best_loss = std::numeric_limits<double>::infinity(); 
	std::vector<double> best_params;

	for (int iter = 0; iter < num_iterations; iter++) {
		// Calculate the gradient of the loss function
		std::vector<double> grad(4);
		for (int i = 0; i < 4; i++) {
			std::vector<double> p(p0);
			p[i] += 1e-6;
			double loss1 = loss(p[0], p[1], p[2], p[3]);
			double loss2 = loss(p0[0], p0[1], p0[2], p0[3]);
			grad[i] = (loss1 - loss2) / 1e-6;
		}
		// Update the parameters
		for (int i = 0; i < 4; i++) {
			p0[i] -= learning_rate * grad[i];
			// Enforce the bounds
			if (p0[i] < bounds_lower[i]) {
				p0[i] = bounds_lower[i];
			}
			if (p0[i] > bounds_upper[i]) {
				p0[i] = bounds_upper[i];
			}
		}
		// Check if the new parameters improve the loss
		double loss_val = loss(p0[0], p0[1], p0[2], p0[3]);
		if (loss_val < best_loss) {
			best_loss = loss_val;
			best_params = p0;              //coef = best_param change this part when you can
		}
	}

	std::cout << "best loss: " << best_loss << endl;

	//std::cout << "B_vals: " << endl;
	//for (int i = 0; i < num_pts; i++){
	//	std::cout << B_vals[i] << " ";
	//}
	auto max = std::max_element(B_vals.begin(), B_vals.end());
	//std::cout << endl;
	std::cout << "max_element: " << *max << endl;

	//std::vector<double> estimate_result = estimate(depths, coefs[0], coefs[1], coefs[2], coefs[3]);
	
}


// Helper function to estimate values
std::vector<double> estimate2(const std::vector<float>& depths, double B_inf, double beta_B, double J_prime, double beta_D_prime) {
	std::vector<double> val;
	for (const auto& depth : depths) {
		double value = (B_inf * (1 - std::exp(-1 * beta_B * depth))) + (J_prime * std::exp(-1 * beta_D_prime * depth));
		val.push_back(value);
	}
	return val;
}

void find_backscatter_value_2(const std::vector<cv::Point2f>& B_pts, const cv::Mat& depths, int restarts = 10, float max_mean_loss_fraction = 0.1) {
	std::vector<float> B_vals;                                                //B_vals, B_depths = B_pts[:, 1], B_pts[:, 0]
	std::vector<float> B_depths;
	for (const auto& point : B_pts) {
		float val = point.y;  // Extract the y-coordinate as the value
		float depth = point.x;  // Extract the x-coordinate as the depth
		B_vals.push_back(val);
		B_depths.push_back(depth);
	}
	double z_min, z_max;                                                      //z_max, z_min = np.max(depths), np.min(depths)
	cv::minMaxLoc(depths, &z_min, &z_max);
	double max_mean_loss = max_mean_loss_fraction * (z_max - z_min);          //max_mean_loss = max_mean_loss_fraction * (z_max - z_min)
	std::vector<float> coefs;                                                 //coefs = None
	// Initialize best_loss to positive infinity
	float best_loss = std::numeric_limits<float>::infinity();                 //best_loss = np.inf
	//std::cout << "best_loss: " << best_loss << endl;
	std::vector<float> bounds_lower = { 0, 0, 0, 0 };                         //bounds_lower = [0,0,0,0]
	std::vector<float> bounds_upper = { 1, 5, 1, 5 };                         //bounds_upper = [1,5,1,5]

	double B_inf;  // Initialize B_inf with the desired value
	double beta_B; // Initialize beta_B with the desired value
	double J_prime; // Initialize J_prime with the desired value
	double beta_D_prime; // Initialize beta_D_prime with the desired value

	auto estimate = [](const std::vector<float>& depths, double B_inf, double beta_B, double J_prime, double beta_D_prime) -> std::vector<double> {   //def estimate(depths, B_inf, beta_B, J_prime, beta_D_prime):
		std::vector<double> val;
		for (const auto& depth : depths) {
			double value = (B_inf * (1 - std::exp(-1 * beta_B * depth))) + (J_prime * std::exp(-1 * beta_D_prime * depth));                            //val = (B_inf * (1 - np.exp(-1 * beta_B * depths))) + (J_prime * np.exp(-1 * beta_D_prime * depths))
			val.push_back(value);
		}
		return val;                                                                                                                                    //return val
	};


	auto loss = [&](double B_inf, double beta_B, double J_prime, double beta_D_prime) -> double {                                                      //def loss(B_inf, beta_B, J_prime, beta_D_prime):
		std::vector<double> estimated_vals = estimate2(B_depths, B_inf, beta_B, J_prime, beta_D_prime);                                                //val = np.mean(np.abs(B_vals - estimate(B_depths, B_inf, beta_B, J_prime, beta_D_prime)))
		double loss_val = 0.0; 
		for (std::size_t i = 0; i < B_vals.size(); ++i) {
			loss_val += std::abs(B_vals[i] - estimated_vals[i]);
		}
		loss_val /= B_vals.size();
		return loss_val;                                                                                                                                //return val
	};

	for (int i = 0; i < restarts; i++){
		try{
			
		}
		catch (const std::exception&){

		}
	}

}


void showPlot(const std::vector<std::tuple<double, double, double>>& data) {
	int numDataPoints = data.size();

	// Create windows for plotting each channel
	cv::namedWindow("Blue Channel", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Green Channel", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Red Channel", cv::WINDOW_AUTOSIZE);

	// Find the maximum value for normalization
	double maxVal = 0.0;
	for (const auto& tuple : data) {
		double blue = std::get<0>(tuple);
		double green = std::get<1>(tuple);
		double red = std::get<2>(tuple);
		maxVal = std::max(maxVal, std::max(blue, std::max(green, red)));
	}

	// Plot the data in each channel window
	for (int i = 0; i < numDataPoints; i++) {
		double blue = std::get<0>(data[i]) / maxVal;
		double green = std::get<1>(data[i]) / maxVal;
		double red = std::get<2>(data[i]) / maxVal;

		int x = i * 2;

		// Plot blue channel
		cv::Mat bluePlot = cv::Mat::zeros(600, 2, CV_8UC1);
		cv::line(bluePlot, cv::Point(1, 599), cv::Point(1, 599 - blue * 599), cv::Scalar(255), 1);
		cv::imshow("Blue Channel", bluePlot);

		// Plot green channel
		cv::Mat greenPlot = cv::Mat::zeros(600, 2, CV_8UC1);
		cv::line(greenPlot, cv::Point(1, 599), cv::Point(1, 599 - green * 599), cv::Scalar(255), 1);
		cv::imshow("Green Channel", greenPlot);

		// Plot red channel
		cv::Mat redPlot = cv::Mat::zeros(600, 2, CV_8UC1);
		cv::line(redPlot, cv::Point(1, 599), cv::Point(1, 599 - red * 599), cv::Scalar(255), 1);
		cv::imshow("Red Channel", redPlot);

		cv::waitKey(30); // Add a small delay between frames
	}

	cv::waitKey(0);
	cv::destroyAllWindows();
}


// Function to convert a float vector to a comma-separated string
std::string floatVectorToCSV(const std::vector<double>& vec) {
	std::stringstream ss;
	for (size_t i = 0; i < vec.size(); ++i) {
		ss << vec[i];
		if (i != vec.size() - 1)
			ss << ",";
	}
	return ss.str();
}

// Function to save four float vectors as a CSV file
void saveFourFloatVectorsToCSV(const std::vector<double>& vector1, const std::vector<double>& vector2,
	const std::vector<double>& vector3, const std::vector<double>& vector4,
	const std::string& filename) {
	std::ofstream file(filename);
	if (file.is_open()) {
		file << floatVectorToCSV(vector1) << "\n";
		file << floatVectorToCSV(vector2) << "\n";
		file << floatVectorToCSV(vector3) << "\n";
		file << floatVectorToCSV(vector4) << "\n";
		file.close();
		std::cout << "Vectors saved to " << filename << "." << std::endl;
	}
	else {
		std::cerr << "Unable to open the file." << std::endl;
	}
}


// Ecuaci�n no lineal f = (b(1) * (1 - exp(1 * b(2) * x)) + (b(3) * exp(-1 * b(4) * x)))
double equation(double x, const std::vector<double>& b) {
	return (b[0] * (1 - exp(1 * b[1] * x)) + (b[2] * exp(-1 * b[3] * x)));
}

// Funci�n de ajuste: f(x) = b1 * (1 - exp(b2 * x)) + (b3 * exp(-b4 * x))
double fitFunction(const std::vector<double>& x, const std::vector<double>& params){
	double b1 = params[0];
	double b2 = params[1];
	double b3 = params[2];
	double b4 = params[3];
	double result = b1 * (1 - exp(b2 * x[0])) + (b3 * exp(-b4 * x[0]));
	return result;
}

// Funci�n de error para el ajuste de m�nimos cuadrados
double errorFunction(const std::vector<double>& params, const std::vector<double>& x, const std::vector<double>& y){
	double error = 0.0;
	size_t dataSize = x.size();
	for (size_t i = 0; i < dataSize; ++i){
		double residual = fitFunction({ x[i] }, params) - y[i];
		error += residual * residual;
		//std::cout <<"error: " << error << " residual: " << residual << " X: "<<x[i] << " y: " << y[i] << endl;
	}
	return error;
}

// Ajuste de curva utilizando m�nimos cuadrados con restricciones en los par�metros
void curveFitting(const std::vector<double>& x, const std::vector<double>& y, std::vector<double>& params){

	// Valores iniciales de los par�metros
	params = { 0.001, 0.001, 0.001, 0.001 };
	// Configuraci�n del ajuste
	const double epsilon = 1e-8; // Tolerancia de convergencia
	const double stepSize = 0.001; // Tama�o de paso para la actualizaci�n de los par�metros
	const size_t maxIterations = 1000; // N�mero m�ximo de iteraciones
	// Ajuste de curva iterativo
	size_t iteration = 0;
	double prevError = errorFunction(params, x, y);
	std::cout << "prevError: " << prevError <<endl;
	while (iteration < maxIterations){
		for (size_t i = 0; i < params.size(); ++i){
			double prevParam = params[i];
			params[i] += stepSize;
			double currError = errorFunction(params, x, y);
			if (currError > prevError){
				// Si el error empeora, revertir la actualizaci�n
				params[i] = prevParam;
			}
			else{
				// Si el error mejora, continuar actualizando
				prevError = currError;
			}
		}
		// Verificar si ha convergido
		if (prevError < epsilon){
			break;
		}
		iteration = iteration + 1;
	}
	//std::cout << "iteration: " << iteration << endl;
	//std::cout << "params: " << params[0] << " , " << params[1] << " , " << params[2] << " , " << params[3] << endl;
	//la ecuacion se repite 1000 veces cambiar el parametro lanzar 100 hilos para que genere 100 comparaciones instantaneas
	//lanzar 1000 hilos por cada canal
	//buscar el resultado mas peque�o
	//1000 datos de 500 procesadores y guardar el minimo y despues de log n pasos se tiene el minimo de memoria
}


//pixel prior
//methodology
void Backscatter_removal(cv::Mat img, cv::Mat depth, std::vector<std::vector<float>> depths_vector) {
	int H = img.rows;
	int W = img.cols;
	std::cout << "H: " << H << "W: " << W << endl;
	cv::Mat Sum = cv::Mat::zeros(H, W, CV_32F);
	std::vector<std::vector<float>> Sum_vector(depth.rows, std::vector<float>(depth.cols));

	std::vector<std::vector<double>> ImgValue_R(H, std::vector<double>(W, 0.0));
	std::vector<std::vector<double>> ImgValue_G(H, std::vector<double>(W, 0.0));
	std::vector<std::vector<double>> ImgValue_B(H, std::vector<double>(W, 0.0));

	std::cout << "Sum rows: " << Sum.rows << " cols: " << Sum.cols <<" channels: "<<Sum.channels() << endl;
	for (int i = 0; i < H; i++){
		for (int j = 0; j < W; j++){
			Vec3b pixel = img.at<Vec3b>(i, j);
			float sum = pixel[0] + pixel[1] + pixel[2];
			double pixel_0 = static_cast<float>(pixel[0]) / 255;
			double pixel_1 = static_cast<float>(pixel[1]) / 255;
			double pixel_2 = static_cast<float>(pixel[2]) / 255;
			Sum.at<float>(i, j) = sum;
			Sum_vector[i][j] = pixel_0 + pixel_1 + pixel_2;

			ImgValue_R[i][j] = static_cast<float>(pixel[2]) / 255;
			ImgValue_G[i][j] = static_cast<float>(pixel[1]) / 255;
			ImgValue_B[i][j] = static_cast<float>(pixel[0]) / 255;
			//double blue = pixel[0] / 255.0;
			//double green = pixel[1] / 255.0;
			//double red = pixel[2] / 255.0;

			//if (j == 1 && i == 0){
			//	std::cout << "pixel[0]: " << pixel_0 << " pixel[1]: " << pixel_1 << " pixel[2]: " << pixel_2 << " sum: " << Sum.at<float>(i, j) << " sum_vector: "<< Sum_vector[i][j] << endl;
			//}
		}
	}

	//ambas formulas para calcular el minimo y el maximo hacen exactamente lo mismo
	// Calculate the scope value
	//double minDepth, maxDepth;
	//minMaxLoc(depth, &minDepth, &maxDepth);
	double minDepth = std::numeric_limits<float>::max();
	double maxDepth = std::numeric_limits<float>::lowest();
	for (const auto& row : depths_vector) {
		for (float value : row) {
			if (value < minDepth) {
				minDepth = value;
			}
			if (value > maxDepth) {
				maxDepth = value;
			}
		}
	}
	double scope = (maxDepth - minDepth) / 10.0;
	double start = minDepth;
	double end = start + scope;
	std::cout << "min: " << minDepth << " max: " << maxDepth << endl;
	std::cout << "scope: " << scope << " start: " << start << " end: " << end << endl;

	std::vector<cv::Vec3b> B;
	std::vector<std::tuple<double, double, double>> B_lol;
	std::vector<double>B_r, B_g, B_b;

	std::vector<double> D;

	for (int i = 1; i <= 10; i++) {
		//std::vector<int> arrayx, arrayy;
		//for (int y = 0; y < H; y++) {
		//	for (int x = 0; x < W; x++) {
		//		double depthValue = depth.at<uchar>(y, x);
		//		if (depthValue>= start && depthValue < end){
		//			arrayx.push_back(x);
		//			arrayy.push_back(y);
					//std::wcout << "x: " << x << " y: " << y << endl;
		//		}
		//	}
		//}

		//int len = arrayx.size();
		std::vector<int> arrayx, arrayy;
		for (int y = 0; y < depths_vector.size(); y++) {
			for (int x = 0; x < depths_vector[0].size(); x++) {
				double depthValue = depths_vector[y][x];
				if (depthValue >= start && depthValue < end) {
					arrayx.push_back(x);
					arrayy.push_back(y);
				}
			}
		}
		int len = arrayx.size();
		//if (i == 1){
		//	for (int k = 0; k < 20; k++){ 
		//		std::cout << " arrayx: " << arrayx[k] << " arrayy: " << arrayy[k] << endl;
		//	}
		//}

		//std::wcout<< "len: "<<len << " arrayx: " << arrayx.size() << " arrayy: " << arrayy.size() << " len2: " << len2 << " arrayx2: " << arrayx2.size() << " arrayy2: " << arrayy2.size() << endl;
		//std:cout<< " len: " << len << " arrayx: " << arrayx.size() << " arrayy: " << arrayy.size() << endl;

		std::vector<int> index_arrayx(len);
		std::vector<int> index_arrayy(len);
		// Sort arrayx and arrayy based on Sum values
		std::vector<std::pair<float, int>> sumIndices;
		for (int j = 0; j < len; j++) {
			int x = arrayx[j];
			int y = arrayy[j];
			float sumValue = Sum.at<float>(y, x);
			float sum_vectorValue = Sum_vector[y][x];
			//std::cout << "sumValue: " << sumValue << "sumVectorValue: "<< sum_vectorValue << " x: " << x << " y: " << y << endl;
			sumIndices.push_back(std::make_pair(sum_vectorValue, j));
		}
		//if (i == 1){
		//	for (int j = 0; j < 10; j++) {
		//		std::cout << "sumindices: " << sumIndices[j].first << "j: " << sumIndices[j].second << endl;
		//	}
		//}
		//std::cout << endl;
		
		std::sort(sumIndices.begin(), sumIndices.end());
		
		//if (i == 1) {
			//for (int j = 0; j < sumIndices.size(); j++) {
				//std::cout << "sumindices: " << sumIndices[j].first << "j: " << sumIndices[j].second << endl;
				//if (sumIndices[j].first < 1.2) {
				//	std::cout << "j: " << j << "sumindices: " << sumIndices[j].first  << endl;
				//}
			//}
		//}

		// Populate the sorted indices
		for (int j = 0; j < len; j++) {
			int sortedIndex = sumIndices[j].second;
			//std::wcout << "index_arrayx: " << sortedIndex <<" sum: "<< sumIndices[j].first << endl;
			index_arrayx[j] = arrayx[sortedIndex];
			index_arrayy[j] = arrayy[sortedIndex];
			//std::cout << "index_arrayx: " << index_arrayx[j] << " index_arrayy: " << index_arrayy[j] << endl;
		}
		//std::wcout << "arrayx: " << arrayx.size() << " index_arrayx: " << index_arrayx.size() << endl;
		//if (i <= 1){
		//	for (int j = 0; j < index_arrayx.size(); j++) {
		//		std::wcout << " arrayx: " << arrayx[j] << " index_arrayx: " << index_arrayx[j] << endl;
		//	}
		//}	
		//	std::vector<int> index_arrayx(len);
		//	std::vector<int> index_arrayy(len);
		// Populate the sorted indices
		//	for (int i = 0; i < len; i++) {
		//		index_arrayx[i] = arrayx[sortedIndices.at<int>(i)];
		//		index_arrayy[i] = arrayy[sortedIndices.at<int>(i)];
		//	}
		// 
		//devuelve el mayor n�mero entero menor o igual a un n�mero de punto flotante dado
		//std::cout << "len: " << len << endl;

		int m = std::floor(len * 0.01); // Calculate the value of m
		if (m > 300){ 
			m = 300;
		}
		
		for (int j = 0; j < m; j++){
			
			int x = index_arrayx[j];
			int y = index_arrayy[j];
			
			cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);
			//double depthValue = depth.at<uchar>(y, x);
			double depthValue = depths_vector[y][x];

			B.push_back(pixel);
			D.push_back(depthValue);

			double blue = pixel[0] / 255.0;
			double green = pixel[1] / 255.0;
			double red = pixel[2] / 255.0;
			B_r.push_back(red);
			B_g.push_back(green);
			B_b.push_back(blue);
			B_lol.push_back(std::make_tuple(blue, green, red));
		}
		start = end;
		end = end + scope;
	}

	//f= @(b,x)(b(1).*(1-exp(1.*b(2).*x)) + (b(3).*exp(-1.*b(4).*x)))
	// Define the initial parameter values
	//investigar descenso gradiente
	std::vector<double> params_r;   //inicializando los valores de los parametros para r
	std::vector<double> params_g;   //inicializando los valores de los parametros para g
	std::vector<double> params_b;   //inicializando los valores de los parametros para b

	std::vector<double> params_CUDA_r;
	std::vector<double> params_CUDA_g;
	std::vector<double> params_CUDA_b;
	//std::vector<double> estimated_parameters;
	//double learning_rate = 0.01;  // Learning rate
	//int num_iterations = 1000;  // Number of iterations
	//nonlinearLeastSquares(D, B_r, params);
	// Save the four float vectors to a CSV file
	//saveFourFloatVectorsToCSV(B_r, B_g, B_b, D, "four_float_vectors.csv");

	

	//auto startTime = std::chrono::high_resolution_clock::now();
	curveFitting(D, B_r, params_r);
	//auto endTime = std::chrono::high_resolution_clock::now();
	//auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
	std::cout << "params: " << params_r[0] << " , " << params_r[1] << " , " << params_r[2] << " , " << params_r[3] << endl;
	// Output the duration
	//std::cout << "curveFitting time: " << duration.count() << " milliseconds." << std::endl;
	
	
	curveFitting(D, B_g, params_g);
	std::cout << "params: " << params_g[0] << " , " << params_g[1] << " , " << params_g[2] << " , " << params_g[3] << endl;
	curveFitting(D, B_b, params_b);
	std::cout << "params: " << params_b[0] << " , " << params_b[1] << " , " << params_b[2] << " , " << params_b[3] << endl;
	
	//auto endTime2 = std::chrono::high_resolution_clock::now();
	//auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(endTime2 - startTime2);
	//std::cout << "curveFitting2 time: " << duration2.count() << " milliseconds." << std::endl;

	std::cout << "depths_vector size: "<< depths_vector.size()<< " depths_vector[0]size: "<< depths_vector[0].size() << endl;

	//auto startTime2 = std::chrono::high_resolution_clock::now();
	//se trata de hacer hacer el lsqcurvefit pero como funcion cuda
	curveFittingCUDA2(D, B_r, params_CUDA_r);
	std::cout << "params_CUDA_r: " << params_CUDA_r[0] << " , " << params_CUDA_r[1] << " , " << params_CUDA_r[2] << " , " << params_CUDA_r[3] << endl;
	
	//curveFittingCUDA(D, B_g, params_CUDA_g);
	//std::cout << "params_CUDA_g: " << params_CUDA_g[0] << " , " << params_CUDA_g[1] << " , " << params_CUDA_g[2] << " , " << params_CUDA_g[3] << endl;
	//curveFittingCUDA(D, B_b, params_CUDA_b);
	//std::cout << "params_CUDA_b: " << params_CUDA_b[0] << " , " << params_CUDA_b[1] << " , " << params_CUDA_b[2] << " , " << params_CUDA_b[3] << endl;


	//vercion optimizada funcional
	std::vector<double> params_CUDA_r_optimized;
	std::vector<double> params_CUDA_g_optimized;
	std::vector<double> params_CUDA_b_optimized;
	curveFittingCUDA3(D, B_r, B_g, B_b, params_CUDA_r_optimized, params_CUDA_g_optimized, params_CUDA_b_optimized);


	std::cout << "H: " << H << " W: " << W << " depths.rows: " << depth.rows << " depths.cols: " << depth.cols << endl;
	//auto endTime2 = std::chrono::high_resolution_clock::now();
	//auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(endTime2 - startTime2);
	//std::cout << "curveFitting2 time: " << duration2.count() << " milliseconds." << std::endl;

	std::vector<std::vector<double>> BValue_Vector_R(H, std::vector<double>(W, 0.0));
	std::vector<std::vector<double>> BValue_Vector_G(H, std::vector<double>(W, 0.0));
	std::vector<std::vector<double>> BValue_Vector_B(H, std::vector<double>(W, 0.0));
	int cont = 0;

	cv::Mat Dc(H, W, CV_8UC3, cv::Scalar(0, 0, 0));

	for (int i = 0; i < depths_vector.size(); i++){
		for (int j = 0; j < depths_vector[0].size(); j++) {
			double BValue_r = equation(depths_vector[i][j],params_r);
			double BValue_g = equation(depths_vector[i][j],params_g);
			double BValue_b = equation(depths_vector[i][j],params_b);
			BValue_Vector_R[i][j] = int(255 * (ImgValue_R[i][j] - BValue_r));    //255 * (ImgValue_R[i][j] - BValue_Vector_R[i][j])
			BValue_Vector_G[i][j] = int(255 * (ImgValue_G[i][j] - BValue_g));
			BValue_Vector_B[i][j] = int(255 * (ImgValue_B[i][j] - BValue_b));
			if (BValue_Vector_R[i][j] <= 0){
				//std::cout << "("<<i<<" , "<<j<<"): BValue_Vector_R[i][j]: " << BValue_Vector_R[i][j] << endl;
				BValue_Vector_R[i][j] = 0;
				//cont = cont + 1;
			}
			if (BValue_Vector_G[i][j] <= 0){
				//std::cout << "(" << i << " , " << j << "): BValue_Vector_R[i][j]: " << BValue_Vector_R[i][j] << " BValue_Vector_G[i][j]: "<< BValue_Vector_G[i][j] << endl;
				BValue_Vector_G[i][j] = 0;
				//cont = cont + 1;
			}
			if (BValue_Vector_B[i][j] <= 0){

				BValue_Vector_B[i][j] = 0;
				//cont = cont + 1;
			}

			Dc.at<cv::Vec3b>(i, j)[0] = static_cast<uchar>(BValue_Vector_B[i][j]);  //B
			Dc.at<cv::Vec3b>(i, j)[1] = static_cast<uchar>(BValue_Vector_G[i][j]);  //G
			Dc.at<cv::Vec3b>(i, j)[2] = static_cast<uchar>(BValue_Vector_R[i][j]);  //R

			//std::cout << "(" << i << "," << j << ")="<< " R: "<< BValue_Vector_R[i][j] <<" G: "<< BValue_Vector_G[i][j] <<" B: "<< BValue_Vector_B[i][j] << endl;
			
			//std::cout << "(" << i << "," << j << ")= R: " << BValue_Vector_R[i][j] << " = " << BValue_r << " G: " << BValue_Vector_G[i][j] << " = " << BValue_g << " B: " << BValue_Vector_B[i][j] <<" = "<< BValue_b << endl;
			//std::cout << "BValue_r: " << BValue_r << " depths_vector: " << depths_vector[i][j] << " B0: " << params_r[0] << " B1: " << params_r[1] << " B2: " << params_r[2] << " B3: " << params_r[3] << endl;
			//std::cout << "BValue_g: " << BValue_g << " depths_vector: " << depths_vector[i][j] << " B0: " << params_g[0] << " B1: " << params_g[1] << " B2: " << params_g[2] << " B3: " << params_g[3] << endl;
			//std::cout << "BValue_b: " << BValue_b << " depths_vector: " << depths_vector[i][j] << " B0: " << params_b[0] << " B1: " << params_b[1] << " B2: " << params_b[2] << " B3: " << params_b[3] << endl;
			//if (255 * (ImgValue_R[i][j] - BValue_Vector_R[i][j]) <= 0){
			//	std::cout << "("<<i<<","<<j<<") = "<< 255 * (ImgValue_R[i][j] - BValue_Vector_R[i][j]) << endl;
			//	cont = cont + 1;
			//}
		}
	}
	//std::cout << "cont: " << cont << endl;
	// Imprimir los valores de la matriz
	//for (int i = 0; i < H; i++) {
	//	for (int j = 0; j < W; j++) {
	//		std::cout << "Valor en la posici�n (" << i << ", " << j << "): ";
	//		std::cout << static_cast<int>(Dc.at<cv::Vec3b>(i, j)[0]) << ", ";
	//		std::cout << static_cast<int>(Dc.at<cv::Vec3b>(i, j)[1]) << ", ";
	//		std::cout << static_cast<int>(Dc.at<cv::Vec3b>(i, j)[2]) << std::endl;
	//	}
	//}
	// Mostrar la imagen en una ventana
	//cv::namedWindow("ImagenDc", cv::WINDOW_NORMAL);
	//cv::namedWindow("ImageIc", cv::WINDOW_NORMAL);
	cv::Mat brighness = Dc * 2;
	
	cv::imshow("ImagenDc", brighness);
	//cv::imshow("ImagenIc", Dc);
	cv::waitKey(0);
	
	//cv::imwrite("ennhanced_" + std::to_string(var) + "_image.png", brighness);
	//cv::destroyAllWindows();


}


//The function construct_neighborhood_map takes an input grayscale image and constructs a
//neighborhood map where each pixel in the output corresponds to the size of its local 
//neighborhood. The neighborhood size is determined by counting the number of pixels in a 
//circular region around each pixel with a given radius. The output map can be used for 
//various image processing tasks like filtering, segmentation, and feature extraction.
//std::pair<cv::Mat, int>
std::pair<cv::Mat, int> construct_neighborhood_map(cv::Mat depths, double epsilon = 0.05){
	//epsilon esta por ejemplo entre el pixel mas bajo y alto 
	//divide el rango en grupos de 2 pixeles por ejemplo
	//si elijo el eposion en 10 pedacitos de 10 hasta 13 
	//esta definiendo en cuantas particiones se puede generar
	double min_depth, max_depth;                                                                //eps = (np.max(depths) - np.min(depths)) * epsilon
	cv::minMaxLoc(depths, &min_depth, &max_depth);

	double eps = (max_depth - min_depth) * epsilon;
	//std::cout << "max_depths: " << max_depth << " min depths: " << min_depth <<" eps: " <<eps<<" epsilon: " << epsilon << endl;
	cv::Mat nmap = cv::Mat::zeros(depths.size(), depths.type());                                // nmap = np.zeros_like(depths).astype(np.int32)
	//print nmaps values the next comment

	//for (int i = 0; i < nmap.rows && i < 7; i++) {
	//	for (int j = 0; j < nmap.cols && j < 7; j++) {
	//		std::cout << nmap.at<int>(i, j) << " ";
	//	}
	//	std::cout << std::endl;
	//}
	//std::wcout << " nmaps rows: " << nmap.rows << " nmap cols: " << nmap.cols << " nmap channel: " << nmap.channels() << endl;
	//while(cv::countNonZero(nmap) > 0)

	int n_neighborhoods = 1;
	while (cv::countNonZero(nmap == 0) > 0) {                                                   //while np.any(nmap == 0):
		// Do the rest of the processing here...
		std::vector<cv::Point> locs;
		cv::findNonZero(nmap == 0, locs);
		std::vector<int> locs_x, locs_y;
		for (const auto& p : locs) {
			locs_x.push_back(p.y);
			locs_y.push_back(p.x);
		}
		//std::cout << "locs_x size: " << locs_x.size() << std::endl;
		
		// Generate a random starting index
		// Generate random start index
		//std::cout << "rand: " << RAND_MAX << endl; //solo se puede elegir valores aleatoriaos entre 32767
		//porue lo hace aleatorio y porque no por segmentos
		//intnetar hacelo por barrido
		// https://stackoverflow.com/questions/9775313/extend-rand-max-range
		//use this code to correct the rand
		//int start_index = (rand()* RAND_MAX + rand()) % locs_x.size();  //locs.size
		//
		//we wont have the same problem because the maximum number is  2^19937 - 1
		std::random_device rd;                                                                 //start_index = np.random.randint(0, len(locs_x))
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> dis(0, locs_x.size() - 1);
		int start_index = dis(gen);

		int start_x = locs_x[start_index];                                                     //start_x, start_y = locs_x[start_index], locs_y[start_index]
		int start_y = locs_y[start_index];
		//std::cout << "start index: " << start_index << " start_x: "<<start_x<<" start_y: "<< start_y<<  endl;
		

		// Create an empty deque of pairs
		std::deque<std::pair<int, int>> q;                                                     //q = collections.deque()

		// Append the starting point to the deque
		q.push_back(std::make_pair(start_x, start_y));                                         //q.append((start_x, start_y))
		
		// Print the contents of the deque
		//for (const auto& p : q) {
		//	std::cout << "(" << p.first << ", " << p.second << ")" << std::endl;
		//}
		//int count = 0;
		
		while (!q.empty()) {
		//while (q.size() > 0){
		//while(count <= 5){
			// pop the front element from the deque
			std::pair<int, int> current = q.back();   //front                                   //x, y = q.pop()
			// unpack the pair into x and y
			int x = current.first;
			int y = current.second;
			q.pop_back();
						
			
			//std::cout << "x: " << x << " y: " << y << endl;
			//std::cout << "count: " << count << " depths: " << depths.at<float>(x, y) << endl;
			//std::cout << "q size: " << q.size() << endl;
			//for (const auto& p : q) {
			//	std::cout << "(" << p.first << "," << p.second << ") ";
			//}
			//std::cout << std::endl;
			//std::cout << "x: " << x << " y: " << y<< endl;
		    //std::cout << q.size() << endl;
			if (std::abs(depths.at<float>(x, y) - depths.at<float>(start_x, start_y)) <= eps) { //if np.abs(depths[x, y] - depths[start_x, start_y]) <= eps:
				nmap.at<int>(x, y) = n_neighborhoods;                                           //nmap[x, y] = n_neighborhoods
				//std::cout << "depths.rows: " << depths.rows << endl;
				if (0 <= x && x < depths.rows - 1) {                                            //if 0 <= x < depths.shape[0] - 1:
					int x2 = x + 1, y2 = y;                                                     //x2, y2 = x + 1, y
					if (nmap.at<int>(x2, y2) == 0) {                                            //if nmap[x2, y2] == 0:
						q.push_back(std::make_pair(x2, y2));                                    //q.append((x2, y2))
					}
				}

				if (1 <= x && x < depths.rows) {                                                //if 1 <= x < depths.shape[0]:
					int x2 = x - 1, y2 = y;                                                     //x2, y2 = x - 1, y
					if (nmap.at<int>(x2, y2) == 0) {                                            //if nmap[x2, y2] == 0:
						q.push_back(std::make_pair(x2, y2));                                    //q.append((x2, y2))
					}
				}

				if (0 <= y && y < depths.cols - 1) {                                            //if 0 <= y < depths.shape[1] - 1:
					int x2 = x, y2 = y + 1;                                                     //x2, y2 = x, y + 1
					if (nmap.at<int>(x2, y2) == 0) {                                            //if nmap[x2, y2] == 0:
						q.push_back(std::make_pair(x2, y2));                                    //q.append((x2, y2))
					}
				}

				if (1 <= y && y < depths.cols) {                                                //if 1 <= y < depths.shape[1]:
					int x2 = x, y2 = y - 1;                                                     //x2, y2 = x, y - 1
					if (nmap.at<int>(x2, y2) == 0) {                                            //if nmap[x2, y2] == 0:
						q.push_back(std::make_pair(x2, y2));                                    //q.append((x2, y2))
					}
				}

				//count += 1;
			}
			
		}
		//std::cout << "n_neirbor: " << n_neighborhoods << endl;
		n_neighborhoods += 1;
	}
	//std::cout << "n_neighborhoods aaaaaaaaaaaaaaa: " << n_neighborhoods << endl;

	std::map<int, int> zero_sizes_map;
	for (int i = 0; i < 50; ++i) {
		for (int j = 0; j < 50; ++j) {
			if (depths.at<float>(i, j) == 0) {
				int zero_size = nmap.at<cv::Vec3f>(i, j)[2];
				zero_sizes_map[zero_size]++;
			}
		}
	}
	std::vector<std::pair<int, int>> zero_sizes_arr;
	zero_sizes_arr.resize(zero_sizes_map.size());
	std::copy(zero_sizes_map.begin(), zero_sizes_map.end(), zero_sizes_arr.begin());
	std::sort(zero_sizes_arr.begin(), zero_sizes_arr.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
		return a.second > b.second;
		});

	std::cout << "zeros size arr: " << zero_sizes_arr.size() << endl;
	//visualize_nmap(nmap);
	return std::make_pair(nmap, n_neighborhoods);
}

//refine_neighborhood_map
//used the construct neighborhood map function
//nmaps works as an input
//it also has two floats
//reduces the number of small regions in the neighborhood map
//refine_neighborhood_map is a function that takes a neighborhood map and refines it by
//removing small regions and filling gaps. It does this by identifying regions based on a 
//minimum size and then dilating them with a given radius. The resulting map is a binary 
//image where each pixel represents a region.
cv::Mat refine_neighborhood_map(cv::Mat nmap,int min_size = 10, int radius = 3) {              
	cv::Mat refined_nmap = cv::Mat::zeros(nmap.size(), CV_8UC1);                            //refined_nmap = np.zeros_like(nmap)
	std::vector<int> vals;                                                                  //vals, counts = np.unique(nmap, return_counts=True)
	std::vector<int> counts;
	int current_val = nmap.at<uchar>(0, 0);
	int count = 1;                                                                          //num_labels = 1
	auto begin = nmap.begin<uchar>();
	auto end = nmap.end<uchar>();
	for (auto it = begin + 1; it != end; ++it) {                        
		if (*it != current_val) {
			vals.push_back(current_val);
			counts.push_back(count);
			current_val = *it;
			count = 1;
		}
		else {
			++count;
		}
	}
	vals.push_back(current_val);
	counts.push_back(count);
	//std::cout << "vals: " << vals.size() << endl;
	//std::cout << "counts: " << counts.size() << endl;
	//for (int i = 0; i < 10; i++)
	//{
	//	std::cout << vals[i]<<" , ";
	//}
	//std::cout << endl;

	std::vector<int> neighborhood_sizes;                                                     //neighborhood_sizes = sorted(zip(vals, counts), key=lambda x: x[1], reverse=True)
	for (size_t i = 0; i < vals.size(); i++) {
		if (counts[i] >= min_size) {
			neighborhood_sizes.push_back(vals[i]);
		}
	}
	std::cout << "neigborhood size: " << neighborhood_sizes.size() << endl;
	for (int y = 0; y < nmap.rows; y++) {
		for (int x = 0; x < nmap.cols; x++) {
			int center_val = nmap.at<uchar>(y, x);
			bool center_is_valid = std::find(neighborhood_sizes.begin(), neighborhood_sizes.end(), center_val) != neighborhood_sizes.end();
			if (center_is_valid) {
				bool all_neighbors_same = true;
				for (int ny = y - radius; ny <= y + radius; ny++) {
					for (int nx = x - radius; nx <= x + radius; nx++) {
						if (ny < 0 || nx < 0 || ny >= nmap.rows || nx >= nmap.cols || (ny == y && nx == x)) {
							continue;
						}

						if (nmap.at<uchar>(ny, nx) != center_val) {
							all_neighbors_same = false;
							break;
						}
					}
					if (!all_neighbors_same) {
						break;
					}
				}
				if (all_neighbors_same) {
					refined_nmap.at<uchar>(y, x) = center_val;
				}
			}
		}
	}
	std::cout << "Refined rows: " << refined_nmap.rows << " cols: " << refined_nmap.cols << " channels: " << refined_nmap.channels() << endl;
	return refined_nmap;
}

void refine_neighborhood_map_2(cv::Mat nmap, int min_size = 10, int radius = 3) {
	cv::Mat refined_nmap = cv::Mat::zeros(nmap.size(), nmap.type());                               //refined_nmap = np.zeros_like(nmap)
	
	cv::Mat flat_nmap = nmap.reshape(1, nmap.total()); // Flatten nmap into a single row           //vals, counts = np.unique(nmap, return_counts=True)
	//std::wcout << "flat_nmap: " << flat_nmap.cols << " flat_ nmap:rows "<<flat_nmap.rows << endl;
	//for (int i = 0; i < flat_nmap.rows; i++){
	//	std::cout << "flat_nmap: " << static_cast<int>(flat_nmap.at<uchar>(i, 0)) << endl;
	//}
	std::set<int> unique_vals;
	int min_val = std::numeric_limits<int>::max();
	// Iterate over the flattened nmap, add each unique value to the set and increment the count
	for (int i = 0; i < flat_nmap.rows; i++) {
		int val = flat_nmap.at<uchar>(i, 0);
		unique_vals.insert(val);
		if (val < min_val) {
			min_val = val;
		}
	}
	std::vector<int> vals(unique_vals.size());
	std::vector<int> counts(vals.size());

	int i = 0;
	// Iterate over the unique values, subtract the minimum value, and store them in the vals vector
	for (const auto& val : unique_vals) {
		vals[i] = val - min_val;
		counts[i] = std::count(flat_nmap.begin<uchar>(), flat_nmap.end<uchar>(), val);
		i++;
	}
	//std::cout << "counts size: " << counts.size() << endl;
	//for (int i = 0; i < counts.size(); i++) {
	//	std::cout << counts[i] << ",";
	//}
	//std::cout << endl;
	//std::cout << "vals size: " << vals.size() << endl;
	//for (int i = 0; i < counts.size(); i++) {
	//	std::cout << vals[i] << ",";
	//}
	//std::cout << endl;

		
	
	//this functions works and starts here
	std::vector<std::pair<int, int>> neighborhood_sizes;                                           //neighborhood_sizes = sorted(zip(vals, counts), key=lambda x: x[1], reverse=True)
	for (int i = 0; i < vals.size(); i++) {
		neighborhood_sizes.push_back(std::make_pair(vals[i], counts[i]));
	}
	std::sort(neighborhood_sizes.begin(), neighborhood_sizes.end(), [](std::pair<int, int>& x, std::pair<int, int>& y) {
		return x.second > y.second;
	});
	//and finish here
	
	//for (const auto& elem : neighborhood_sizes) {
	//	std::cout << elem.first << " : " << elem.second << std::endl;
	//}

	int num_labels = 1;                                                                            //num_labels = 1
	

	for (const auto& elem : neighborhood_sizes) {                                                  //for label, size in neighborhood_sizes:  
		int label = static_cast<int>(elem.first + min_val);
		int size = static_cast<int>(elem.second);
		//std::cout << "label: " << label << " size: " << size << endl;
		if (size >= min_size && label != 0) {                                                     //if size >= min_size and label != 0:
			                   //refined_nmap[nmap == label] = num_labels
			refined_nmap.forEach<uchar>([&](uchar& pixel, const int* position) -> void {          
				if (nmap.at<uchar>(position[0], position[1]) == label && size >= min_size && label != 0) {
					pixel = num_labels;
				}
				});
			num_labels++;                                                                        //num_labels += 1
		}
	}
	std::cout << "counts: " << counts.size() << endl;
	std::cout << "num_labels: " << num_labels << endl;
	//visualize_nmap(refined_nmap);

	//for (const auto& elem : neighborhood_sizes) {                                                //for label, size in neighborhood_sizes:
	//	int label = elem.first;
	//	int size = elem.second;
	//	if (size < min_size && label != 0) {                                                     //if size < min_size and label != 0:
	//		std::cout << "ebntro" << endl;		
	//	}
	//}

		
	//for (int i = 0; i < 200; i++) {
	//	for (int j = 0; j < 200; j++) {                              //tener en cuenta que tipo de datos llegan al resultado
	//		std::cout << static_cast<double>(refined_nmap.at<uchar>(i, j)) << " ";
	//	}
	//}
	//std::wcout << endl;
	//visualize_nmap(refined_nmap);
	
}


cv::Mat fill_depth_map(cv::Mat depth_map, cv::Mat mask){
	std::cout << "entering fill_depth_map function: " << endl;
	int H = depth_map.rows;
	int W = depth_map.cols;
	
	cv::Mat filled_depth_map = depth_map.clone();
	cv::Mat detect_depth_map = depth_map.clone(); //detectan elmapa de profundidad y los escriben en este mapa
	cv::Mat mask_32f;
	mask.convertTo(mask_32f, CV_32F);
	cv::bitwise_not(mask, mask_32f);
	//cv::imshow("mascara 2", mask_32f);
	//cv::waitKey(0);

	//cv::Mat empty_map = cv::Mat::zeros(depth_map.size(), depth_map.type());
	//std::cout << "empty map: " << empty_map.rows << empty_map.cols << endl;
	for (int i = 0; i < H; i++){
		int Hole = 0;
		int count = 0;
		int initial_pixel = 0;
		for (int j = 0; j < W; j++){
			if (mask_32f.at<uchar>(i, j) == 0.0f && Hole == 0){
				Hole = 1;
				count = 1;
				initial_pixel = filled_depth_map.at<uchar>(i, j - 1);
				//std::wcout << "(i,j): ("<<i<<","<<j<<")" << " - initial pixel: " << initial_pixel << endl;
				//cambiar el valor de 0 del inicio de la matriz para que el codigo no se confunda
				if (j == 0 && mask_32f.at<uchar>(i, j) == 0){
					//filled_depth_map.at<uchar>(i, j - 1) == 1;
					initial_pixel = filled_depth_map.at<uchar>(i, j);
				}
			}
			else{
				if (mask_32f.at<uchar>(i, j) == 0.0f && Hole == 1){
					count = count + 1;
				}
				else{
					//en caso de que ya no encuentre hueco pero siga dentro del holw procedera a guardar los datos dentro de una matriz extra
					if (mask_32f.at<uchar>(i, j) != 0.0f && Hole == 1){
						//filled_depth_map.at<uchar>(i, j - 1) = initial_pixel;
						//filled_depth_map.at<uchar>(i, j - 2) = count;   //para guardar la informacaion del count de manera adecuada se recomienda reemplazar el uchart por int
						detect_depth_map.at<int>(i, j) = initial_pixel;
						detect_depth_map.at<int>(i, j - 1) = count;
						//std::wcout << "(i,j): (" << i << "," << j << ")" << " - count: "<<count <<"-" << detect_depth_map.at<int>(i, j - 1) << " initial pixel: " << detect_depth_map.at<int>(i, j) << " last pixel: " << filled_depth_map.at<uchar>(i, j) << endl;
						Hole = 0;
						count = 0;
						initial_pixel = 0;
					}
					
				}
			}

			//condicional puesto para probar el algoritmo de llenado de pixeles
			if (j == (W - 1) && mask_32f.at<uchar>(i, j) == 0.0f){
				filled_depth_map.at<uchar>(i, j) = initial_pixel;
				detect_depth_map.at<int>(i, j) = initial_pixel;
				detect_depth_map.at<int>(i, j - 1) = count;
				//filled_depth_map.at<uchar>(i, j - 1) = static_cast<int>(initial_pixel);
				//filled_depth_map.at<uchar>(i, j - 2) = static_cast<int>(count);
				//std::wcout << "(i,j): (" << i << "," << j << ")" << " - count: " << count << "-" << detect_depth_map.at<int>(i, j - 1) << " initial pixel: " << detect_depth_map.at<int>(i, j) << " last pixel: " << filled_depth_map.at<uchar>(i, j) << endl;
				Hole = 0;
				count = 0;
				initial_pixel = 0;
			}
		}
	}
	//cv::imshow("Filled Depth Map", filled_depth_map);
	//cv::waitKey(0); 
	
	// Generate mask using thresholding
	cv::Mat mask_right2left = (filled_depth_map > 0);
	// Invert mask
	//cv::Mat inverted_mask;
	//cv::bitwise_not(mask2, inverted_mask);
	//cv::imshow("mask_right2left",mask_right2left);
	//cv::waitKey(0);
	//for (int i = 0; i < W; i++){
	//	std::cout << static_cast<int>(detect_depth_map.at<int>(0, i)) << ", ";
	//}

	std::cout << endl;
	
	for (int i = 0; i < H; i++){
		int Hole = 0;
		int initial_pixel = 0;
		int initial_position = 0;
		double m = 0;
		int count = 0;
		for (int j = W - 1; j >= 0; j--){
			if (mask_right2left.at<uchar>(i, j) == 0.0f && Hole == 0){
				initial_pixel = static_cast<int>(detect_depth_map.at<int>(i, j + 1));
				int Final_pixel = filled_depth_map.at<uchar>(i, j + 1);
				count = static_cast<int>(detect_depth_map.at<int>(i, j));
				initial_position = j + 1 - count;
				Hole = 1;
				//std::wcout << "(i,j): (" << i << "," << j << ")" << " - count: " << count << "-" << detect_depth_map.at<int>(i, j) << " initial pixel: " << detect_depth_map.at<int>(i, j + 1)<<" - "<< initial_pixel << " last pixel: " << filled_depth_map.at<uchar>(i, j + 1) << endl;
				//std::cout << "initial_position: " << initial_position << endl;
				float superior = (Final_pixel - initial_pixel);
				m = double(superior / count);
				if (m < 0) {
					m = -m;
				}
				//std::cout << "m: " << m << " count: "<<count << " Final_pixel: "<<Final_pixel<<" initial_pixel: "<<initial_pixel<<endl;
				
				filled_depth_map.at<uchar>(i, j) = int(m * (count - 1) + initial_pixel);
			}
			else{
				if (mask_32f.at<uchar>(i, j) == 0.0f && Hole == 1) {
					//if ((m * (j - initial_position) + initial_pixel) < 0) {
					//	std::cout << "imprime: " << (m * (count - 1) + initial_pixel) << endl;
					//}
					
					filled_depth_map.at<uchar>(i, j) = int(m * (j - initial_position) + initial_pixel);
				}
				else{
					if (mask_32f.at<uchar>(i, j) != 0.0f && Hole == 1) {
						Hole = 0;
						count = 0;
						initial_pixel = 0; 
						initial_position = 0;
						m = 0;
					}
				}
			}

		}
	}
	
	cv::Mat mask2 = (filled_depth_map > 0);
	//cv::imshow("inverted_mask", mask2);
	//cv::waitKey(0);
	//cv::imshow("mask 2", mask2);
	//cv::waitKey(0);
	
	//ahora se hace el mismo proceso para el eje vertical 
	for (int i = 0; i < W; i++) {
		int Hole = 0;
		int count = 0;
		int initial_pixel = 0;
		int contador = 0;
		for (int j = 0; j < H; j++) {
			if (mask2.at<uchar>(j, i) == 0.0f && Hole == 0) {
				Hole = 1;
				count = count + 1;
				//std::cout << "(j,i): (" << j << "," << i << ")" << endl;
				if (j == 0 && mask2.at<uchar>(j, i) == 0) {
					filled_depth_map.at<uchar>(j - 1, i) == 1;
					initial_pixel = filled_depth_map.at<uchar>(j, i);
				}
				else {
					initial_pixel = filled_depth_map.at<uchar>(j - 1, i);
				}
			}
			else{
				if (mask2.at<uchar>(j, i) == 0.0f && Hole == 1) {
					count = count + 1;
				}
				else{
					if (mask2.at<uchar>(j, i) != 0.0f && Hole == 1) {
						detect_depth_map.at<int>(j, i) = initial_pixel;
						detect_depth_map.at<int>(j - 1, i) = count;
						//std::wcout << "(j,i): (" << j << "," << i << ")" << " - count: " << count << "-" << detect_depth_map.at<int>(j - 1, i) << " initial pixel: " << detect_depth_map.at<int>(j, i) << " last pixel: " << filled_depth_map.at<uchar>(j, i) << endl;
						Hole = 0;
						count = 0;
						initial_pixel = 0;
					}
				}
			}
			//condicional puesto para probar el algoritmo de llenado de pixeles
			if (j == (H - 1) && mask2.at<uchar>(j, i) == 0.0f) {
				filled_depth_map.at<uchar>(j, i) = static_cast<int>(initial_pixel);
				detect_depth_map.at<int>(j, i) = initial_pixel;
				detect_depth_map.at<int>(j - 1,  i) = count;
				//std::wcout << "(j,i): (" << j << "," << i << ")" << " - count: " << count << "-" << detect_depth_map.at<int>(j - 1, i) << " initial pixel: " << detect_depth_map.at<int>(j, i) << " last pixel: " << filled_depth_map.at<uchar>(j, i) << endl;
				Hole = 0;
				count = 0;
				initial_pixel = 0;
			}
		}
	}
	cv::Mat mask3 = (filled_depth_map > 0);
	//cv::imshow("mask3", mask3);
	//cv::waitKey(0);

	for (int i = 0; i < W; i++) {
		int Hole = 0;
		int initial_pixel = 0;
		int initial_position = 0;
		double m = 0;
		int count = 0;
		for (int j = H - 1; j >= 0; j--) {
			//std::cout << "j: " << j << endl;
			if (mask3.at<uchar>(j, i) == 0.0f && Hole == 0) {
				initial_pixel = static_cast<int>(detect_depth_map.at<int>(j + 1, i));
				int Final_pixel = filled_depth_map.at<uchar>(j + 1, i);
				count = static_cast<int>(detect_depth_map.at<int>(j, i));
				initial_position = j + 1 - count;
				
				Hole = 1;
				float superior = (Final_pixel - initial_pixel);
				m = double(superior / count);
				if (m < 0) {
					m = -m;
				}
				//std::cout << "initial_pixel: " << Final_pixel << endl;
				filled_depth_map.at<uchar>(j, i) = int(m * (count - 1) + initial_pixel);
				
			}
			else{
				if (mask3.at<uchar>(j, i) == 0.0f && Hole == 1) {
					filled_depth_map.at<uchar>(j, i) = int(m * (j - initial_position) + initial_pixel);
				}
				else{
					if (mask3.at<uchar>(j, i) != 0.0f && Hole == 1) {
						Hole = 0;
						count = 0;
						initial_pixel = 0;
						initial_position = 0;
						m = 0;
					}
				}
			}
		}
	}
	//cv::imshow("Filled Depth Map final", filled_depth_map);
	//cv::waitKey(0);
	//cv::Mat mask3 = (filled_depth_map > 0);
	//cv::imshow("inverted_mask", mask3);
	//cv::waitKey(0);

	return filled_depth_map;

}

cv::Mat inpaint_depth_map_patch(cv::Mat depth_map, cv::Mat mask, int patch_size) {
	cv::Mat filled_depth_map = depth_map.clone();
	std::cout << "filled depth map: " << filled_depth_map.rows << " width: " << filled_depth_map.cols << " channels: " << filled_depth_map.channels() << endl;
	cv::Mat mask_32f;
	mask.convertTo(mask_32f, CV_32F);
	cv::bitwise_not(mask, mask_32f);
	cv::Mat inpainted_depth_map = cv::Mat::zeros(depth_map.rows, depth_map.cols, CV_32F);
	//cv::imshow("Filled Depth Map", mask_32f);
	//cv::waitKey(0);
	// Fill holes in depth map using Poisson image editing for each pixel
	//search the horizontal and vertical values
	//sacar la derivada para la derivada
	for (int i = 0; i < filled_depth_map.rows; i++) {
		for (int j = 0; j < filled_depth_map.cols; j++) {
			if (mask_32f.at<uchar>(i, j) == 0.0f && j != 0) {
				float sum = 0;
				int count = 0;
				for (int ii = -patch_size / 2; ii <= patch_size / 2; ii++) {
					for (int jj = -patch_size / 2; jj <= patch_size / 2; jj++) {
						int i2 = i + ii;
						int j2 = j + jj;
						if (i2 < 0 || i2 >= filled_depth_map.rows || j2 < 0 || j2 >= filled_depth_map.cols) {
							continue;
						}
						if (mask.at<uchar>(i2, j2) == 255) {
							sum += depth_map.at<float>(i2, j2);
							count++;
						}
					}
				}
				if (count > 0) {
					float avg = sum / count;
					filled_depth_map.at<uchar>(i, j) = filled_depth_map.at<uchar>(i, j-1);
				}
			}
		}
	}
	//el error se puede deber a que al usar 0 como inicial la recta haya superado el valor necesario para el x y por eso este saliendo ese error
	//cv::imshow("Filled Depth Map", depth_map);
	//cv::waitKey(0);
	//cv::imshow("Filled Depth Map", filled_depth_map);
	//cv::waitKey(0); // Wait for a key press
	return depth_map;
}


//run pipeline
//runs all the code
//run the whole code
void run_pipeline(cv::Mat img, cv::Mat depths, std::vector<std::vector<float>> depths_vector, float* args) {
	std::string output_graph = "output_graph";                                                                        //string output_graphs = 'output_graphs';
	std::string preprocess_for_monodepth = "store_true";                                                              //store_true equalize_image = 'store true'
	std::string equalize_image = "store_true";                                                                        //store true  equalize_image = 'store true'
	
	std::cout<<"Estimating backscatter..."<<endl;                                                                     //print('Estimating backscatter...', flush=True)
	auto startTime1 = std::chrono::high_resolution_clock::now();
	std::vector<cv::Point2f> pts_r;
	std::vector<cv::Point2f> pts_g;
	std::vector<cv::Point2f> pts_b;
	std::tie(pts_r, pts_g, pts_b) = find_backscatter_estimation_points(img, depths, 10, 0.01, 20, 0.1);             //ptsR, ptsG, ptsB = find_backscatter_estimation_points(img, depths, fraction=0.01, min_depth_percent = args[3])
	auto endTime1 = std::chrono::high_resolution_clock::now();
	auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(endTime1 - startTime1);
	std::cout << "construct_neighborhood_map map time: " << duration1.count() << " milliseconds." << std::endl;

	// Access the individual vectors
	//std::cout << "vector1: " << pts_r.size() << endl;
	//for (const auto& point : points_b) {
	//	std::cout << "Point: (" << point.x << ", " << point.y << ")" << std::endl;
	//}

	std::cout<< "Find backscatter coefficients" << endl;
	int restarts;
	//Br, coefsR = find_backscatter_values(ptsR, depths, restarts = 25)
	
	//find_backscatter_values(pts_r, depths, restarts = 10, 0.05);
	
	//Bg, coefsG = find_backscatter_values(ptsG, depths, restarts = 25)
	//Bb, coefsB = find_backscatter_values(ptsB, depths, restarts = 25)
	find_backscatter_value_2(pts_r, depths, restarts = 25);
	
	std::cout << "constructing neighborhood map ..." << endl;
	auto startTime2 = std::chrono::high_resolution_clock::now();
	std::pair<cv::Mat, int> result = construct_neighborhood_map(depths, 0.1);                             //nmap, _ = construct_neighborhood_map(depths, 0.1)
	auto endTime2 = std::chrono::high_resolution_clock::now();
	auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(endTime2 - startTime2);
	std::cout << "construct_neighborhood_map map time: " << duration2.count() << " milliseconds." << std::endl;
	cv::Mat nmaps = result.first;
	int n_neighborhood = result.second;
	

	//visualize_nmap(nmaps);
	//for (int i = 0; i < 200; i++) {
	//	for (int j = 0; j < 200; j++) {                              //tener en cuenta que tipo de datos llegan al resultado
	//		std::cout << static_cast<double>(nmaps.at<uchar>(i, j)) << " ";
	//	}
	//}
	//std::wcout << endl;
	std::cout << "n_neigborhood: " << n_neighborhood << endl;

	std::cout << "Refining neighborhood map..." << endl;
	// falta imprimir el valor de refined_nmap
	//std::cout << "salio" << endl;
	refine_neighborhood_map_2(nmaps, 10, 3);                                                              //nmap, n = refine_neighborhood_map(nmap, 50)
	
	Backscatter_removal(img, depths, depths_vector);
	//std::cout << "Vector height: " << depths_vector.size() << " width: " << depths_vector[0].size() << endl;
}

//fill the depth map with data 
//restores the holes provided by the metashape
//code
void fill_depth_colorization(cv::Mat imgRgb, cv::Mat imgDepthInput, int alpha = 1) {

	//cv::imshow("Filled Depth Map final", imgDepthInput);
	//cv::waitKey(0);

	cv::Mat mask = (imgDepthInput > 0);
	// Invert mask

	cv::Mat inverted_mask;
	cv::bitwise_not(mask, inverted_mask);

	//cv::imshow("Filled Depth Map final", imgDepthInput);
	//cv::waitKey(0);
	
	//cv::imshow("mascara 1", inverted_mask);
	//cv::waitKey(0);
	// Inpaint depth map using generated mask
	//cv::Mat inpainted_depth_map = inpaint_depth_map_patch(imgDepthInput, inverted_mask,15);
	//cv::imshow("mascara 2", inverted_mask);
	//cv::waitKey(0);
	//cv::imshow("imgDepthInput", imgDepthInput);
	//cv::waitKey(0);
	auto startTime = std::chrono::high_resolution_clock::now();
	cv::Mat filled_depth_map = fill_depth_map(imgDepthInput, inverted_mask);
	auto endTime = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

	// Output the duration
	std::cout << "Filling depth map time: " << duration.count() << " milliseconds." << std::endl;

	std::cout << " rows: " << filled_depth_map.rows << " cols: " << filled_depth_map.cols << " channels: " << filled_depth_map.channels() << endl;
	//cv::imshow("Filled Depth Map final", filled_depth_map);
	//cv::waitKey(0);
}

//main function
//depth map input
//rgb image input
int main() {
	//inputs
	//put them in run pipeline
	float f = 2.0;                                                                                                      //f value(controls brightness)
	float l = 0.5;                                                                                                      //l value(controls balance of attenuation constants)
	float p = 0.01;                                                                                                     //p value(controls locality of illuminant map)
	float min_depth = 0.1;                                                                                              //Minimum depth value to use in estimations(range 0 - 1)
	float max_depth = 1.0;                                                                                              //Replacement depth percentile value for invalid depths(range 0 - 1)
	float spread_data_fraction = 0.01;                                                                                  //Require data to be this fraction of depth range away from each other in attenuation estimations
	float size = 1024;    //320                                                                                          //size of the output change depending the size of the image
	float monodepth = 2.0;
	float monodepth_add_depth = 2.0;
	float monodepth_multiply_depth = 10.0;                                                                              //multiplicative value for depth map

	//Anisotropic diffuse filter input
	std::string img_original = "image_original.png";
	std::string depth_original = "depth_map_original.png";
	
	auto img_depth_originals = load_image_and_depth_map(img_original, depth_original, size);
	cv::Mat imgs_original = img_depth_originals.first; 
	cv::Mat depths_original = img_depth_originals.second;
	std::cout << "original img: " << imgs_original.rows << " width: " << imgs_original.cols << " , channels: " << imgs_original.channels() << endl;
	std::cout << "original depths: " << depths_original.rows << " width: " << depths_original.cols << " , channels: " << depths_original.channels() << endl;
	

	//for (int i = 0; i < 10; i++){
	//	for (int j = 0; j < depths_original.cols; j++) {
	//		std::cout << static_cast<double>(depths_original.at<uchar>(i, j)) << " ";
	//	}
	//	std::cout << std::endl;
	//}
	//cv::normalize(depths_original, depths_original, 0, 255, cv::NORM_MINMAX);
	//cv::imshow("Depth map", depths_original);
	fill_depth_colorization(imgs_original, depths_original, 1);

	float args[] = {f,l,p,min_depth,max_depth,spread_data_fraction,size,monodepth,monodepth_add_depth,monodepth_add_depth,monodepth_multiply_depth};

	//cameras and depth image input
	//std::string Input_im_name = "real48_image.png";
	//std::string depth_map_name = "48_Depth_map.png";

	
	std::string cadena_img = "cutter" + std::to_string(var) + ".png";
	std::string cadena_depth = "cutter" + std::to_string(var) + "_depth_map.png";
	std::cout << cadena_img << endl;
	std::cout << cadena_depth << endl;

	std::string Input_im_name = cadena_img;        // "cutter2.png";
	std::string depth_map_name = cadena_depth;     //"cutter2_depth_map.png";

	auto img_depth_map = load_image_and_depth_map(Input_im_name, depth_map_name, size);
	cv::Mat img = img_depth_map.first;
	cv::Mat depths = img_depth_map.second;
	//for (int i = 0; i < 10; i++){
	//	for (int j = 0; j < depths.channels(); j++) {
	//		std::cout << static_cast<double>(depths.at<uchar>(i, j)) << " ";
	//	}
	//	std::cout << std::endl;
	//}

	//cv::imshow("Filled Depth Map final", depths);
	//cv::waitKey(0);

	std::cout << "Height: " << img.rows << ", width: " << img.cols << ", channels: " << img.channels() << endl;
	std::cout << "Height: " << depths.rows << ", width: " << depths.cols << ", channels: " << depths.channels() << endl;
	
	std::tuple<cv::Mat, std::vector<std::vector<float>>> result = preprocess_monodepth_map(depths, monodepth_add_depth, monodepth_multiply_depth);
	cv::Mat depth_preprocess = std::get<0>(result);
	std::vector<std::vector<float>> depths_preprocess_float = std::get<1>(result);
	std::cout << "Height: " << depth_preprocess.rows << ", width: " << depth_preprocess.cols << ", channels: " << depth_preprocess.channels() << endl;
	std::cout << "Vector height: " << depths_preprocess_float.size() << " width: " << depths_preprocess_float[0].size() << endl;
	//for (int i = 0; i < depths_preprocess_float.size(); i++){
	//	for (int j = 0; j < depths_preprocess_float[0].size(); j++){
	//		std::cout << " dpths values: " << depths_preprocess_float[i][j] << endl;
	//	}
	//}

	//visualizar imagen
	//cv::namedWindow("Image", cv::WINDOW_NORMAL);
	//cv::imshow("Image", img);
	// Wait for a key press and exit
	//cv::waitKey(0);
	//cv::normalize(depths, depths, 0, 255, cv::NORM_MINMAX);
    //cv::imshow("Depth map", depth_preprocess);
	//cv::waitKey(0);
	//visualize_nmap(depth_preprocess);
	run_pipeline(img, depth_preprocess , depths_preprocess_float, args);

	return 0;
}