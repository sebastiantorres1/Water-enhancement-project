#pragma once
#include <vector>

void curveFittingCUDA(const std::vector<double>& x, const std::vector<double>& y, std::vector<double>& params);

//version Naive
void curveFittingCUDA2(const std::vector<double>& x, const std::vector<double>& y, std::vector<double>& params);
//vercion optimizada
void curveFittingCUDA3(const std::vector<double>& x,
	const std::vector<double>& y_r, const std::vector<double>& y_g, const std::vector<double>& y_b,
	std::vector<double>& params_r, std::vector<double>& params_g, std::vector<double>& params_b);