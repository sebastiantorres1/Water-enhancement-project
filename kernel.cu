
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "Header.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <numeric>




__global__ void errorFunctionCUDA(const double* params, const double* x, const double* y, double* error, size_t data_size) {
	int locs = threadIdx.x + blockIdx.x * blockDim.x;
	if (locs < data_size){
		double b_1 = params[0];
		double b_2 = params[1];
		double b_3 = params[2];
		double b_4 = params[3];
		double result = b_1 * (1 - exp(b_2 * x[locs])) + (b_3 * exp(-b_4 * x[locs])) - y[locs];
		error[locs] = result * result;
		//atomicAdd(&error[0], result * result);
	}
}

void curveFittingCUDA(const std::vector<double>& x, const std::vector<double>& y, std::vector<double>& params){
	//printf("Largo de depths\n");
	size_t data_size = x.size();
	//printf("%zu\n", data_size);
	params = { 0.001, 0.001, 0.001, 0.001 };
	
	double* dev_x;
	double* dev_y;
	cudaMalloc((void**)&dev_x,data_size * sizeof(double));
	cudaMalloc((void**)&dev_y,data_size * sizeof(double));
	//host to device
	cudaMemcpy(dev_x, x.data(), data_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, y.data(), data_size * sizeof(double), cudaMemcpyHostToDevice);

	//parametros del dispositivo
	double* dev_params;
	cudaMalloc((void**)&dev_params, params.size() * sizeof(double));
	cudaMemcpy(dev_params, params.data(), params.size() * sizeof(double), cudaMemcpyHostToDevice);

	//error de grafica desviacion en el codigo
	double* dev_error;
	cudaMalloc((void**)&dev_error, data_size * sizeof(double));
	
	//launch the kernel
	int threadsP = 256;
	int blocks = (data_size + threadsP - 1) / threadsP;
	
	// Iniciar el timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);



	const double epsilon = 1e-8; // Tolerancia de convergencia
	const double stepSize = 0.001; // Tamaño de paso para la actualización de los parámetros
	const size_t maxIterations = 1000; // Número máximo de iteraciones

	double constante_error = 0.0;
	double prev_error = 0.0;
	size_t iteration = 0;
	
	double* h_vector = (double*)malloc(data_size * sizeof(double));  // Vector en el host
	double contador_error;

	while (iteration < maxIterations) {
		errorFunctionCUDA << <blocks, threadsP >> > (dev_params, dev_x, dev_y, dev_error, data_size);
		cudaMemcpy(&constante_error, dev_error, sizeof(double), cudaMemcpyDeviceToHost);
		//prev_error = constante_error;
		//printf("The value of the double is: %lf\n", prev_error);
		// Salir si se alcanza la convergencia
		if (fabs(constante_error - prev_error) < epsilon) {
			break;
		}
		prev_error = constante_error;
		//update of the parameters information
		for (int i = 0; i < params.size(); i++){
			double prevParam = params[i];
			params[i] += stepSize;
			cudaMemcpy(dev_params, params.data(), params.size() * sizeof(double), cudaMemcpyHostToDevice);
			errorFunctionCUDA << <blocks, threadsP >> > (dev_params, dev_x, dev_y, dev_error, data_size);
			cudaMemcpy(&constante_error, dev_error, sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_vector, dev_error, data_size * sizeof(double), cudaMemcpyDeviceToHost);

			//for (int i = 0; i < data_size; i++) {
			//	printf("%lf ", h_vector[i]);
			
			//}
			//printf("\n");

			if (constante_error > prev_error){
				params[i] = prevParam;
			}
			else{
				prev_error = constante_error;
			}
		}
		
		

		iteration = iteration + 1;
	}
	// Detener el timer
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	// Calcular el tiempo transcurrido
	float milliseconds = 0.0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaMemcpy(params.data(), dev_params, params.size() * sizeof(double), cudaMemcpyDeviceToHost);
	// Imprimir el tiempo transcurrido
	printf("Tiempo transcurrido: %.4f ms\n", milliseconds);

	
	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_params);
	cudaFree(dev_error);


	// Liberar los eventos CUDA
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

#define BLOCK_SIZE 256

__global__ void errorFunctionCUDA2(const double* params, const double* x, const double* y, double* error, size_t data_size) {
	__shared__ double s_error[BLOCK_SIZE];
	int locs = threadIdx.x + blockIdx.x * blockDim.x;
	int local_tid = threadIdx.x;
	

	if (locs < data_size) {
		double b_1 = params[0];
		double b_2 = params[1];
		double b_3 = params[2];
		double b_4 = params[3];
		double result = b_1 * (1 - exp(b_2 * x[locs])) + (b_3 * exp(-b_4 * x[locs])) - y[locs];
		error[locs] = result * result;
		//error[locs] = result * result;
		//printf("%f", error[locs]);
		//atomicAdd(&error[0], result * result);
	}
	
	//__syncthreads();
	//for i in locs
	// Realizar reducción utilizando la memoria compartida
	//for (int stride = 1; stride < data_size * int(data_size/BLOCK_SIZE); stride *= 2) {
		//int index = (locs + 1) * stride * 2 - 1;
		//if (locs >= stride && locs < (data_size * int(data_size / BLOCK_SIZE))) {
			//std::cout << stride << endl;
		//error[locs] += error[local_tid - stride];
		//}
		
	//	__syncthreads();
	//}
	//if
	//error += error [i - 1]
	// El resultado final se encuentra en s_error[BLOCK_SIZE - 1] del último hilo del bloque
	
	
}

void curveFittingCUDA2(const std::vector<double>& x, const std::vector<double>& y, std::vector<double>& params) {
	params = { 0.001, 0.001, 0.001, 0.001 };
	size_t data_size = x.size();
	// Configuración del ajuste
	const double epsilon = 1e-8; // Tolerancia de convergencia
	const double stepSize = 0.001; // Tamaño de paso para la actualización de los parámetros
	const size_t maxIterations = 1000; // Número máximo de iteraciones
	// Ajuste de curva iterativo
	double* dev_x;
	double* dev_y;
	size_t iteration = 0;
	cudaMalloc((void**)&dev_x, data_size * sizeof(double));
	cudaMalloc((void**)&dev_y, data_size * sizeof(double));
	//host to device
	cudaMemcpy(dev_x, x.data(), data_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, y.data(), data_size * sizeof(double), cudaMemcpyHostToDevice);
	double* dev_params;
	cudaMalloc((void**)&dev_params, params.size() * sizeof(double));
	cudaMemcpy(dev_params, params.data(), params.size()*sizeof(double), cudaMemcpyHostToDevice);
	std::vector<double> errors(x.size(), 0);
	double* dev_errors;
	cudaMalloc((void**)&dev_errors, data_size * sizeof(double));
	cudaMemcpy(dev_errors, errors.data(), data_size * sizeof(double), cudaMemcpyHostToDevice);
	//launch the kernel
	int threadsP = 256;
	int blocks = (data_size + threadsP - 1) / threadsP;
	errorFunctionCUDA2 << <blocks, threadsP >> > (dev_params, dev_x, dev_y, dev_errors, data_size);
	cudaMemcpy(errors.data(), dev_errors, errors.size() * sizeof(double), cudaMemcpyDeviceToHost);
	
	double cont = 0;
	double prevError = std::accumulate(errors.begin(), errors.end(), 0.000);
	//for (int i = 0; i < errors.size(); i++){
		//printf("%f ", errors.size());
	//	cont += errors[i];
	//	std::cout << errors[i] << std::endl;
	//}
	//std::cout << "count " << cont << std::endl;
	std::cout << "error_cont " << prevError << std::endl;
	while (iteration < maxIterations) {
		for (int i = 0; i < params.size(); i++) {
			double prevParam = params[i];
			params[i] += stepSize;
			cudaMemcpy(dev_params, params.data(), params.size() * sizeof(double), cudaMemcpyHostToDevice);
			errorFunctionCUDA2 << <blocks, threadsP >> > (dev_params, dev_x, dev_y, dev_errors, data_size);
			cudaMemcpy(errors.data(), dev_errors, errors.size() * sizeof(double), cudaMemcpyDeviceToHost);
			double currError = std::accumulate(errors.begin(), errors.end(), 0.000);
			if (currError > prevError) {
				// Si el error empeora, revertir la actualización
				params[i] = prevParam;
			}
			else {
				// Si el error mejora, continuar actualizando
				prevError = currError;
			}
		}
		// Verificar si ha convergido
		if (prevError < epsilon) {
			break;
		}
		iteration = iteration + 1;
	}
	std::cout << "param 0: " << params[0] << " params 1: " << params[1] << " params 2: " << params[2] << " params 3: " << params[3] << std::endl;
	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_params);
	cudaFree(dev_errors);
}


//esta secciion representa la funcion optimizada del codigo 
//
__global__ void errorFunctionCUDA3(const double* params_r, const double* params_g, const double* params_b,
	const double* x, const double* y_r, const double* y_g, const double* y_b,
	double* error_r, double* error_g, double* error_b, size_t data_size) {
	int locs = threadIdx.x + blockIdx.x * blockDim.x;
	int local_tid = threadIdx.x;
	if (locs < data_size) {
		//r
		double b_1_r = params_r[0];
		double b_2_r = params_r[1];
		double b_3_r = params_r[2];
		double b_4_r = params_r[3];
		//g
		double b_1_g = params_g[0];
		double b_2_g = params_g[1];
		double b_3_g = params_g[2];
		double b_4_g = params_g[3];
		//b
		double b_1_b = params_b[0];
		double b_2_b = params_b[1];
		double b_3_b = params_b[2];
		double b_4_b = params_b[3];
		//results
		//r
		double result_r = b_1_r * (1 - exp(b_2_r * x[locs])) + (b_3_r * exp(-b_4_r * x[locs])) - y_r[locs];
		error_r[locs] = result_r * result_r;
		//g
		double result_g = b_1_g * (1 - exp(b_2_g * x[locs])) + (b_3_g * exp(-b_4_g * x[locs])) - y_g[locs];
		error_g[locs] = result_g * result_g;
		//r
		double result_b = b_1_b * (1 - exp(b_2_b * x[locs])) + (b_3_b * exp(-b_4_b * x[locs])) - y_b[locs];
		error_b[locs] = result_b * result_b;
	}
}


//vercion optimizada del codigo, en este caso el codigo funciona con los3 canales al mismo tiempo por lo que
//los datos procesados proceden a hacerlo de manera paralela entre los tres canales, el error se calcula en 
//el GPU y los valores proceden a sumare en el mismo GPU utilizando comandos que permiten hacerlo de manera instantanea
void curveFittingCUDA3(const std::vector<double>& x, 
	const std::vector<double>& y_r,	const std::vector<double>& y_g, const std::vector<double>& y_b,
	std::vector<double>& params_r, std::vector<double>& params_g, std::vector<double>& params_b) {

	params_r = { 0.001, 0.001, 0.001, 0.001 };
	params_g = { 0.001, 0.001, 0.001, 0.001 };
	params_b = { 0.001, 0.001, 0.001, 0.001 };
	size_t data_size = x.size();
	// Configuración del ajuste
	const double epsilon = 1e-8; // Tolerancia de convergencia
	const double stepSize = 0.001; // Tamaño de paso para la actualización de los parámetros
	const size_t maxIterations = 1000; // Número máximo de iteraciones
	double* dev_x;
	double* dev_y_r;
	double* dev_y_g;
	double* dev_y_b;
	size_t iteration = 0;
	cudaMalloc((void**)&dev_x, data_size * sizeof(double));
	cudaMalloc((void**)&dev_y_r, data_size * sizeof(double));
	cudaMalloc((void**)&dev_y_g, data_size * sizeof(double));
	cudaMalloc((void**)&dev_y_b, data_size * sizeof(double));

	//host to device
	cudaMemcpy(dev_x, x.data(), data_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y_r, y_r.data(), data_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y_g, y_g.data(), data_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y_b, y_b.data(), data_size * sizeof(double), cudaMemcpyHostToDevice);
	double* dev_params_r;
	double* dev_params_g;
	double* dev_params_b;
	cudaMalloc((void**)&dev_params_r, params_r.size() * sizeof(double));
	cudaMalloc((void**)&dev_params_g, params_g.size() * sizeof(double));
	cudaMalloc((void**)&dev_params_b, params_b.size() * sizeof(double));
	cudaMemcpy(dev_params_r, params_r.data(), params_r.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_params_g, params_g.data(), params_g.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_params_b, params_b.data(), params_b.size() * sizeof(double), cudaMemcpyHostToDevice);

	std::vector<double> errors_r(x.size(), 0);
	std::vector<double> errors_g(x.size(), 0);
	std::vector<double> errors_b(x.size(), 0);
	double* dev_errors_r;
	double* dev_errors_g;
	double* dev_errors_b;
	cudaMalloc((void**)&dev_errors_r, data_size * sizeof(double));
	cudaMalloc((void**)&dev_errors_g, data_size * sizeof(double));
	cudaMalloc((void**)&dev_errors_b, data_size * sizeof(double));
	cudaMemcpy(dev_errors_r, errors_r.data(), data_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_errors_g, errors_g.data(), data_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_errors_b, errors_b.data(), data_size * sizeof(double), cudaMemcpyHostToDevice);
	//launch the kernel
	int threadsP = 256;
	int blocks = (data_size + threadsP - 1) / threadsP;
	errorFunctionCUDA3 << <blocks, threadsP >> > (dev_params_r, dev_params_g, dev_params_b,
		dev_x, dev_y_r, dev_y_g, dev_y_b, 
		dev_errors_r, dev_errors_g, dev_errors_b, data_size);

	cudaMemcpy(errors_r.data(), dev_errors_r, errors_r.size() * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(errors_g.data(), dev_errors_g, errors_g.size() * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(errors_b.data(), dev_errors_b, errors_b.size() * sizeof(double), cudaMemcpyDeviceToHost);
	double prevError_r = std::accumulate(errors_r.begin(), errors_r.end(), 0.0);
	double prevError_g = std::accumulate(errors_g.begin(), errors_g.end(), 0.0);
	double prevError_b = std::accumulate(errors_b.begin(), errors_b.end(), 0.0);

	//std::cout << "prevError_r: " << prevError_r << std::endl;
	//std::cout << "prevError_g: " << prevError_g << std::endl;
	//std::cout << "prevError_b: " << prevError_b << std::endl;

	while (iteration < maxIterations) {
		for (int i = 0; i < params_r.size(); i++) {
			double prevParam_r = params_r[i];
			double prevParam_g = params_g[i];
			double prevParam_b = params_b[i];

			params_r[i] += stepSize;
			params_g[i] += stepSize;
			params_b[i] += stepSize;

			cudaMemcpy(dev_params_r, params_r.data(), params_r.size() * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_params_g, params_g.data(), params_g.size() * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_params_b, params_b.data(), params_b.size() * sizeof(double), cudaMemcpyHostToDevice);
			errorFunctionCUDA3 << <blocks, threadsP >> > (dev_params_r, dev_params_g, dev_params_b,
				dev_x, dev_y_r, dev_y_g, dev_y_b,
				dev_errors_r, dev_errors_g, dev_errors_b, data_size);
			cudaMemcpy(errors_r.data(), dev_errors_r, errors_r.size() * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(errors_g.data(), dev_errors_g, errors_g.size() * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(errors_b.data(), dev_errors_b, errors_b.size() * sizeof(double), cudaMemcpyDeviceToHost);
			double currError_r = std::accumulate(errors_r.begin(), errors_r.end(), 0.0);
			double currError_g = std::accumulate(errors_g.begin(), errors_g.end(), 0.0);
			double currError_b = std::accumulate(errors_b.begin(), errors_b.end(), 0.0);
			//r
			if (currError_r > prevError_r){
				// Si el error empeora, revertir la actualización
				params_r[i] = prevParam_r;
			}
			else{
				// Si el error mejora, continuar actualizando
				prevError_r = currError_r;
			}
			//g
			if (currError_g > prevError_g){
				// Si el error empeora, revertir la actualización
				params_g[i] = prevParam_g;
			}
			else{
				// Si el error mejora, continuar actualizando
				prevError_g = currError_g;
			}
			//b
			if (currError_b > prevError_b){
				// Si el error empeora, revertir la actualización
				params_b[i] = prevParam_b;
			}
			else{
				// Si el error mejora, continuar actualizando
				prevError_b = currError_b;
			}
		}

		iteration = iteration + 1;
	}

	std::cout << "param_r 0: " << params_r[0] << " params_r 1: " << params_r[1] << " params_r 2: " << params_r[2] << " params_r 3: " << params_r[3] << std::endl;
	std::cout << "param_g 0: " << params_g[0] << " params_r 1: " << params_g[1] << " params_g 2: " << params_g[2] << " params_g 3: " << params_g[3] << std::endl;
	std::cout << "param_b 0: " << params_b[0] << " params_r 1: " << params_b[1] << " params_b 2: " << params_b[2] << " params_b 3: " << params_b[3] << std::endl;


	cudaFree(dev_x);
	
	cudaFree(dev_y_r);
	cudaFree(dev_y_g);
	cudaFree(dev_y_b);

	cudaFree(dev_params_r);
	cudaFree(dev_params_g);
	cudaFree(dev_params_b);

	cudaFree(dev_errors_r);
	cudaFree(dev_errors_g);
	cudaFree(dev_errors_b);
}

