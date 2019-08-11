#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0,0,0,0;

	// check the validity of the following inputs:
	//  * the estimation vector size should not be zero
	//  * the estimation vector size should equal ground truth vector size
	if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
		cout << "CalculateRMSE[ERR]: Invalid estimation or ground_truth data" << endl;
		return rmse;
	}

	for (unsigned int i=0; i < estimations.size(); ++i) {

		VectorXd residual = estimations[i] - ground_truth[i];
		// coefficient-wise multiplication
		residual = residual.array()*residual.array();
		rmse += residual;
	}

	// calculate the mean
	rmse = rmse/estimations.size();

	cout << rmse << endl;

	// calculate the squared root
	rmse = rmse.array().sqrt();

	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

	MatrixXd Hj(3,4);
	// recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	float sqrt_px_py = sqrt(px*px + py*py);
	float sq_px_py = (px*px + py*py);
	float cube_sqrt_px_py = pow(sqrt_px_py, 3);

	// check division by zero
	if( (sqrt_px_py == 0) || (sq_px_py == 0) || (cube_sqrt_px_py == 0))
	{
	  cout<<"CalculateJacobian[ERR]: Divide by Zero" << endl;
	  goto done;
	}
	// compute the Jacobian matrix
	Hj(0,0) = px/sqrt_px_py;
	Hj(0,1) = py/sqrt_px_py;
	Hj(0,2) = 0;
	Hj(0,3) = 0;

	Hj(1,0) = -py/sq_px_py;
	Hj(1,1) = px/sq_px_py;
	Hj(1,2) = 0;
	Hj(1,3) = 0;

	Hj(2,0) = (py*(vx*py - vy*px))/cube_sqrt_px_py;
	Hj(2,1) = (px*(vy*px - vx*py))/cube_sqrt_px_py;
	Hj(2,2) = px/sqrt_px_py;
	Hj(2,3) = py/sqrt_px_py;
done:
 	return Hj;
}
