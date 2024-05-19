//
// Created by Roberto Sala on 12/09/23.
//

#include "Adam.h"
#include <random>
#include <cmath>


std::vector<int> Adam::create_batch() {

    std::vector<int> indices;

    for (std::size_t i = 0; i < dim_batch; ++i)
        indices.push_back(distribution(generator));

    return indices;

}


double Adam::evaluate_batch(const Point &parameters, const std::vector<int> & batch) const {

    double value = 0.0;

    for (const auto i : batch)
        value += f.evaluate(observations[i], parameters);

    return value / batch.size();

}


double Adam::evaluate_partial_derivative_batch(std::size_t j, const Point &parameters, const std::vector<int> & batch) const {

    double value = 0.0;

    for(const auto i : batch) {
        value += f.evaluate_partial_derivative(j, observations[i], parameters);
    }

    return value / batch.size();

}


void Adam::set_f(const FunctionRn &f_) {

    f = f_;

}


void Adam::set_observations(const std::vector<Point> &observations_) {

    observations = observations_;

}


void Adam::set_dim_batch(unsigned int dim_batch_) {

    dim_batch = dim_batch_;

}


void Adam::set_tolerance(double tolerance_) {

    tolerance = tolerance_;

}


void Adam::set_max_iterations(unsigned int max_iterations_) {

    max_iterations = max_iterations_;

}


void Adam::set_inf_limits(const std::vector<double> &inf_limits_) {

    inf_limits = inf_limits_;

}


void Adam::set_sup_limits(const std::vector<double> &sup_limits_) {

    sup_limits = sup_limits_;

}


const FunctionRn & Adam::get_f() const {

    return f;

}


const std::vector<Point> & Adam::get_observations() const {

    return observations;

}


unsigned int Adam::get_dim_batch() const {

    return dim_batch;

}


double Adam::get_tolerance() const {

    return tolerance;

}


unsigned int Adam::get_max_iterations() const {

    return max_iterations;

}


const std::vector<double> & Adam::get_inf_limits() const {

    return inf_limits;

}


const std::vector<double> & Adam::get_sup_limits() const {

    return sup_limits;

}


Point Adam::solve(const Point &initial_parameters) {

    const size_t sizeP = initial_parameters.get_dimension();

    std::vector<double> m(sizeP,0.0),v(sizeP,0.0);

    Point paramPost(initial_parameters);

    bool converged = false;


    for (size_t t = 1; t <= max_iterations && !converged; ++t) {
        Point paramPre(paramPost);

        std::vector<int> batch(dim_batch,0);
        batch=create_batch();

        double gradient,mHat,vHat;
        for (size_t i = 0; i < sizeP; ++i) {
            gradient=evaluate_partial_derivative_batch(i,paramPre,batch);
            m[i]=gamma1*m[i]+(1-gamma1)*gradient;
            v[i]=gamma2*v[i]+(1-gamma2)*pow(gradient,2);
            mHat=m[i]/(1- pow(gamma1,t));
            vHat=v[i]/(1- pow(gamma2,t));
            paramPost.set_coordinate(i, paramPre.get_coordinate(i) - alpha*mHat/(sqrt(vHat)+epsilon) );

            if(paramPost.get_coordinate(i)<inf_limits[i]){
                paramPost.set_coordinate(i,inf_limits[i]);
            }
            else if(paramPost.get_coordinate(i)>sup_limits[i]) {
                paramPost.set_coordinate(i,sup_limits[i]);
            }
        }

        if (paramPost.distance(paramPre) < tolerance || std::abs(evaluate_batch(paramPost, batch)-evaluate_batch(paramPre,batch)) < tolerance){
            converged=true;
            std::cout << "\nIter: " << t << std::endl;
        }

    }
    return paramPost;

}
