#include "bsdf.h"
#include "bsdf.h"
#include "bsdf.h"

#include "application/visual_debugger.h"

#include <algorithm>
#include <iostream>
#include <utility>


using std::max;
using std::min;
using std::swap;

namespace CGL {

/**
 * This function creates a object space (basis vectors) from the normal vector
 */
void make_coord_space(Matrix3x3 &o2w, const double3&& n) {

}

/**
 * Evaluate diffuse lambertian BSDF.
 * Given incident light direction wi and outgoing light direction wo. Note
 * that both wi and wo are defined in the local coordinate system at the
 * point of intersection.
 * \param wo outgoing light direction in local space of point of intersection
 * \param wi incident light direction in local space of point of intersection
 * \return reflectance in the given incident/outgoing directions
 */
double3 DiffuseBSDF::f(const double3& wo, const double3& wi) {
  // TODO (Part 3.1):
  // This function takes in both wo and wi and returns the evaluation of
  // the BSDF for those two directions.


  return double3(1.0);

}

/**
 * Evalutate diffuse lambertian BSDF.
 */
double3 DiffuseBSDF::sample_f(const double3& wo, double3 *wi, double *pdf) {
  // TODO (Part 3.1):
  // This function takes in only wo and provides pointers for wi and pdf,
  // which should be assigned by this function.
  // After sampling a value for wi, it returns the evaluation of the BSDF
  // at (wo, *wi).
  // You can use the `f` function. The reference solution only takes two lines.


  return double3(1.0);

}

/**
 * Evalutate Emission BSDF (Light Source)
 */
double3 EmissionBSDF::f(const double3& wo, const double3& wi) {
  return double3();
}

/**
 * Evalutate Emission BSDF (Light Source)
 */
double3 EmissionBSDF::sample_f(const double3& wo, double3 *wi, double *pdf) {
  *pdf = 1.0 / PI;
  *wi = sampler.get_sample(pdf);
  return double3();
}


} // namespace CGL
