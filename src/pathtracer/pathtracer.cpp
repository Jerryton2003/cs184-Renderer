#include "pathtracer.h"

#include "scene/light.h"
#include "scene/sphere.h"
#include "scene/triangle.h"

using namespace CGL::SceneObjects;

namespace CGL {
PathTracer::PathTracer() {
  gridSampler = new UniformGridSampler2D();
  hemisphereSampler = new UniformHemisphereSampler3D();

  tm_gamma = 2.2f;
  tm_level = 1.0f;
  tm_key = 0.18;
  tm_wht = 5.0f;
}

PathTracer::~PathTracer() {
  delete gridSampler;
  delete hemisphereSampler;
}

void PathTracer::set_frame_size(size_t width, size_t height) {
  sampleBuffer.resize(width, height);
  sampleCountBuffer.resize(width * height);
}

void PathTracer::clear() {
  bvh = NULL;
  scene = NULL;
  camera = NULL;
  sampleBuffer.clear();
  sampleCountBuffer.clear();
  sampleBuffer.resize(0, 0);
  sampleCountBuffer.resize(0, 0);
}

void PathTracer::write_to_framebuffer(ImageBuffer&framebuffer,
                                      size_t x0,
                                      size_t y0,
                                      size_t x1,
                                      size_t y1) {
  sampleBuffer.toColor(framebuffer, x0, y0, x1, y1);
}

Vector3D
PathTracer::estimate_direct_lighting_hemisphere(const Ray&r,
                                                const Intersection&isect) {
  // Estimate the lighting from this intersection coming directly from a light.
  // For this function, sample uniformly in a hemisphere.

  // Note: When comparing Cornel Box (CBxxx.dae) results to importance sampling, you may find the "glow" around the light source is gone.
  // This is totally fine: the area lights in importance sampling has directionality, however in hemisphere sampling we don't model this behaviour.

  // make a coordinate system for a hit point
  // with N aligned with the Z direction.
  Matrix3x3 o2w;
  make_coord_space(o2w, isect.n);
  Matrix3x3 w2o = o2w.T();

  // w_out points towards the source of the ray (e.g.,
  // toward the camera if this is a primary ray)
  const Vector3D hit_p = r.o + r.d * isect.t;
  const Vector3D w_out = w2o * (-r.d);

  // This is the same number of total samples as
  // estimate_direct_lighting_importance (outside of delta lights). We keep the
  // same number of samples for clarity of comparison.
  int num_samples = scene->lights.size() * ns_area_light;
  Vector3D L_out;

  // UPDATE `est_radiance_global_illumination` to return direct lighting instead of normal shading
  auto bsdf = isect.bsdf;
  for (int i = 0; i < num_samples; i++) {
    Vector3D sample = hemisphereSampler->get_sample();
    Vector3D w_in = o2w * sample;
    float pdf = 1.0 / (2 * PI);
    Ray shadow_ray(hit_p + EPS_F * isect.n, w_in);
    Intersection shadow_isect;
    if (bvh->intersect(shadow_ray, &shadow_isect) && !shadow_isect.bsdf->is_delta())
      L_out += bsdf->f(w_out, sample) * dot(isect.n, w_in) * shadow_isect.bsdf->get_emission() / pdf;
  }
  return L_out / num_samples;
}

Vector3D
PathTracer::estimate_direct_lighting_importance(const Ray&r,
                                                const Intersection&isect) {
  // Estimate the lighting from this intersection coming directly from a light.
  // To implement importance sampling, sample only from lights, not uniformly in
  // a hemisphere.

  // make a coordinate system for a hit point
  // with N aligned with the Z direction.
  Matrix3x3 o2w;
  make_coord_space(o2w, isect.n);
  Matrix3x3 w2o = o2w.T();

  // w_out points towards the source of the ray (e.g.,
  // toward the camera if this is a primary ray)
  const Vector3D hit_p = r.o + r.d * isect.t;
  const Vector3D w_out = w2o * (-r.d);
  Vector3D L_out;
  auto bsdf = isect.bsdf;
  int total_samples = 0;
  for (int i = 0; i < scene->lights.size(); i++) {
    if (scene->lights[i]->is_delta_light()) continue;
    total_samples += ns_area_light;
  }
  Vector3D L_delta;
  for (auto light : scene->lights) {
    if (light->is_delta_light()) {
      Vector3D sample, wi;
      double dist_to_light, pdf;
      sample = light->sample_L(hit_p, &wi, &dist_to_light, &pdf);
      Vector3D w_in = w2o * wi;
      Ray shadow_ray(hit_p + EPS_F * wi, wi);
      Intersection shadow_isect;
      if (bvh->intersect(shadow_ray, &shadow_isect)) continue;
      L_delta += bsdf->f(w_out, w_in) * dot(isect.n, wi) * sample / pdf;
      continue;
    }
    int num_samples = ns_area_light;
    for (int i = 0; i < num_samples; i++) {
      Vector3D sample, wi;
      double dist_to_light, pdf;
      sample = light->sample_L(hit_p, &wi, &dist_to_light, &pdf);
      Vector3D w_in = w2o * wi;
      Ray shadow_ray(hit_p + EPS_F * wi, wi);
      Intersection shadow_isect;
      if (dist_to_light < INF_D) {
        if (!bvh->intersect(shadow_ray, &shadow_isect) || shadow_isect.t < dist_to_light - 2 *
          EPS_F)
          continue;
        L_out += bsdf->f(w_out, w_in) * dot(isect.n, wi) * sample / pdf;
      }
      else {
        if (bvh->intersect(shadow_ray, &shadow_isect)) continue;
        L_out += bsdf->f(w_out, w_in) * dot(isect.n, wi) * sample / pdf;
      }
    }
  }
  return L_out / total_samples + L_delta;
}

Vector3D PathTracer::zero_bounce_radiance(const Ray&r,
                                          const Intersection&isect) {
  // Returns the light that results from no bounces of light
  if (isect.bsdf == nullptr || isect.bsdf->is_delta())
    return {};
  auto emittance = isect.bsdf->get_emission();
  return emittance;
}

Vector3D PathTracer::one_bounce_radiance(const Ray&r,
                                         const Intersection&isect) {
  // Returns either the direct illumination by hemisphere or importance sampling
  // depending on `direct_hemisphere_sample`
  if (direct_hemisphere_sample) {
    return estimate_direct_lighting_hemisphere(r, isect);
  }
  return estimate_direct_lighting_importance(r, isect);
}

Vector3D PathTracer::at_least_one_bounce_radiance(const Ray&r,
                                                  const Intersection&isect) {
  Matrix3x3 o2w;
  make_coord_space(o2w, isect.n);
  Matrix3x3 w2o = o2w.T();

  Vector3D hit_p = r.o + r.d * isect.t;
  Vector3D w_out = w2o * (-r.d);

  Vector3D L_out(0, 0, 0);

  // Returns the one bounce radiance + radiance from extra bounces at this point.
  // Should be called recursively to simulate extra bounces.

  if (isect.bsdf == nullptr)
    return {};
  if (isect.bsdf->get_emission().norm() > 0)
    return {};

  // one bounce
  Vector3D L_onebounce = one_bounce_radiance(r, isect);
  if (r.depth == max_ray_depth - 1 && !isAccumBounces)
    return L_onebounce;
  if (isAccumBounces)
    L_out += L_onebounce;
  if (r.depth == max_ray_depth - 1)
    return L_out;
  bool count_rr = false;
  if (coin_flip(0.3))
  return L_out;
  count_rr = true;
  // sample a direction and trace a ray
  double pdf;
  Vector3D w_in;
  Vector3D sample = isect.bsdf->sample_f(w_out, &w_in, &pdf);
  Vector3D wi = o2w * w_in;
  Ray new_ray(hit_p + EPS_F * wi, wi);
  new_ray.depth = r.depth + 1;
  Intersection new_isect;
  if (!bvh->intersect(new_ray, &new_isect))
    return L_out;
  L_out += sample * dot(isect.n, wi) * at_least_one_bounce_radiance(new_ray, new_isect) / pdf;
  if (count_rr)
  L_out /= 0.7;
  return L_out;
}

Vector3D PathTracer::est_radiance_global_illumination(const Ray&r) {
  Intersection isect;
  Vector3D L_out;

  // You will extend this in assignment 3-2.
  // If no intersection occurs, we simply return black.
  // This changes if you implement hemispherical lighting for extra credit.

  // The following line of code returns a debug color depending
  // on whether ray intersection with triangles or spheres has
  // been implemented.
  //
  // REMOVE THIS LINE when you are ready to begin Part 3.

  if (!bvh->intersect(r, &isect))
    return envLight ? envLight->sample_dir(r) : L_out;
  if (isAccumBounces) {
    if (max_ray_depth > 1)
      L_out = zero_bounce_radiance(r, isect) + at_least_one_bounce_radiance(r, isect);
    else if (max_ray_depth == 1)
      L_out = zero_bounce_radiance(r, isect) + one_bounce_radiance(r, isect);
    else
      L_out = zero_bounce_radiance(r, isect);
  }
  else if (max_ray_depth == 0)
    L_out = zero_bounce_radiance(r, isect);
  else if (max_ray_depth == 1)
    L_out = one_bounce_radiance(r, isect);
  else
    L_out = at_least_one_bounce_radiance(r, isect);
  return L_out;
}

void PathTracer::raytrace_pixel(size_t x, size_t y) {
  // Make a loop that generates num_samples camera rays and traces them
  // through the scene. Return the average Vector3D.
  // You should call est_radiance_global_illumination in this function.

  // TODO (Part 5):
  // Modify your implementation to include adaptive sampling.
  // Use the command line parameters "samplesPerBatch" and "maxTolerance"
  int num_samples = ns_aa; // total samples to evaluate
  Vector2D origin = Vector2D(x, y); // bottom left corner of the pixel
  Vector3D pixel_sum = Vector3D(0, 0, 0);
  double s1 = 0, s2 = 0;
  int total_samples = num_samples;
  for (int i = 0; i < num_samples; i++) {
    Vector2D sample = gridSampler->get_sample();
    Vector2D sample_pos = origin + sample;
    Ray r = camera->generate_ray(sample_pos.x / sampleBuffer.w, sample_pos.y / sampleBuffer.h);
    Vector3D radiance_sample = est_radiance_global_illumination(r);
    pixel_sum += radiance_sample;
    double illum = radiance_sample.illum();
    s1 += illum;
    s2 += illum * illum;
    if ((i + 1) % samplesPerBatch == 0) {
      double mu = s1 / (i + 1);
      double sigma_sqr = s2 / (i + 1) - mu * mu;
      double I = 1.96 * sqrt(sigma_sqr / (i + 1));
      if (I <= maxTolerance * mu) {
        total_samples = i + 1;
        break;
      }
    }
  }
  sampleBuffer.update_pixel(pixel_sum / total_samples, x, y);
  sampleCountBuffer[x + y * sampleBuffer.w] = total_samples;
}

void PathTracer::autofocus(Vector2D loc) {
  Ray r = camera->generate_ray(loc.x / sampleBuffer.w, loc.y / sampleBuffer.h);
  Intersection isect;

  bvh->intersect(r, &isect);

  camera->focalDistance = isect.t;
}
} // namespace CGL
