#include "object.h"
#include "sphere.h"
#include "triangle.h"

#include <vector>
#include <iostream>
#include <unordered_map>

#include "util/device-vec-ops.h"

using std::vector;
using std::unordered_map;

namespace CGL {
namespace SceneObjects {

// Mesh object //

Mesh::Mesh(const HalfedgeMesh &mesh, BSDF *bsdf) {

  unordered_map<const Vertex *, int> vertexLabels;
  vector<const Vertex *> verts;

  size_t vertexI = 0;
  for (VertexCIter it = mesh.verticesBegin(); it != mesh.verticesEnd(); it++) {
    const Vertex *v = &*it;
    verts.push_back(v);
    vertexLabels[v] = vertexI;
    vertexI++;
  }
  num_vertices = vertexI;
  positions = new Vector3D[vertexI];
  normals = new Vector3D[vertexI];
  std::cout << mesh.nFaces() << std::endl;
  std::cout << num_vertices << std::endl;
  for (int i = 0; i < vertexI; i++) {
    positions[i] = verts[i]->position;
    normals[i]   = verts[i]->normal;
    if (num_vertices == 4) {
      std::cout << positions[i] << " " << normals[i] << std::endl;
    }
  }
  for (FaceCIter f = mesh.facesBegin(); f != mesh.facesEnd(); f++) {
    HalfedgeCIter h = f->halfedge();
    indices.push_back(vertexLabels[&*h->vertex()]);
    indices.push_back(vertexLabels[&*h->next()->vertex()]);
    indices.push_back(vertexLabels[&*h->next()->next()->vertex()]);
  }

  this->bsdf = bsdf;

}

BSDF *Mesh::get_bsdf() const {
  return bsdf;
}

// Sphere object //

SphereObject::SphereObject(const Vector3D o, double r, BSDF *bsdf) {

  this->o = o;
  this->r = r;
  this->bsdf = bsdf;

}

BSDF *SphereObject::get_bsdf() const {
  return bsdf;
}


} // namespace SceneObjects
} // namespace CGL
