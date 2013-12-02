#include <reflection.h>
#include <mesh.h>
#include <geometry.h>
#include <assert.h>
#include <math.h>

__device__ double calcIntersectionAngle(const Ray ray, double *reflectionAngle){
  // Calc intesection angle with z-plane
  double nominator = abs(ray.dir.z);
  double denominator = sqrt((ray.dir.x * ray.dir.x) + (ray.dir.y * ray.dir.y) + (ray.dir.z * ray.dir.z));
  if(denominator != 0.0){
    double radian = acos(nominator / denominator);
    *reflectionAngle = ((180. / M_PI) * radian);
    return 0;
  }
  return 1;
}

__device__ int calcPlaneIntersectionPoint(const Ray reflectionRay, const ReflectionPlane reflectionPlane, const Mesh &mesh, Point *intersectionPoint){
  // Assume that mesh is on x/y axis and parallel to x/y axis
  double planeZ = 0.0;
  if(reflectionPlane == TOP_REFLECTION){
    // Reflection on TOP plane
    planeZ = mesh.thickness * mesh.numberOfLevels;
  }
  double denominator = reflectionRay.dir.z;  
  if (denominator != 0.0){
    double nominator = planeZ - reflectionRay.p.z;
    double length = nominator / denominator;
    if(length > 0){
      intersectionPoint->x = reflectionRay.p.x + length * reflectionRay.dir.x;
      intersectionPoint->y = reflectionRay.p.y + length * reflectionRay.dir.y;
      intersectionPoint->z = reflectionRay.p.z + length * reflectionRay.dir.z;
      return 0;
    }

  }
  return 1;
}


__device__ Ray generateReflectionRay(const Point startPoint, Point endPoint,  const int reflectionsLeft, const ReflectionPlane reflectionPlane, const Mesh &mesh){
  float mirrorPlaneZ = 0;
  if(reflectionsLeft % 2 == 0){
    // Even reflectionCount is postponement
    endPoint.z = endPoint.z + reflectionPlane * (reflectionsLeft * mesh.thickness * mesh.numberOfLevels); 
  }
  else {
    // Odd reflectionsCount is reflection

    if(reflectionPlane == TOP_REFLECTION){
      mirrorPlaneZ = ceil(reflectionsLeft/(double)2) * mesh.thickness * mesh.numberOfLevels;
    }
    else{
      mirrorPlaneZ = floor(reflectionsLeft/(double)2) * mesh.thickness * mesh.numberOfLevels;
    }

    endPoint.z = reflectionPlane * abs(( mirrorPlaneZ + mirrorPlaneZ - endPoint.z));
    
  }
  return generateRay(startPoint, endPoint);
}

__device__ int calcNextReflection(const Point startPoint, const Point endPoint, const unsigned reflectionsLeft, const ReflectionPlane reflectionPlane, Point *reflectionPoint, double *reflectionAngle, const Mesh &mesh){
  Ray reflectionRay = generateReflectionRay(startPoint, endPoint, reflectionsLeft, reflectionPlane, mesh);
  if(calcPlaneIntersectionPoint(reflectionRay, reflectionPlane, mesh, reflectionPoint)) return 1;
  if(calcIntersectionAngle(reflectionRay, reflectionAngle)) return 1;

  return 0;
}
