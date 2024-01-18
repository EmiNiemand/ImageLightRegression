#ifndef IMAGELIGHTREGRESSION_CUM_H
#define IMAGELIGHTREGRESSION_CUM_H

#include "ApplicationTypes.h"
#include "Structures.h"

#include "nlohmann/json.hpp"

#include <string>

class CUM {
public:
    static uint64 Hash(const std::string& path);
    static bool IsInViewport(glm::ivec2 position, Viewport* viewport);

    static unsigned char* ResizeImage(const unsigned char* image, int width, int height, int newWidth, int newHeight);
    static unsigned char* RotateImage(const unsigned char* image, int width, int height, int dim);

    /// Returns two angles (phi, theta) as array of floats
    static float* CartesianCoordsToSphericalAngles(glm::vec3 position);
    /// Return position as vec3
    static glm::vec3 SphericalAnglesToCartesianCoordinates(float phi, float theta, float radius = 1);

    static void SaveJsonToFile(const std::string& filePath, const nlohmann::json& json);
    static bool LoadJsonFromFile(const std::string& filePath, nlohmann::json& json);
};


#endif //IMAGELIGHTREGRESSION_CUM_H
