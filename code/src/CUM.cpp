#include "CUM.h"
#include "Macros.h"
#include "Application.h"

#include <fstream>

uint64 CUM::Hash(const std::string &path) {
    std::hash<std::string> hash;
    return hash(path);
}

bool CUM::IsInViewport(glm::ivec2 position, Viewport* viewport) {
    return position.x >= viewport->position.x &&
    position.x <= viewport->position.x + viewport->resolution.x &&
    position.y <= Application::resolution.y - viewport->position.y &&
    position.y >= Application::resolution.y - (viewport->position.y + viewport->resolution.y);
}

unsigned char* CUM::ResizeImage(const unsigned char *image, int width, int height, int newWidth, int newHeight) {
    unsigned char* resizedImage = new unsigned char[newWidth * newHeight * 3];

    float widthRatio = (float)(width - 1) / (float)(newWidth - 1);
    float heightRatio = (float)(height - 1) / (float)(newHeight - 1);

    // Bilinear interpolation
    for (int y = 0; y < newHeight; y++) {
        for (int x = 0; x < newWidth; x++) {
            float xFloor = floor(widthRatio * (float)x);
            float yFloor = floor(heightRatio * (float)y);
            float xCeil = ceil(widthRatio * (float)x);
            float yCeil = ceil(heightRatio * (float)y);

            float xWeight = (widthRatio * (float)x) - xFloor;
            float yWeight = (heightRatio * (float)y) - yFloor;

            float a = image[(int)yFloor * width + (int)xFloor];
            float b = image[(int)yFloor * width + (int)xCeil];
            float c = image[(int)yCeil * width + (int)xFloor];
            float d = image[(int)yCeil * width + (int)xCeil];

            resizedImage[y * newWidth + x] = a * (1.0 - xWeight) * (1.0 - yWeight) +
                                             b * xWeight * (1.0 - yWeight) +
                                             c * yWeight * (1.0 - xWeight) +
                                             d * xWeight * yWeight;
        }
    }

    return resizedImage;
}

unsigned char* CUM::RotateImage(const unsigned char *image, int width, int height, int dim) {
    unsigned char* flippedData = new unsigned char[width * height * dim];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width * dim; ++j) {
            flippedData[(height - 1 - i) * width * dim + j] = image[i * width * dim + j];
        }
    }

    return flippedData;
}

float *CUM::CartesianCoordsToSphericalAngles(glm::vec3 position) {
    float* output = new float[2];

    position = glm::normalize(position);

    output[0] = std::atan2(position.z, position.x);
    output[1] = std::acos(position.y);

    return output;
}

glm::vec3 CUM::SphericalAnglesToCartesianCoordinates(float phi, float theta, float radius) {
    glm::vec3 position;

    position.x = std::cos(phi) * std::sin(theta);
    position.y = std::cos(theta);
    position.z = std::sin(phi) * std::sin(theta);

    return glm::normalize(position) * radius;
}

void CUM::SaveJsonToFile(const std::string& filePath, const nlohmann::json& json) {
    std::filesystem::path path(filePath);
    if (std::filesystem::exists(path)) {
        std::filesystem::remove(path);
    }
    std::ofstream file(filePath);
    file << json.dump(4);
    file.close();

#ifdef DEBUG
    ILR_INFO_MSG("Json was successfully saved to a file: " + filePath);
#endif
}

bool CUM::LoadJsonFromFile(const std::string &filePath, nlohmann::json &json) {
    if (filePath.empty()) {
#ifdef DEBUG
        ILR_ERROR_MSG("Wrong path: " + filePath + " should not be empty");
#endif
        return false;
    }
    std::ifstream file(filePath);
    if (!file.is_open() || std::filesystem::file_size(filePath) == 0) {
#ifdef DEBUG
        ILR_ERROR_MSG("Failed to open file or file is empty: " + filePath);
#endif
        return false;
    }

    json = nlohmann::json::parse(file);
    file.close();

#ifdef DEBUG
    ILR_INFO_MSG("Json was successfully loaded from a file: " + filePath);
#endif
    return true;
}
