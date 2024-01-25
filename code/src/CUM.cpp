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

    for (int y = 0; y < newHeight; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            float imageX = (float)x / newWidth * width;
            float imageY = (float)y / newHeight * height;

            int x1 = (int)std::floor(imageX);
            int y1 = (int)std::floor(imageY);
            int x2 = x1 + 1;
            int y2 = y1 + 1;

            // Bilinear interpolation
            float u = imageX - x1;
            float v = imageY - y1;

            for (int channel = 0; channel < 3; ++channel) {
                resizedImage[(y * newWidth + x) * 3 + channel] =
                        (unsigned char)((1 - u) * (1 - v) * image[(y1 * width + x1) * 3 + channel] +
                        u * (1 - v) * image[(y1 * width + x2) * 3 + channel] +
                        (1 - u) * v * image[(y2 * width + x1) * 3 + channel] +
                        u * v * image[(y2 * width + x2) * 3 + channel]);
            }
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
