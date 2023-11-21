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
    unsigned char* resizedImage = new unsigned char[width * height * 3];

    float widthRatio = (float)(width - 1) / (float)(newWidth - 1);
    float heightRatio = (float)(height - 1) / (float)(newHeight - 1);

    for (int y = 0; y < newHeight; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            int srcX = (int)((float)x * widthRatio);
            int srcY = (int)((float)y * heightRatio);

            resizedImage[(y * newWidth + x) * 3] = image[(srcY * width + srcX) * 3];           // Red component
            resizedImage[(y * newWidth + x) * 3 + 1] = image[(srcY * width + srcX) * 3 + 1];   // Green component
            resizedImage[(y * newWidth + x) * 3 + 2] = image[(srcY * width + srcX) * 3 + 2];   // Blue component
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

float *CUM::CartesianToSphericalCoordinates(glm::vec3 position) {
    float* output = new float[2];

    position = glm::normalize(position);

    output[0] = std::atan2(position.z, position.x);
    output[1] = std::acos(position.y);

    return output;
}

glm::vec3 CUM::SphericalToCartesianCoordinates(float phi, float theta, float radius) {
    glm::vec3 position;

    position.x = std::cos(phi) * std::sin(theta);
    position.y = std::cos(theta);
    position.z = std::sin(phi) * std::sin(theta);

    return position * radius;
}

void CUM::SaveJsonToFile(const std::string& filePath, const nlohmann::json& json) {
    std::filesystem::path path(filePath);
    if (std::filesystem::exists(path)) {
#ifdef DEBUG
        ILR_INFO_MSG("Remove file");
#endif
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
