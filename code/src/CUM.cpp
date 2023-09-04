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

void CUM::LoadJsonFromFile(const std::string &filePath, nlohmann::json &json) {
    if (filePath.empty()) {
#ifdef DEBUG
        ILR_ERROR_MSG("Wrong path: " + filePath + " should not be empty");
#endif
        return;
    }
    std::ifstream file(filePath);
    if (!file.is_open()) {
#ifdef DEBUG
        ILR_ERROR_MSG("Failed to open file: " + filePath);
#endif
        return;
    }
    json = nlohmann::json::parse(file);
    file.close();

#ifdef DEBUG
    ILR_INFO_MSG("Json was successfully loaded from a file: " + filePath);
#endif
}