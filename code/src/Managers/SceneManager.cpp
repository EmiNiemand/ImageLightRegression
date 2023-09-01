#include "Managers/SceneManager.h"
#include "Core/Object.h"
#include "Macros.h"
#include "Application.h"

#include <fstream>

SceneManager::SceneManager() = default;
SceneManager::~SceneManager() = default;

SceneManager *SceneManager::GetInstance() {
    if (sceneManager == nullptr) {
        sceneManager = new SceneManager();
    }
    return sceneManager;
}

void SceneManager::Startup() {

}

void SceneManager::Shutdown() {
    delete sceneManager;
}

void SceneManager::SaveScene(const std::string& filePath) {
    nlohmann::json jsonScene = nlohmann::json::array();

    jsonScene.push_back(nlohmann::json::object());
    Application::GetInstance()->scene->Save(jsonScene.back());

    SaveJsonToFile(filePath, jsonScene);
}

void SceneManager::LoadScene(const std::string& fileName) {
    nlohmann::json jsonScene;
    LoadJsonFromFile(fileName, jsonScene);

    Application::GetInstance()->scene->Load(jsonScene.front());
}

void SceneManager::SaveJsonToFile(const std::string& filePath, const nlohmann::json& json) {
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

void SceneManager::LoadJsonFromFile(const std::string &filePath, nlohmann::json &json) {
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
