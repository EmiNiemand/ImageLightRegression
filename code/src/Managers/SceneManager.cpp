#include "Managers/SceneManager.h"
#include "Managers/InputManager.h"
#include "Managers/EditorManager.h"
#include "Core/Object.h"
#include "Macros.h"
#include "CUM.h"
#include "Application.h"

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
    ClearScene();
    delete sceneManager;
}

void SceneManager::ClearScene() {
    Application* application = Application::GetInstance();
    Object* scene = application->scene;

    for (auto child : scene->children) {
        if (child.second->visibleInEditor) {
            Object::Destroy(child.second);
        }
    }

    application->DestroyQueuedComponents();
    application->DestroyQueuedObjects();
}

void SceneManager::SaveScene(const std::string& filePath) {
    nlohmann::json jsonScene = nlohmann::json::array();

    jsonScene.push_back(nlohmann::json::object());
    Application::GetInstance()->scene->Save(jsonScene.back());

    CUM::SaveJsonToFile(filePath, jsonScene);
}

void SceneManager::LoadScene(const std::string& filePath) {
    nlohmann::json jsonScene;
    CUM::LoadJsonFromFile(filePath, jsonScene);

    loadedPath = filePath;

    Application::GetInstance()->scene->Load(jsonScene.front());
}

void SceneManager::Update() {
    if (InputManager::GetInstance()->IsKeyPressed(Key::KEY_LEFT_CONTROL) && InputManager::GetInstance()->IsKeyDown(Key::KEY_S)) {
        if (!loadedPath.empty()) {
            SaveScene(loadedPath);
        }
        else {
            for (int i = 0;; ++i) {
                std::filesystem::path filePath(EditorManager::GetInstance()->fileExplorerCurrentPath);

                if (i == 0) {
                    filePath /= (std::string("Scene") + ".scn");
                }
                else {
                    filePath /= ("Scene_" + std::to_string(i) + ".scn");
                }
                if (!std::filesystem::exists(filePath)) {
                    SaveScene(filePath.string());
                    break;
                }
            }

        }
    }
}
