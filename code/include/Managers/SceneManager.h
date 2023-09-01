#ifndef IMAGELIGHTREGRESSION_SCENEMANAGER_H
#define IMAGELIGHTREGRESSION_SCENEMANAGER_H

#include "nlohmann/json.hpp"

#include <filesystem>

class SceneManager {
public:
    std::string loadedPath;

private:
    inline static SceneManager* sceneManager;

public:
    SceneManager(SceneManager &other) = delete;
    virtual ~SceneManager();
    void operator=(const SceneManager&) = delete;

    static SceneManager* GetInstance();

    void Startup();
    void Shutdown();

    void SaveScene(const std::string& filePath);
    void LoadScene(const std::string& filePath);

private:
    explicit SceneManager();

    void SaveJsonToFile(const std::string& filePath, const nlohmann::json& json);
    void LoadJsonFromFile(const std::string& filePath, nlohmann::json& json);
};


#endif //IMAGELIGHTREGRESSION_SCENEMANAGER_H
