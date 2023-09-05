#ifndef IMAGELIGHTREGRESSION_SCENEMANAGER_H
#define IMAGELIGHTREGRESSION_SCENEMANAGER_H

#include <string>

class Object;

class SceneManager {
public:
    Object* scene = nullptr;

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

    void Update();

    void ClearScene();
    void SaveScene(const std::string& filePath);
    void LoadScene(const std::string& filePath);

private:
    explicit SceneManager();
};


#endif //IMAGELIGHTREGRESSION_SCENEMANAGER_H
