#ifndef IMAGELIGHTREGRESSION_EDITORMANAGER_H
#define IMAGELIGHTREGRESSION_EDITORMANAGER_H

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include <string>

class Object;
class Texture;
class Gizmos;

class EditorManager {
public:
    Object* selectedNode = nullptr;

    std::string dndPath = {};
    std::string fileExplorerCurrentPath = "resources";

    Gizmos* gizmos = nullptr;

    Texture* fileTexture = nullptr;
    Texture* directoryTexture = nullptr;

    Texture* startTexture = nullptr;
    Texture* stopTexture = nullptr;
    Texture* renderToFileTexture = nullptr;
    Texture* newScene = nullptr;
    Texture* saveScene = nullptr;

private:
    inline static EditorManager* editorManager;

    // Decide GL+GLSL versions
    // GL 4.3 + GLSL 430
    const char* glsl_version = "#version 430";

public:
    EditorManager(EditorManager &other) = delete;
    void operator=(const EditorManager&) = delete;
    virtual ~EditorManager();

    static EditorManager* GetInstance();

    void Startup();
    void Shutdown();

    void Update();
    void Draw();

private:
    explicit EditorManager();

    void ShowToolBar() const;
    void ShowSceneTree() const;
    void ShowInspector() const;
    void ShowFileExplorer() const;
    void ShowLoadedImage() const;

    static void SetUnityTheme();

    void SaveSettings();
    void LoadSettings();
};


#endif //IMAGELIGHTREGRESSION_EDITORMANAGER_H
