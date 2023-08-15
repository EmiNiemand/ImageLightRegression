#ifndef IMAGELIGHTREGRESSION_EDITORMANAGER_H
#define IMAGELIGHTREGRESSION_EDITORMANAGER_H

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

class Object;

class EditorManager {
public:
    Object* selectedNode = nullptr;

private:
    inline static EditorManager* editorManager;

    // Decide GL+GLSL versions
    // GL 4.3 + GLSL 430
    const char* glsl_version = "#version 430";

    int windowFlags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoScrollbar;

public:
    EditorManager(EditorManager &other) = delete;
    void operator=(const EditorManager&) = delete;
    virtual ~EditorManager();

    static EditorManager* GetInstance();

    void Startup();
    void Shutdown();

    void Show();

private:
    explicit EditorManager();

    void ShowToolBar();
    void ShowSceneTree();
    void ShowInspector();
    void ShowFileExplorer();
};


#endif //IMAGELIGHTREGRESSION_EDITORMANAGER_H
