#include "Managers/EditorManager.h"
#include "Editor/SceneTree.h"
#include "Application.h"
#include "Core/Object.h"
#include "Macros.h"

EditorManager::EditorManager() = default;

EditorManager::~EditorManager() = default;

EditorManager *EditorManager::GetInstance() {
    if (editorManager == nullptr) {
        editorManager = new EditorManager();
    }
    return editorManager;
}

void EditorManager::Startup() {
    ImGui::CreateContext();

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(Application::GetInstance()->window, false);
    ImGui_ImplOpenGL3_Init(glsl_version);

    ImGui_ImplGlfw_InstallCallbacks(Application::GetInstance()->window);
}

void EditorManager::Shutdown() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void EditorManager::Show() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Put code here
    ShowToolBar();
    ShowSceneTree();
    ShowInspector();
    ShowFileExplorer();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void EditorManager::ShowToolBar() {
    ImGui::Begin("ToolBar", nullptr, windowFlags);
    ImGui::SetWindowSize("ToolBar", ImVec2((float)Application::resolution.x, (float)Application::resolution.y / 9));
    ImGui::SetWindowPos("ToolBar", ImVec2(0.0f, 0.0f));
    ImGui::End();
}

void EditorManager::ShowSceneTree() {
    ImGui::Begin("SceneTree", nullptr, windowFlags);
    ImGui::SetWindowSize("SceneTree", ImVec2((float)Application::resolution.x / 16 * 3, (float)Application::resolution.y / 9 * 4));
    ImGui::SetWindowPos("SceneTree", ImVec2(0.0f, (float)Application::resolution.y / 9));


    SceneTree::GetInstance()->ShowTreeNode(Application::GetInstance()->scene);
    SceneTree::GetInstance()->ShowPopUp();

    ImGui::End();
}

void EditorManager::ShowInspector() {
    ImGui::Begin("Properties", nullptr, windowFlags);
    ImGui::SetWindowSize("Properties", ImVec2((float)Application::resolution.x / 16 * 3, (float)Application::resolution.y / 9 * 6));
    ImGui::SetWindowPos("Properties", ImVec2((float)Application::resolution.x / 16 * 13, (float)Application::resolution.y / 9));
    ImGui::End();
}

void EditorManager::ShowFileExplorer() {
    ImGui::Begin("File Explorer", nullptr, windowFlags);
    ImGui::SetWindowSize("File Explorer", ImVec2((float)Application::resolution.x, (float)Application::resolution.y / 9 * 2));
    ImGui::SetWindowPos("File Explorer", ImVec2(0.0f, (float)Application::resolution.y / 9 * 7));
    ImGui::End();
}
