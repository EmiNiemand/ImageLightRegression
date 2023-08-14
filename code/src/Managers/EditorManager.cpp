#include "Managers/EditorManager.h"
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
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;

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
    ShowProperties();
    ShowFileExplorer();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void EditorManager::ShowToolBar() {
    ImGui::Begin("ToolBar", &showSceneTree, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
                                              ImGuiWindowFlags_NoDecoration);
    ImGui::SetWindowSize("ToolBar", ImVec2((float)Application::resolution.x, (float)Application::resolution.y / 9));
    ImGui::SetWindowPos("ToolBar", ImVec2(0.0f, 0.0f));
    ImGui::End();
}

void EditorManager::ShowSceneTree() {
    ImGui::Begin("SceneTree", &showSceneTree, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
                                                ImGuiWindowFlags_NoDecoration);
    ImGui::SetWindowSize("SceneTree", ImVec2((float)Application::resolution.x / 16 * 3, (float)Application::resolution.y / 9 * 4));
    ImGui::SetWindowPos("SceneTree", ImVec2(0.0f, (float)Application::resolution.y / 9));

    ShowTreeChild(Application::GetInstance()->scene);

    ImGui::End();
}

void EditorManager::ShowTreeChild(Object* parent) {
    if (!parent->visibleInEditor) return;

    ImGuiTreeNodeFlags nodeFlags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick |
                                   ImGuiTreeNodeFlags_DefaultOpen;

    if (parent->children.empty()) {
        nodeFlags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
        ImGui::TreeNodeEx((void*)(intptr_t)parent->id, nodeFlags, "%s", parent->name.c_str());
    }
    else {
        bool isNodeOpen = ImGui::TreeNodeEx((void*)(intptr_t)parent->id, nodeFlags, "%s", parent->name.c_str());

        if (isNodeOpen) {
            for (auto& child : parent->children) {
                ShowTreeChild(child.second);
            }
            ImGui::TreePop();
        }
    }
}

void EditorManager::ShowProperties() {
    ImGui::Begin("Properties", &showSceneTree, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
                                              ImGuiWindowFlags_NoDecoration);
    ImGui::SetWindowSize("Properties", ImVec2((float)Application::resolution.x / 16 * 3, (float)Application::resolution.y / 9 * 6));
    ImGui::SetWindowPos("Properties", ImVec2((float)Application::resolution.x / 16 * 13, (float)Application::resolution.y / 9));
    ImGui::End();
}

void EditorManager::ShowFileExplorer() {
    ImGui::Begin("File Explorer", &showSceneTree, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
                                               ImGuiWindowFlags_NoDecoration);
    ImGui::SetWindowSize("File Explorer", ImVec2((float)Application::resolution.x, (float)Application::resolution.y / 9 * 2));
    ImGui::SetWindowPos("File Explorer", ImVec2(0.0f, (float)Application::resolution.y / 9 * 7));
    ImGui::End();
}
