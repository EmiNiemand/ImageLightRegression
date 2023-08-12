#include "Managers/EditorManager.h"
#include "Application.h"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

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

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
