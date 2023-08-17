#include "Managers/EditorManager.h"
#include "Managers/ResourceManager.h"
#include "Editor/SceneTree.h"
#include "Editor/Inspector.h"
#include "Editor/FileExplorer.h"
#include "Editor/IconsMaterialDesign.h"
#include "Components/Rendering/UI/Image.h"
#include "Resources/Texture.h"
#include "Application.h"
#include "Core/Object.h"

#define WINDOW_FLAGS (ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | \
                      ImGuiWindowFlags_NoScrollbar)

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
    SetUnityTheme();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(Application::GetInstance()->window, false);
    ImGui_ImplOpenGL3_Init(glsl_version);

    ImGui_ImplGlfw_InstallCallbacks(Application::GetInstance()->window);

    fileTexture = ResourceManager::LoadResource<Texture>("resources/EditorIcons/File.png");
    directoryTexture = ResourceManager::LoadResource<Texture>("resources/EditorIcons/Directory.png");
}

void EditorManager::Shutdown() {
    ResourceManager::UnloadResource(fileTexture->GetPath());
    ResourceManager::UnloadResource(directoryTexture->GetPath());

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    delete editorManager;
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
    ShowLoadedImage();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void EditorManager::ShowToolBar() const {
    ImGui::Begin("ToolBar", nullptr, WINDOW_FLAGS);
    ImGui::SetWindowSize("ToolBar", ImVec2((float)Application::resolution.x, (float)Application::resolution.y / 9));
    ImGui::SetWindowPos("ToolBar", ImVec2(0.0f, 0.0f));
    ImGui::End();
}

void EditorManager::ShowSceneTree() const {
    ImGui::Begin("SceneTree", nullptr, WINDOW_FLAGS);
    ImGui::SetWindowSize("SceneTree", ImVec2((float)Application::resolution.x / 16 * 3, (float)Application::resolution.y / 9 * 4));
    ImGui::SetWindowPos("SceneTree", ImVec2(0.0f, (float)Application::resolution.y / 9));

    SceneTree::ShowTreeNode(Application::GetInstance()->scene);
    SceneTree::ShowPopUp();

    ImGui::End();
}

void EditorManager::ShowInspector() const {
    ImGui::Begin("Properties", nullptr, WINDOW_FLAGS);
    ImGui::SetWindowSize("Properties", ImVec2((float)Application::resolution.x / 16 * 3, (float)Application::resolution.y / 9 * 6));
    ImGui::SetWindowPos("Properties", ImVec2((float)Application::resolution.x / 16 * 13, (float)Application::resolution.y / 9));

    if (selectedNode != nullptr && selectedNode != Application::GetInstance()->scene) {
        Inspector::ShowName();
        for (auto& component : selectedNode->components) {
            Inspector::ShowComponentProperties(component.second);
        }
        Inspector::ShowPopUp();
    }

    ImGui::End();
}

void EditorManager::ShowFileExplorer() const {
    ImGui::Begin("File Explorer", nullptr, WINDOW_FLAGS);
    ImGui::SetWindowSize("File Explorer", ImVec2((float)Application::resolution.x, (float)Application::resolution.y / 9 * 2));
    ImGui::SetWindowPos("File Explorer", ImVec2(0.0f, (float)Application::resolution.y / 9 * 7));

    FileExplorer::ShowFiles();

    ImGui::End();
}

void EditorManager::ShowLoadedImage() const {
    ImGui::Begin("Loaded Image", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground);
    ImGui::SetWindowSize("Loaded Image", ImVec2((float)Application::viewports[1].resolution.x,
                                                (float)Application::viewports[1].resolution.y));
    ImGui::SetWindowPos("Loaded Image", ImVec2((float)0.0f, (float)(float)Application::resolution.y / 9 * 5));

    ImGui::BeginChild("##canvas");
    ImGui::EndChild();

    if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload *payload = ImGui::AcceptDragDropPayload("DNDTexturePath")) {
            std::string payloadData = *(const std::string *)payload->Data;
            std::string searchString = "\\";
            std::string replaceString = "/";

            size_t pos = payloadData.find(searchString);
            while (pos != std::string::npos) {
                payloadData.replace(pos, searchString.length(), replaceString);
                pos = payloadData.find(searchString, pos + replaceString.length());
            }
            Application::GetInstance()->loadedImage->GetComponentByClass<Image>()->SetTexture(payloadData);
        }

        ImGui::EndDragDropTarget();
    }

    ImGui::End();
}

void EditorManager::SetUnityTheme() {
    ImVec4 *colors = ImGui::GetStyle().Colors;
    colors[ImGuiCol_Text] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
    colors[ImGuiCol_WindowBg] = ImVec4(0.22f, 0.22f, 0.22f, 1.00f);
    colors[ImGuiCol_ChildBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg] = ImVec4(0.19f, 0.19f, 0.19f, 0.92f);
    colors[ImGuiCol_Border] = ImVec4(0.13f, 0.13f, 0.13f, 1.00f);
    colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.06f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    colors[ImGuiCol_FrameBgActive] = ImVec4(0.24f, 0.24f, 0.24f, 1.00f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
    colors[ImGuiCol_MenuBarBg] = ImVec4(0.10f, 0.10f, 0.10f, 1.00f);
    colors[ImGuiCol_ScrollbarBg] = ImVec4(0.21f, 0.21f, 0.21f, 0.54f);
    colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.37f, 0.37f, 0.37f, 0.54f);
    colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.41f, 0.41f, 0.41f, 0.54f);
    colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.37f, 0.37f, 0.37f, 0.54f);
    colors[ImGuiCol_CheckMark] = ImVec4(0.90f, 0.90f, 0.90f, 1.00f);
    colors[ImGuiCol_SliderGrab] = ImVec4(0.34f, 0.34f, 0.34f, 0.54f);
    colors[ImGuiCol_SliderGrabActive] = ImVec4(0.56f, 0.56f, 0.56f, 0.54f);
    colors[ImGuiCol_Button] = ImVec4(0.35f, 0.35f, 0.35f, 1.00f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.40f, 0.40f, 0.40f, 1.00f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.28f, 0.38f, 0.49f, 1.00f);
    colors[ImGuiCol_Header] = ImVec4(0.24f, 0.24f, 0.24f, 1.00f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.28f, 0.28f, 0.28f, 1.00f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.24f, 0.24f, 0.24f, 0.33f);
    colors[ImGuiCol_Separator] = ImVec4(0.10f, 0.10f, 0.10f, 1.00f);
    colors[ImGuiCol_SeparatorHovered] = ImVec4(0.10f, 0.10f, 0.10f, 1.00f);
    colors[ImGuiCol_SeparatorActive] = ImVec4(0.10f, 0.10f, 0.10f, 1.00f);
    colors[ImGuiCol_ResizeGrip] = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
    colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.44f, 0.44f, 0.44f, 0.29f);
    colors[ImGuiCol_ResizeGripActive] = ImVec4(0.40f, 0.44f, 0.47f, 1.00f);
    colors[ImGuiCol_Tab] = ImVec4(0.16f, 0.16f, 0.16f, 0.52f);
    colors[ImGuiCol_TabHovered] = ImVec4(0.19f, 0.19f, 0.19f, 1.00f);
    colors[ImGuiCol_TabActive] = ImVec4(0.24f, 0.24f, 0.24f, 1.00f);
    colors[ImGuiCol_TabUnfocused] = ImVec4(0.16f, 0.16f, 0.16f, 0.52f);
    colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.24f, 0.24f, 0.24f, 1.00f);
    colors[ImGuiCol_PlotLines] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotHistogram] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_TableHeaderBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_TableBorderStrong] = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_TableBorderLight] = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
    colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.00f, 1.00f, 1.00f, 0.06f);
    colors[ImGuiCol_TextSelectedBg] = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
    colors[ImGuiCol_DragDropTarget] = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
    colors[ImGuiCol_NavHighlight] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_NavWindowingHighlight] = ImVec4(0.71f, 0.71f, 0.71f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg] = ImVec4(1.00f, 0.00f, 0.00f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg] = ImVec4(1.00f, 0.00f, 0.00f, 0.35f);

    ImGuiStyle &style = ImGui::GetStyle();
    style.WindowPadding = ImVec2(8.00f, 8.00f);
    style.FramePadding = ImVec2(5.00f, 2.00f);
    style.CellPadding = ImVec2(6.00f, 1.00f);
    style.ItemSpacing = ImVec2(6.00f, 3.00f);
    style.ItemInnerSpacing = ImVec2(6.00f, 6.00f);
    style.TouchExtraPadding = ImVec2(0.00f, 0.00f);
    style.IndentSpacing = 25;
    style.ScrollbarSize = 15;
    style.GrabMinSize = 10;
    style.WindowBorderSize = 1;
    style.ChildBorderSize = 0;
    style.PopupBorderSize = 1;
    style.FrameBorderSize = 1;
    style.TabBorderSize = 0;
    style.WindowRounding = 7;
    style.ChildRounding = 4;
    style.FrameRounding = 2;
    style.PopupRounding = 4;
    style.ScrollbarRounding = 9;
    style.GrabRounding = 2;
    style.LogSliderDeadzone = 4;
    style.TabRounding = 2;

    ImGuiIO &io = ImGui::GetIO();
    ImFontConfig config;
    config.OversampleH = 4;
    config.OversampleV = 4;
    float baseFontSize = 15.0f;
    io.Fonts->AddFontFromFileTTF("resources/Fonts/Lato-Regular.ttf", 15, &config);

    // Setup ImGui icons
    static const ImWchar iconsRanges[] = {ICON_MIN_MD, ICON_MAX_16_MD, 0};
    float iconFontSize = baseFontSize * 2.0f / 3.0f;
    ImFontConfig iconsConfig;
    iconsConfig.MergeMode = true;
    iconsConfig.PixelSnapH = true;
    iconsConfig.OversampleH = 4;
    iconsConfig.OversampleV = 4;
    iconsConfig.GlyphMinAdvanceX = iconFontSize;

    io.Fonts->AddFontFromFileTTF("resources/Fonts/MaterialIcons-Regular.ttf", iconFontSize, &iconsConfig, iconsRanges);
}
