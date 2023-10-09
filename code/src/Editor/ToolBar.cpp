#include "Editor/ToolBar.h"
#include "Managers/EditorManager.h"
#include "Managers/RenderingManager.h"
#include "Managers/SceneManager.h"
#include "Managers/NeuralNetworkManager.h"
#include "Rendering/ObjectRenderer.h"
#include "Resources/Texture.h"
#include "Application.h"

#include "imgui.h"

#include <glad/glad.h>
#include <stb_image_write.h>
#include <filesystem>

#define PADDING 16.0f
#define THUMBNAIL_SIZE 48.0f
#define CELL_SIZE (PADDING + THUMBNAIL_SIZE)

void ToolBar::ShowToolBar() {
    Application* application = Application::GetInstance();
    SceneManager* sceneManager = SceneManager::GetInstance();
    EditorManager* editorManager = EditorManager::GetInstance();

    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + (ImGui::GetContentRegionAvail().y - 50.0f) * 0.5f);

    ImGui::BeginChild("##canvas");

    ImGui::SameLine();
    ShowButton("NewScene", editorManager->newScene->GetID());
    if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
        ImGui::OpenPopup("SceneCreationPopup");
    }
    if (ImGui::BeginPopupModal("SceneCreationPopup", 0, ImGuiWindowFlags_Popup | ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("resources/");

        ImGui::SameLine();
        static char text[126] = "Scene Path And Name";
        ImGui::InputText("##ScenePathAndName", &text[0], sizeof(char) * 126);

        ImGui::SameLine();
        ImGui::Text(".jpg");

        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetWindowSize().x * 0.05f);

        if (ImGui::Button("Cancel", ImVec2(ImGui::GetWindowSize().x*0.40f, 0.0f))) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::SameLine();
        if (ImGui::Button("Accept", ImVec2(ImGui::GetWindowSize().x*0.40f, 0.0f))) {
            EditorManager::GetInstance()->selectedNode = nullptr;
            sceneManager->ClearScene();

            std::filesystem::path newScenePath(editorManager->fileExplorerCurrentPath);
            newScenePath /= text;
            sceneManager->SaveScene(newScenePath.string() + ".scn");


            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    ImGui::SameLine();
    ShowButton("SaveScene", editorManager->saveScene->GetID());
    if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
        sceneManager->SaveScene(sceneManager->loadedPath);
    }

    ImGui::SameLine(ImGui::GetWindowContentRegionWidth() - 120.0f);
    if (!application->isStarted) {
        ShowButton("StartButton", editorManager->startTexture->GetID());
    }
    else {
        ShowButton("StopButton", editorManager->stopTexture->GetID());
    }
    if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
        if (!application->isStarted && !editorManager->loadedImage) return;

        application->isStarted = !application->isStarted;

        if (application->isStarted) NeuralNetworkManager::GetInstance()->InitializeNetwork();
        if (!application->isStarted) NeuralNetworkManager::GetInstance()->Finalize();
    }

    ImGui::SameLine();
    ShowButton("RenderToFileButton", editorManager->renderToFileTexture->GetID());
    if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
        ImGui::OpenPopup("RenderToFilePopup");
    }

    if (ImGui::BeginPopupModal("RenderToFilePopup", 0, ImGuiWindowFlags_Popup | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("resources/Outputs/");

        ImGui::SameLine();
        static char text[126] = "File Name";
        ImGui::InputText("##FileName", &text[0], sizeof(char) * 126);

        ImGui::SameLine();
        ImGui::Text(".jpg");

        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetWindowSize().x * 0.05f);

        if (ImGui::Button("Cancel", ImVec2(ImGui::GetWindowSize().x*0.40f, 0.0f))) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::SameLine();
        if (ImGui::Button("Accept", ImVec2(ImGui::GetWindowSize().x*0.40f, 0.0f))) {
            std::filesystem::path filePath("resources/Outputs");
            filePath /= text;

            SaveRenderToFile(filePath.string() + ".png");

            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    ImGui::EndChild();
}

void ToolBar::ShowButton(const std::string& label, unsigned int textureID) {
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_BorderShadow, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));

    ImVec2 thumbnailSizeVec2 = ImVec2(THUMBNAIL_SIZE, THUMBNAIL_SIZE);
    ImVec2 thumbnailUV0 = ImVec2(0.0f, 1.0f);
    ImVec2 thumbnailUV1 = ImVec2(1.0f, 0.0f);

    ImGui::PushID(label.c_str());
    ImGui::ImageButton((ImTextureID)textureID, thumbnailSizeVec2, thumbnailUV0, thumbnailUV1);
    ImGui::PopID();

    ImGui::PopStyleColor(3);
}

void ToolBar::SaveRenderToFile(const std::string& path) {
    int width = Application::viewports[0].resolution.x;
    int height = Application::viewports[0].resolution.y;

    char* data = new char[width*height*4];

    // Get texture image
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, RenderingManager::GetInstance()->objectRenderer->screenTexture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

    char* flippedData = new char[width * height * 4];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width * 4; ++j) {
            flippedData[(height - 1 - i) * width * 4 + j] = data[i * width * 4 + j];
        }
    }
    delete[] data;

    stbi_write_png(path.c_str(), width, height , 4, flippedData, width * 4);
    delete[] flippedData;
}
