#include "Editor/ToolBar.h"
#include "Managers/EditorManager.h"
#include "Managers/RenderingManager.h"
#include "Managers/SceneManager.h"
#include "Managers/NeuralNetworkManager.h"
#include "Rendering/ObjectRenderer.h"
#include "Resources/Texture.h"
#include "CUM.h"
#include "Application.h"

#include "imgui.h"
#include "imgui_internal.h"

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

    NeuralNetworkManager* neuralNetworkManager = NeuralNetworkManager::GetInstance();

    ImGui::SameLine(ImGui::GetWindowContentRegionWidth() - 180.0f);
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

    ImGui::SameLine();
    if (!application->isStarted || (application->isStarted && neuralNetworkManager->state != NetworkState::Training)) {
        ShowButton("TrainButton", editorManager->trainTexture->GetID(), neuralNetworkManager->state == NetworkState::Idle);
    }
    else if (application->isStarted && neuralNetworkManager->state == NetworkState::Training) {
        ShowButton("StopTrainButton", editorManager->stopTexture->GetID());
    }
    if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
        if (!application->isStarted && !editorManager->loadedImage) return;
        if (RenderingManager::GetInstance()->objectRenderer->pointLights[0] == nullptr) {
            ImGui::OpenPopup("MessagePopup");
            return;
        }

        application->isStarted = !application->isStarted;

        if (application->isStarted) NeuralNetworkManager::GetInstance()->InitializeNetwork(NetworkTask::TrainNetwork);
        if (!application->isStarted) NeuralNetworkManager::GetInstance()->FinalizeNetwork();
    }

    ImGui::SameLine();
    if (!application->isStarted || (application->isStarted && neuralNetworkManager->state != NetworkState::Processing)) {
        ShowButton("StartButton", editorManager->startTexture->GetID(), neuralNetworkManager->state == NetworkState::Idle);
    }
    else if (application->isStarted && neuralNetworkManager->state == NetworkState::Processing) {
        ShowButton("StopButton", editorManager->stopTexture->GetID());
    }
    if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
        if (!application->isStarted && !editorManager->loadedImage) return;
        if (RenderingManager::GetInstance()->objectRenderer->pointLights[0] == nullptr) {
            ImGui::OpenPopup("MessagePopup");
            return;
        }

        application->isStarted = !application->isStarted;

        if (application->isStarted) NeuralNetworkManager::GetInstance()->InitializeNetwork(NetworkTask::ProcessImage);
        if (!application->isStarted) NeuralNetworkManager::GetInstance()->FinalizeNetwork();
    }
    if (ImGui::BeginPopupModal("MessagePopup", 0, ImGuiWindowFlags_Popup | ImGuiWindowFlags_NoTitleBar |
                                                  ImGuiWindowFlags_AlwaysAutoResize)) {
        ImVec2 windowSize = ImVec2((float)Application::resolution.x * 0.15f, (float)Application::resolution.y * 0.15f);
        ImVec2 windowPosition = ImVec2((float)Application::resolution.x * 0.5f - windowSize.x / 2,
                                       (float)Application::resolution.y * 0.5f - windowSize.y / 2);

        ImGui::SetWindowSize(windowSize);
        ImGui::SetWindowPos(windowPosition);

        ImFont* font = ImGui::GetFont();
        float oldScale = font->Scale;
        font->Scale = 1.5f;
        ImGui::PushFont(font);

        float textWidth = ImGui::CalcTextSize("No light on the scene").x;

        ImGui::SetCursorPosY(windowSize.y * 0.15f);
        ImGui::Text("No light on the scene");

        font->Scale = oldScale;
        ImGui::PopFont();

        ImGui::SetCursorPosX((windowSize.x - textWidth) * 0.5f);
        ImGui::SetCursorPosY(windowSize.y * 0.45f);

        if (ImGui::Button("Ok", ImVec2(windowSize.x * 0.5f, 0.0f))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    ImGui::EndChild();
}

void ToolBar::ShowButton(const std::string& label, unsigned int textureID, bool isActive) {
    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, !isActive);

    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_BorderShadow, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));

    ImVec2 thumbnailSizeVec2 = ImVec2(THUMBNAIL_SIZE, THUMBNAIL_SIZE);
    ImVec2 thumbnailUV0 = ImVec2(0.0f, 1.0f);
    ImVec2 thumbnailUV1 = ImVec2(1.0f, 0.0f);

    ImGui::PushID(label.c_str());

    if (isActive) {
        ImGui::ImageButton((ImTextureID)textureID, thumbnailSizeVec2, thumbnailUV0, thumbnailUV1);
    }
    else {
        ImGui::ImageButton((ImTextureID)textureID, thumbnailSizeVec2, thumbnailUV0, thumbnailUV1, -1, {0, 0, 0, 0},
                           {0.5, 0.5, 0.5, 1});
    }

    ImGui::PopID();

    ImGui::PopStyleColor(3);

    ImGui::PopItemFlag();
}

void ToolBar::SaveRenderToFile(const std::string& path) {
    int width = Application::viewports[0].resolution.x;
    int height = Application::viewports[0].resolution.y;

    unsigned char* data = new unsigned char[width * height * 4];

    // Get texture image
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, RenderingManager::GetInstance()->objectRenderer->renderingCameraTexture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

    unsigned char* flippedData = CUM::RotateImage(data, width, height, 4);

    stbi_write_png(path.c_str(), width, height , 4, flippedData, width * 4);

    delete[] data;
    delete[] flippedData;
}
