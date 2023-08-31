#include "Editor/ToolBar.h"
#include "Managers/EditorManager.h"
#include "Managers/RenderingManager.h"
#include "Rendering/ObjectRenderer.h"
#include "Resources/Texture.h"
#include "Application.h"

#include "imgui.h"

#include <glad/glad.h>
#include <stb_image_write.h>

#define PADDING 16.0f
#define THUMBNAIL_SIZE 48.0f
#define CELL_SIZE (PADDING + THUMBNAIL_SIZE)

void ToolBar::ShowToolBar() {
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + (ImGui::GetContentRegionAvail().y - 50.0f) * 0.5f);

    ImGui::BeginChild("##canvas");

    ImGui::SameLine(ImGui::GetWindowContentRegionWidth() - 120.0f);
    if (!Application::GetInstance()->isStarted) {
        ShowButton("StartButton", EditorManager::GetInstance()->startTexture->GetID());
    }
    else {
        ShowButton("StopButton", EditorManager::GetInstance()->stopTexture->GetID());
    }
    if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
        Application::GetInstance()->isStarted = !Application::GetInstance()->isStarted;
    }
    ImGui::SameLine();
    ShowButton("RenderToFileButton", EditorManager::GetInstance()->renderToFileTexture->GetID());
    if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
        SaveRenderToFile();
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

void ToolBar::SaveRenderToFile() {
    int width = Application::viewports[0].resolution.x;
    int height = Application::viewports[0].resolution.y;

    unsigned char* data = new unsigned char[width*height*4];

    // Get texture image
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, RenderingManager::GetInstance()->objectRenderer->screenTexture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

    unsigned char* flippedData = new unsigned char[width*height*4];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width * 4; ++j) {
            flippedData[(height - 1 - i) * width * 4 + j] = data[i * width * 4 + j];
        }
    }
    delete[] data;

    stbi_write_png("resources/Outputs/Render.png", width, height , 4, flippedData, width * 4);
    delete[] flippedData;
}
