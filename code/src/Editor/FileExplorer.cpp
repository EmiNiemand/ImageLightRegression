#include "Editor/FileExplorer.h"
#include "Editor/IconsMaterialDesign.h"
#include "Managers/EditorManager.h"
#include "Managers/SceneManager.h"
#include "Resources/Texture.h"
#include "Core/Object.h"
#include "Application.h"

#include "imgui.h"
#include "imgui_internal.h"

#include <filesystem>

#define PADDING 16.0f
#define THUMBNAIL_SIZE 48.0f
#define CELL_SIZE (PADDING + THUMBNAIL_SIZE)


void FileExplorer::ShowFiles() {
    std::string& fileExplorerCurrentPath = EditorManager::GetInstance()->fileExplorerCurrentPath;

    ImGui::BeginTable("File Explorer", 2, ImGuiTableFlags_Resizable);
    ImGui::TableSetupColumn("Folders", ImGuiTableColumnFlags_WidthFixed, 120.0f);

    ImGui::TableNextRow(ImGuiTableRowFlags_None, ImGui::GetContentRegionAvail().y);
    ImGui::TableSetColumnIndex(0);

    ShowDirectory("resources");

    ImGui::TableNextColumn();

    float panelWidth = ImGui::GetContentRegionAvail().x;
    int columnCount = (int)(panelWidth / CELL_SIZE);
    if (columnCount < 1) {
        columnCount = 1;
    }

    ImGui::BeginChild("##ScrollingRegion", ImVec2(0, 0), false);

    if (ImGui::Button(ICON_MD_ARROW_LEFT)) {
        if (fileExplorerCurrentPath != "resources") {
            fileExplorerCurrentPath = std::filesystem::path(fileExplorerCurrentPath).parent_path().string();
        }
    }
    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();
    ImGui::Text("%s", fileExplorerCurrentPath.c_str());

    ImGui::Spacing();

    bool isOpen = ImGui::BeginTable("##File Explorer", columnCount, ImGuiTableFlags_NoBordersInBody);

    ImGui::TableNextColumn();

    for (auto &entry : std::filesystem::directory_iterator(fileExplorerCurrentPath)) {
        ImGui::BeginGroup();

        std::string label = entry.path().filename().string();
        std::string extension = entry.path().extension().string();

        ImVec2 thumbnailSizeVec2 = ImVec2(THUMBNAIL_SIZE, THUMBNAIL_SIZE);
        ImVec2 thumbnailUV0 = ImVec2(0.0f, 1.0f);
        ImVec2 thumbnailUV1 = ImVec2(1.0f, 0.0f);

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
        ImGui::PushStyleColor(ImGuiCol_BorderShadow, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));

        unsigned int entryTexture = EditorManager::GetInstance()->fileTexture->GetID();
        if (entry.is_directory()) {
            entryTexture = EditorManager::GetInstance()->directoryTexture->GetID();
        }
        ImGui::PushID(label.c_str());
        ImGui::ImageButton((ImTextureID)entryTexture, thumbnailSizeVec2, thumbnailUV0, thumbnailUV1);
        ImGui::PopID();

        std::string& dndPath = EditorManager::GetInstance()->dndPath;

        if (ImGui::BeginDragDropSource()) {
            if (extension == ".jpg" || extension == ".png") {
                dndPath = entry.path().string();

                ImGui::SetDragDropPayload("DNDTexturePath", &dndPath, sizeof(std::string));
                ImGui::Text("%s", label.c_str());
            }
            else if (extension == ".obj") {
                dndPath = entry.path().string();

                ImGui::SetDragDropPayload("DNDModelPath", &dndPath, sizeof(std::string));
                ImGui::Text("%s", label.c_str());
            }
            ImGui::EndDragDropSource();
        }

        ImGui::PopStyleColor(3);

        float text_scale = 0.85f;
        ImGui::SetWindowFontScale(text_scale);
        ImGuiStyle &style = ImGui::GetStyle();

        float size = ImGui::CalcTextSize(label.c_str()).x * text_scale + style.FramePadding.x * 2.0f;
        float avail = THUMBNAIL_SIZE;

        float off = (avail - size) * 0.5f;
        if (off > 0.0f) {
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + off);
        }

        ImGui::Text("%s", label.c_str());
        ImGui::SetWindowFontScale(1.0f);

        ImGui::Spacing();

        ImGui::EndGroup();

        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            if (entry.is_directory()) {
                fileExplorerCurrentPath = entry.path().string();
            } else {
                std::string name = entry.path().filename().string();
                if (extension == ".scn") {
                    EditorManager::GetInstance()->selectedNode = nullptr;
                    SceneManager::GetInstance()->ClearScene();
                    SceneManager::GetInstance()->LoadScene(entry.path().string());
                }
            }
        }

        ImGui::TableNextColumn();
    }

    if (isOpen) {
        ImGui::EndTable();
    }

    ImGui::EndChild();

    ImGui::EndTable();
}

void FileExplorer::ShowDirectory(const std::string& path) {
    std::string& fileExplorerCurrentPath = EditorManager::GetInstance()->fileExplorerCurrentPath;

    for (auto &entry : std::filesystem::directory_iterator(path)) {
        if (!entry.is_directory()) {
            continue;
        }

        static ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick |
                                               ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_SpanFullWidth;

        ImGuiTreeNodeFlags node_flags = base_flags;

        // Check if entry has folders inside it
        bool is_empty = true;
        for (auto &child : std::filesystem::directory_iterator(entry.path())) {
            if (child.is_directory()) {
                is_empty = false;
                break;
            }
        }
        if (is_empty) {
            node_flags |= ImGuiTreeNodeFlags_Leaf;
        } else {
            node_flags |= ImGuiTreeNodeFlags_OpenOnArrow;
        }

        if (fileExplorerCurrentPath == entry.path().string()) {
            node_flags |= ImGuiTreeNodeFlags_Selected;
        }

        bool node_open = ImGui::TreeNodeEx(entry.path().filename().string().c_str(), node_flags);

        if (ImGui::IsItemClicked()) {
            fileExplorerCurrentPath = entry.path().string();
        }

        if (node_open) {
            ShowDirectory(entry.path().string());

            ImGui::TreePop();
        }
    }
}
