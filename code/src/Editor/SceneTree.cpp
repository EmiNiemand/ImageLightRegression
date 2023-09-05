#include "Editor/SceneTree.h"
#include "Managers/EditorManager.h"
#include "Managers/SceneManager.h"
#include "Core/Object.h"
#include "Macros.h"

#include "imgui.h"

void SceneTree::ShowTreeNode(Object* parent) {
    if (!parent->visibleInEditor) return;

    ImGuiTreeNodeFlags nodeFlags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick |
                                   ImGuiTreeNodeFlags_DefaultOpen;

    if (!parent->GetEnabled()) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 0.3f, 0.3f, 1.0f));
    if (parent->children.empty()) {
        nodeFlags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;

        ImGui::TreeNodeEx((void*)(intptr_t)parent->id, nodeFlags, "%s", parent->name.c_str());
        if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
            EditorManager::GetInstance()->selectedNode = parent;
        }
        ManageNodeInput(parent);
    }
    else {
        bool isNodeOpen = ImGui::TreeNodeEx((void*)(intptr_t)parent->id, nodeFlags, "%s", parent->name.c_str());
        ManageNodeInput(parent);

        if (isNodeOpen) {
            for (auto& child : parent->children) {
                ShowTreeNode(child.second);
            }
            ImGui::TreePop();
        }
    }
    if (!parent->GetEnabled()) ImGui::PopStyleColor();
}

void SceneTree::ManageNodeInput(Object* hoveredObject) {
    if (ImGui::BeginDragDropSource()) {
        if (hoveredObject != SceneManager::GetInstance()->scene) {
            ImGui::SetDragDropPayload("DNDSceneObject", &hoveredObject->id, sizeof(int));
        }
        ImGui::EndDragDropSource();
    }
    if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload *payload = ImGui::AcceptDragDropPayload("DNDSceneObject")) {
            int payloadData = *(const int *)payload->Data;

            Object* child = Application::GetInstance()->objects.at(payloadData);

            if (hoveredObject->children.contains(payloadData)) {
                child->SetParent(hoveredObject->GetParent());
            }
            else {
                hoveredObject->AddChild(child);
            }
        }

        ImGui::EndDragDropTarget();
    }

    if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
        EditorManager::GetInstance()->selectedNode = hoveredObject;

    }
    if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
        ImGui::OpenPopup((std::to_string(hoveredObject->id) + "ContextMenu").c_str(), ImGuiPopupFlags_NoOpenOverExistingPopup);
    }
    if (ImGui::BeginPopup((std::to_string(hoveredObject->id) + "ContextMenu").c_str())) {
        if (ImGui::MenuItem("Add Object")) {
            EditorManager::GetInstance()->selectedNode = Object::Instantiate("Object");
            ImGui::CloseCurrentPopup();
        }

        if (hoveredObject == SceneManager::GetInstance()->scene) {
            ImGui::EndPopup();
            return;
        }

        if (ImGui::MenuItem("Add Child Object")) {
            EditorManager::GetInstance()->selectedNode = Object::Instantiate("Object", hoveredObject);
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Remove Object")) {
            Object::Destroy(hoveredObject);
            EditorManager::GetInstance()->selectedNode = nullptr;
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

void SceneTree::ShowPopUp() {
    if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        EditorManager::GetInstance()->selectedNode = nullptr;
    }
    if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        ImGui::OpenPopup("SceneTreePopUpContextMenu", ImGuiPopupFlags_NoOpenOverExistingPopup);
    }
    if (ImGui::BeginPopup("SceneTreePopUpContextMenu")) {
        if (ImGui::MenuItem("Add Object")) {
            EditorManager::GetInstance()->selectedNode = Object::Instantiate("Object");
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

