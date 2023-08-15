#include "Editor/SceneTree.h"
#include "Managers/EditorManager.h"
#include "Core/Object.h"
#include "Macros.h"

#include "imgui.h"

SceneTree::SceneTree() = default;

SceneTree::~SceneTree() = default;

SceneTree *SceneTree::GetInstance() {
    if (sceneTree == nullptr) {
        sceneTree = new SceneTree();
    }
    return sceneTree;
}

void SceneTree::ShowTreeNode(Object* parent) {
    if (!parent->visibleInEditor) return;

    ImGuiTreeNodeFlags nodeFlags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick |
                                   ImGuiTreeNodeFlags_DefaultOpen;

    if (parent->children.empty()) {
        nodeFlags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;

        ImGui::TreeNodeEx((void*)(intptr_t)parent->id, nodeFlags, "%s", parent->name.c_str());

        if (ImGui::IsItemClicked() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
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
}

void SceneTree::ManageNodeInput(Object* hoveredObject) {
    if (ImGui::IsItemClicked() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        EditorManager::GetInstance()->selectedNode = hoveredObject;
    }
    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        ImGui::OpenPopup((hoveredObject->name + "ContextMenu").c_str(), ImGuiPopupFlags_NoOpenOverExistingPopup);
    }
    if (ImGui::BeginPopup((hoveredObject->name + "ContextMenu").c_str())) {
        if (ImGui::MenuItem("Add Object")) {
            Object::Instantiate("Object");
            ImGui::CloseCurrentPopup();
        }

        if (hoveredObject == Application::GetInstance()->scene) {
            ImGui::EndPopup();
            return;
        }

        if (ImGui::MenuItem("Add Child Object")) {
            Object::Instantiate("Object", hoveredObject);
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Remove Object")) {
            Object::Destroy(hoveredObject);
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

void SceneTree::ShowPopUp() {
    if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        ImGui::OpenPopup("SceneTreePopUpContext", ImGuiPopupFlags_NoOpenOverExistingPopup);
    }
    if (ImGui::BeginPopup("SceneTreePopUpContext")) {
        if (ImGui::MenuItem("Add Object")) {
            Object::Instantiate("Object");
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}
