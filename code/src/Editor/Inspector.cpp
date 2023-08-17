#pragma region Includes
#include "Editor/Inspector.h"
#include "Managers/ResourceManager.h"
#include "Managers/EditorManager.h"
#include "Core/Object.h"
#include "Components/Transform.h"
#include "Components/Rendering/Camera.h"
#include "Components/Rendering/EditorCamera.h"
#include "Components/Rendering/Renderer.h"
#include "Components/Rendering/Lights/DirectionalLight.h"
#include "Components/Rendering/Lights/PointLight.h"
#include "Components/Rendering/Lights/Spotlight.h"
#include "Components/Rendering/Skybox.h"
#include "Resources/Model.h"
#include "Resources/CubeMap.h"
#pragma endregion

#define HEADER_FLAGS (ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_DefaultOpen | \
                      ImGuiTreeNodeFlags_SpanFullWidth)

void Inspector::ShowPopUp() {
    if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        ImGui::OpenPopup("InspectorPopUpContextMenu", ImGuiPopupFlags_NoOpenOverExistingPopup);
    }
    if (ImGui::BeginPopup("InspectorPopUpContextMenu")) {
        if (ImGui::MenuItem("Add Camera Component")) {
            EditorManager::GetInstance()->selectedNode->AddComponent<Camera>();
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Add Renderer Component")) {
            EditorManager::GetInstance()->selectedNode->AddComponent<Renderer>();
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Add Directional Light Component")) {
            EditorManager::GetInstance()->selectedNode->AddComponent<DirectionalLight>();
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Add Point Light Component")) {
            EditorManager::GetInstance()->selectedNode->AddComponent<PointLight>();
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Add Spot Light Component")) {
            EditorManager::GetInstance()->selectedNode->AddComponent<SpotLight>();
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Add Skybox Component")) {
            EditorManager::GetInstance()->selectedNode->AddComponent<Skybox>();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void Inspector::ShowName() {
    float availableWidth = ImGui::GetContentRegionAvail().x;
    ImGui::BeginTable("Object", 2);
    ImGui::TableSetupColumn("##Col1", ImGuiTableColumnFlags_WidthFixed, availableWidth * 0.90f);

    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::InputText("##name", &EditorManager::GetInstance()->selectedNode->name[0], 256);

    ImGui::TableSetColumnIndex(1);
    ImGui::SetNextItemWidth(-FLT_MIN);

    bool enabled = EditorManager::GetInstance()->selectedNode->GetEnabled();

    if (ImGui::Checkbox("##enabled", &enabled)) {
        if (enabled) {
            EditorManager::GetInstance()->selectedNode->EnableSelfAndChildren();
        }
        else {
            EditorManager::GetInstance()->selectedNode->DisableSelfAndChildren();
        }
    }

    ImGui::EndTable();
}

void Inspector::ShowComponentProperties(Component* component) {
    if (dynamic_cast<Transform*>(component) != nullptr) {
        ShowTransform();
    }
    else if (dynamic_cast<Camera*>(component) != nullptr) {
        ShowCamera();
    }
    else if (dynamic_cast<Renderer*>(component) != nullptr) {
        ShowRenderer();
    }
    else if (dynamic_cast<DirectionalLight*>(component) != nullptr) {
        ShowDirectionalLight();
    }
    else if (dynamic_cast<PointLight*>(component) != nullptr) {
        ShowPointLight();
    }
    else if (dynamic_cast<SpotLight*>(component) != nullptr) {
        ShowSpotLight();
    }
    else if (dynamic_cast<Skybox*>(component) != nullptr) {
        ShowSkybox();
    }
}

#pragma region ComponentsProperties
void Inspector::ShowTransform() {
    Transform* transform = EditorManager::GetInstance()->selectedNode->transform;

    if (ImGui::CollapsingHeader("Transform", HEADER_FLAGS)) {
        if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
            ImGui::OpenPopup("TransformContextMenu");
        }
        if (ImGui::BeginPopup("TransformContextMenu", ImGuiPopupFlags_NoOpenOverExistingPopup)) {
            if (ImGui::MenuItem("Reset")) {
                transform->SetLocalPosition(glm::vec3(0.0f));
                transform->SetLocalRotation(glm::vec3(0.0f));
                transform->SetLocalScale(glm::vec3(1.0f));

                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }
        float availableWidth = ImGui::GetContentRegionAvail().x;
        ImGui::BeginTable("Transform", 2);
        ImGui::TableSetupColumn("##Col1", ImGuiTableColumnFlags_WidthFixed, availableWidth * 0.33f);


        glm::vec3 position = transform->GetLocalPosition();
        glm::vec3 rotation = transform->GetLocalRotation();
        glm::vec3 scale = transform->GetLocalScale();

        if (ShowVec3("Position", position)) {
            transform->SetLocalPosition(position);
        }
        if (ShowVec3("Rotation", rotation, 1.0f)) {
            transform->SetLocalRotation(rotation);
        }
        if (ShowVec3("Scale", scale, 0.02f, 1.0f)) {
            transform->SetLocalScale(scale);
        }

        ImGui::EndTable();
    }
}

void Inspector::ShowCamera() {
    Camera* camera = EditorManager::GetInstance()->selectedNode->GetComponentByClass<Camera>();

    if (!ShowComponentHeader(camera, "Camera")) {
        return;
    }

    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        ImGui::OpenPopup("CameraContextMenu");
    }
    if (ImGui::BeginPopup("CameraContextMenu", ImGuiPopupFlags_NoOpenOverExistingPopup)) {
        if (ImGui::MenuItem("Set As Rendering Camera")) {
            Camera::SetRenderingCamera(EditorManager::GetInstance()->selectedNode);

            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Reset")) {
            camera->SetFOV(45.0f);
            camera->SetZNear(0.1f);
            camera->SetZFar(100.0f);

            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Remove")) {
            Component::Destroy(camera);

            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    float availableWidth = ImGui::GetContentRegionAvail().x;
    ImGui::BeginTable("Camera", 2);
    ImGui::TableSetupColumn("##Col1", ImGuiTableColumnFlags_WidthFixed, availableWidth * 0.33f);

    float fov = camera->GetFOV();
    float zNear = camera->GetZNear();
    float zFar = camera->GetZFar();

    if (ShowFloat("FOV", &fov, 0.1f, 0.0f, 0.0f, 360.0f)) {
        camera->SetFOV(fov);
    }
    if (ShowFloat("zNear", &zNear)) {
        camera->SetZNear(zNear);
    }
    if (ShowFloat("zFar", &zFar)) {
        camera->SetZFar(zFar);
    }

    ImGui::EndTable();
}

void Inspector::ShowRenderer() {
    Renderer* renderer = EditorManager::GetInstance()->selectedNode->GetComponentByClass<Renderer>();

    if (!ShowComponentHeader(renderer, "Renderer")) {
        return;
    }

    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        ImGui::OpenPopup("RendererContextMenu");
    }
    if (ImGui::BeginPopup("RendererContextMenu", ImGuiPopupFlags_NoOpenOverExistingPopup)) {
        if (ImGui::MenuItem("Reset")) {
            renderer->material = {{1.0f, 1.0f, 1.0f}, 32.0f, 0, 0};
            renderer->texScale = glm::vec2(1.0f, 1.0f);
            renderer->model = nullptr;
            renderer->drawShadows = true;

            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Remove")) {
            Component::Destroy(renderer);

            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
    float availableWidth = ImGui::GetContentRegionAvail().x;
    ImGui::BeginTable("Renderer", 2);
    ImGui::TableSetupColumn("##Col1", ImGuiTableColumnFlags_WidthFixed, availableWidth * 0.33f);

    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("%s", "Material");
    ImGui::TableSetColumnIndex(1);
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::CollapsingHeader("Material", HEADER_FLAGS)) {
        ImGui::BeginTable("Material", 2);
        ShowVec3("Color", renderer->material.color, 0.01f, 0.0f, 0.0f, 1.0f);
        ShowFloat("Shininess", &renderer->material.shininess, 1.0f, 0.0f, 0.0f);
        ShowFloat("Reflection", &renderer->material.reflection, 0.01f, 0.0f, 0.0f, 1.0f);
        ShowFloat("Refraction", &renderer->material.refraction, 0.01f, 0.0f, 0.0f, 1.0f);
        ImGui::EndTable();
    }

    ShowVec2("Texture scale", renderer->texScale);

    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("Model");
    ImGui::TableSetColumnIndex(1);
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (renderer->model)
        ImGui::Text("%s", renderer->model->GetPath().substr(renderer->model->GetPath().find_last_of('/') + 1).c_str());
    else
        ImGui::Text("No model");

    if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload *payload = ImGui::AcceptDragDropPayload("DNDModelPath")) {
            std::string payloadData = *(const std::string *)payload->Data;
            std::string searchString = "\\";
            std::string replaceString = "/";

            size_t pos = payloadData.find(searchString);
            while (pos != std::string::npos) {
                payloadData.replace(pos, searchString.length(), replaceString);
                pos = payloadData.find(searchString, pos + replaceString.length());
            }
            renderer->LoadModel(payloadData);
        }

        ImGui::EndDragDropTarget();
    }

    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("%s", "Shadows");
    ImGui::TableSetColumnIndex(1);
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::Checkbox("##shadows", &renderer->drawShadows);
    ImGui::EndTable();
}

void Inspector::ShowDirectionalLight() {
    DirectionalLight* directionalLight = EditorManager::GetInstance()->selectedNode->GetComponentByClass<DirectionalLight>();

    if (!ShowComponentHeader(directionalLight, "Directional Light")) {
        return;
    }

    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        ImGui::OpenPopup("DirectionalLightContextMenu");
    }
    if (ImGui::BeginPopup("DirectionalLightContextMenu", ImGuiPopupFlags_NoOpenOverExistingPopup)) {
        if (ImGui::MenuItem("Reset")) {
            directionalLight->SetAmbient({0.4f, 0.4f, 0.4f});
            directionalLight->SetDiffuse({0.69f, 0.69f, 0.69f});
            directionalLight->SetSpecular({0.9f, 0.9f, 0.9f});
            directionalLight->SetColor({1.0f, 1.0f, 1.0f});

            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Remove")) {
            Component::Destroy(directionalLight);

            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    float availableWidth = ImGui::GetContentRegionAvail().x;
    ImGui::BeginTable("Directional Light", 2);
    ImGui::TableSetupColumn("##Col1", ImGuiTableColumnFlags_WidthFixed, availableWidth * 0.33f);

    glm::vec3 ambient = directionalLight->GetAmbient();
    glm::vec3 diffuse = directionalLight->GetDiffuse();
    glm::vec3 specular = directionalLight->GetSpecular();
    glm::vec3 color = directionalLight->GetColor();

    if (ShowVec3("Ambient", ambient, 0.1f, 0.0f, 0.0f)) {
        directionalLight->SetAmbient(ambient);
    }
    if (ShowVec3("Diffuse", diffuse, 0.1f, 0.0f, 0.0f)) {
        directionalLight->SetDiffuse(diffuse);
    }
    if (ShowVec3("Specular", specular, 0.1f, 0.0f, 0.0f)) {
        directionalLight->SetSpecular(specular);
    }
    if (ShowVec3("Color", color, 0.01f, 0.0f, 0.0f, 1.0f)) {
        directionalLight->SetColor(color);
    }

    ImGui::EndTable();
}

void Inspector::ShowPointLight() {
    PointLight* pointLight = EditorManager::GetInstance()->selectedNode->GetComponentByClass<PointLight>();

    if (!ShowComponentHeader(pointLight, "Point Light")) {
        return;
    }

    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        ImGui::OpenPopup("PointLightContextMenu");
    }
    if (ImGui::BeginPopup("PointLightContextMenu", ImGuiPopupFlags_NoOpenOverExistingPopup)) {
        if (ImGui::MenuItem("Reset")) {
            pointLight->SetAmbient({0.4f, 0.4f, 0.4f});
            pointLight->SetDiffuse({0.69f, 0.69f, 0.69f});
            pointLight->SetSpecular({0.9f, 0.9f, 0.9f});
            pointLight->SetColor({1.0f, 1.0f, 1.0f});
            pointLight->SetConstant(1.0f);
            pointLight->SetLinear(0.007f);
            pointLight->SetQuadratic(0.0002f);

            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Remove")) {
            Component::Destroy(pointLight);

            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    float availableWidth = ImGui::GetContentRegionAvail().x;
    ImGui::BeginTable("Point Light", 2);
    ImGui::TableSetupColumn("##Col1", ImGuiTableColumnFlags_WidthFixed, availableWidth * 0.33f);

    glm::vec3 ambient = pointLight->GetAmbient();
    glm::vec3 diffuse = pointLight->GetDiffuse();
    glm::vec3 specular = pointLight->GetSpecular();
    glm::vec3 color = pointLight->GetColor();
    float constant = pointLight->GetConstant();
    float linear = pointLight->GetLinear();
    float quadratic = pointLight->GetQuadratic();

    if (ShowVec3("Ambient", ambient, 0.1f, 0.0f, 0.0f)) {
        pointLight->SetAmbient(ambient);
    }
    if (ShowVec3("Diffuse", diffuse, 0.1f, 0.0f, 0.0f)) {
        pointLight->SetDiffuse(diffuse);
    }
    if (ShowVec3("Specular", specular, 0.1f, 0.0f, 0.0f)) {
        pointLight->SetSpecular(specular);
    }
    if (ShowVec3("Color", color, 0.01f, 0.0f, 0.0f, 1.0f)) {
        pointLight->SetColor(color);
    }
    if (ShowFloat("Constant", &constant, 0.0001f, 0.0f, 0.0f)) {
        pointLight->SetConstant(constant);
    }
    if (ShowFloat("Linear", &linear, 0.0001f, 0.0f, 0.0f)) {
        pointLight->SetLinear(linear);
    }
    if (ShowFloat("Quadratic", &quadratic, 0.0001f, 0.0f, 0.0f)) {
        pointLight->SetQuadratic(quadratic);
    }

    ImGui::EndTable();
}

void Inspector::ShowSpotLight() {
    SpotLight* spotLight = EditorManager::GetInstance()->selectedNode->GetComponentByClass<SpotLight>();

    if (!ShowComponentHeader(spotLight, "Spot Light")) {
        return;
    }

    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        ImGui::OpenPopup("SpotLightContextMenu");
    }
    if (ImGui::BeginPopup("SpotLightContextMenu", ImGuiPopupFlags_NoOpenOverExistingPopup)) {
        if (ImGui::MenuItem("Reset")) {
            spotLight->SetAmbient({0.4f, 0.4f, 0.4f});
            spotLight->SetDiffuse({0.69f, 0.69f, 0.69f});
            spotLight->SetSpecular({0.9f, 0.9f, 0.9f});
            spotLight->SetColor({1.0f, 1.0f, 1.0f});
            spotLight->SetConstant(1.0f);
            spotLight->SetLinear(0.045f);
            spotLight->SetQuadratic(0.0075f);
            spotLight->SetCutOff(12.5f);
            spotLight->SetOuterCutOff(15.0f);

            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Remove")) {
            Component::Destroy(spotLight);

            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    float availableWidth = ImGui::GetContentRegionAvail().x;
    ImGui::BeginTable("Spot Light", 2);
    ImGui::TableSetupColumn("##Col1", ImGuiTableColumnFlags_WidthFixed, availableWidth * 0.33f);

    glm::vec3 ambient = spotLight->GetAmbient();
    glm::vec3 diffuse = spotLight->GetDiffuse();
    glm::vec3 specular = spotLight->GetSpecular();
    glm::vec3 color = spotLight->GetColor();
    float constant = spotLight->GetConstant();
    float linear = spotLight->GetLinear();
    float quadratic = spotLight->GetQuadratic();
    float cutOff = spotLight->GetCutOff();
    float outerCutOff = spotLight->GetOuterCutOff();

    if (ShowVec3("Ambient", ambient, 0.1f, 0.0f, 0.0f)) {
        spotLight->SetAmbient(ambient);
    }
    if (ShowVec3("Diffuse", diffuse, 0.1f, 0.0f, 0.0f)) {
        spotLight->SetDiffuse(diffuse);
    }
    if (ShowVec3("Specular", specular, 0.1f, 0.0f, 0.0f)) {
        spotLight->SetAmbient(specular);
    }
    if (ShowVec3("Color", color, 0.01f, 0.0f, 0.0f, 1.0f)) {
        spotLight->SetColor(color);
    }
    if (ShowFloat("Constant", &constant, 0.0001f, 0.0f, 0.0f)) {
        spotLight->SetConstant(constant);
    }
    if (ShowFloat("Linear", &linear, 0.0001f, 0.0f, 0.0f)) {
        spotLight->SetLinear(linear);
    }
    if (ShowFloat("Quadratic", &quadratic, 0.0001f, 0.0f, 0.0f)) {
        spotLight->SetQuadratic(quadratic);
    }
    if (ShowFloat("Cut Off", &cutOff, 0.01f, 0.0f, 0.0f, 360.0f)) {
        spotLight->SetAmbient(color);
    }
    if (ShowFloat("Outer Cut Off", &outerCutOff, 0.01f, 0.0f, 0.0f, 360.0f)) {
        spotLight->SetOuterCutOff(outerCutOff);
    }

    ImGui::EndTable();
}

void Inspector::ShowSkybox() {
    Skybox* skybox = EditorManager::GetInstance()->selectedNode->GetComponentByClass<Skybox>();

    if (!ShowComponentHeader(skybox, "Skybox")) {
        return;
    }

    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        ImGui::OpenPopup("SkyboxContextMenu");
    }
    if (ImGui::BeginPopup("SkyboxContextMenu", ImGuiPopupFlags_NoOpenOverExistingPopup)) {
        if (ImGui::MenuItem("Set Active")) {
            Skybox::SetActiveSkybox(EditorManager::GetInstance()->selectedNode);

            ImGui::CloseCurrentPopup();
        }

        if (ImGui::MenuItem("Remove")) {
            Component::Destroy(skybox);

            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    float availableWidth = ImGui::GetContentRegionAvail().x;
    ImGui::BeginTable("Skybox", 2);
    ImGui::TableSetupColumn("##Col1", ImGuiTableColumnFlags_WidthFixed, availableWidth * 0.33f);

    std::string cubeMapWalls[6] = {"Right", "Left", "Top", "Bottom", "Front", "Back"};
    CubeMap* cubeMap = skybox->GetCubeMap();

    for (int i = 0; i < 6; ++i) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("%s", (cubeMapWalls[i] + " Texture").c_str());
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (cubeMap)
            ImGui::Text("%s", cubeMap->textures[i].substr(cubeMap->textures[i].find_last_of('/') + 1).c_str());
        else
            ImGui::Text("No texture");

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
                cubeMap->textures[i] = payloadData;
                cubeMap->Reload();
            }

            ImGui::EndDragDropTarget();
        }
    }

    ImGui::EndTable();
}
#pragma endregion

bool Inspector::ShowComponentHeader(Component *component, const std::string& headerText) {
    float availableWidth = ImGui::GetContentRegionAvail().x;

    ImGui::BeginTable("Header", 2);
    ImGui::TableSetupColumn("##Col1", ImGuiTableColumnFlags_WidthFixed, availableWidth * 0.90f);

    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    bool isOpen = ImGui::CollapsingHeader(headerText.c_str(), HEADER_FLAGS);

    ImGui::TableSetColumnIndex(1);
    ImGui::SetNextItemWidth(-FLT_MIN);

    bool enabled = component->GetEnabled();

    if (ImGui::Checkbox("##enabled", &enabled)) {
        if (enabled) {
            component->SetEnabled(true);
        }
        else {
            component->SetEnabled(false);
        }
    }

    ImGui::EndTable();

    return isOpen;
}

bool Inspector::ShowVec3(const char *label, glm::vec3& vec3, float speed, float resetValue, float minValue, float maxValue) {
    bool changed = false;
    bool openContextMenu = false;

    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("%s", label);
    if (ImGui::IsItemHovered()) {
        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            vec3 = glm::vec3(resetValue);
            changed = true;
        }
        else if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
            openContextMenu = true;
        }
    }
    ImGui::TableSetColumnIndex(1);
    ImGui::SetNextItemWidth(-FLT_MIN);
    changed |= ImGui::DragFloat3(label, &vec3.x, speed, minValue, maxValue);

    if (openContextMenu) {
        ImGui::OpenPopup(label);
    }

    if (ImGui::BeginPopup(label)) {
        if (ImGui::MenuItem("Copy")) {
            copiedVec3 = vec3;
        }
        if (ImGui::MenuItem("Paste")) {
            vec3 = copiedVec3;
            changed = true;
        }

        ImGui::EndPopup();
    }

    return changed;
}

bool Inspector::ShowVec2(const char *label, glm::vec2& vec2, float speed, float resetValue, float minValue, float maxValue) {
    bool changed = false;
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("%s", label);
    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        vec2 = glm::vec2(resetValue);
        changed = true;
    }
    ImGui::TableSetColumnIndex(1);
    ImGui::SetNextItemWidth(-FLT_MIN);
    changed |= ImGui::DragFloat2(label, &vec2.x, speed, minValue, maxValue);
    return changed;
}

bool Inspector::ShowFloat(const char *label, float* value, float speed, float resetValue, float minValue, float maxValue) {
    bool changed = false;
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("%s", label);
    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        value = &resetValue;
        changed = true;
    }
    ImGui::TableSetColumnIndex(1);
    ImGui::SetNextItemWidth(-FLT_MIN);
    changed |= ImGui::DragFloat(label, value, speed, minValue, maxValue);
    return changed;
}
