#include "Components/Transform.h"
#include "Application.h"
#include "Core/Object.h"

Transform::Transform(Object *parent, int id) : Component(parent, id) {}

Transform::~Transform() = default;

glm::mat4 Transform::GetLocalModelMatrix() const {
    const glm::mat4 transformX = glm::rotate(glm::mat4(1.0f), glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
    const glm::mat4 transformY = glm::rotate(glm::mat4(1.0f), glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::mat4 transformZ = glm::rotate(glm::mat4(1.0f), glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

    // Y * X * Z
    const glm::mat4 rotationMatrix = transformY * transformX * transformZ;

    // translation * rotation * scale (also known as TRS matrix)
    return glm::translate(glm::mat4(1.0f), position) * rotationMatrix * glm::scale(glm::mat4(1.0f), scale);
}

void Transform::ComputeModelMatrix() {
    mModelMatrix = GetLocalModelMatrix();
    dirtyFlag = false;
}

void Transform::ComputeModelMatrix(const glm::mat4& parentGlobalModelMatrix) {
    mModelMatrix = parentGlobalModelMatrix * GetLocalModelMatrix();
    dirtyFlag = false;
}

void Transform::SetLocalPosition(const glm::vec3& newPosition) {
    position = newPosition;
    if (!parent) return;
    if (dirtyFlag) return;
    SetDirtyFlag();
}

void Transform::SetLocalRotation(const glm::vec3& newRotation) {
    rotation = newRotation;
    if (!parent->GetParent()) return;
    CalculateGlobalRotation();
    if (dirtyFlag) return;
    SetDirtyFlag();
}

void Transform::SetLocalScale(const glm::vec3& newScale) {
    scale = newScale;
    if (!parent) return;
    if (dirtyFlag) return;
    SetDirtyFlag();
}

void Transform::SetDirtyFlag() {
    const int gameObjectsSize = (int)Application::GetInstance()->objects.size() + 1;
    auto toSet = new Object*[gameObjectsSize];

    int checkIterator = 1;
    toSet[0] = parent;

    for (int i = 0; i < checkIterator; i++) {
        toSet[i]->transform->dirtyFlag = true;
        for (const auto& child : toSet[i]->children) {
            toSet[checkIterator] = child.second;
            checkIterator++;
        }
    }

    delete[] toSet;
}

glm::vec3 Transform::GetGlobalPosition() const {
    return mModelMatrix[3];
}

glm::vec3 Transform::GetLocalPosition() const {
    return position;
}

glm::vec3 Transform::GetLocalRotation() const {
    return rotation;
}
glm::vec3 Transform::GetGlobalRotation() const {
    return globalRotation;
}

glm::vec3 Transform::GetLocalScale() const {
    return scale;
}

glm::vec3 Transform::GetGlobalScale() const {
    return {glm::length(GetRight()), glm::length(GetUp()), glm::length(GetBackward()) };
}

glm::mat4 Transform::GetModelMatrix() const {
    return mModelMatrix;
}

glm::vec3 Transform::GetRight() const {
    return mModelMatrix[0];
}


glm::vec3 Transform::GetUp() const {
    return mModelMatrix[1];
}

glm::vec3 Transform::GetBackward() const {
    return mModelMatrix[2];
}

glm::vec3 Transform::GetForward() const {
    return -mModelMatrix[2];
}

bool Transform::GetDirtyFlag() {
    return dirtyFlag;
}

void Transform::CalculateGlobalRotation() {
    globalRotation = parent->GetParent()->transform->globalRotation + rotation;

    for (auto&& child : parent->children) {
        child.second->transform->CalculateGlobalRotation();
    }
}

void Transform::Save(nlohmann::json &json) {
    Component::Save(json);

    json["Position"] = nlohmann::json::array();
    json["Position"].push_back(position.x);
    json["Position"].push_back(position.y);
    json["Position"].push_back(position.z);

    json["Rotation"] = nlohmann::json::array();
    json["Rotation"].push_back(rotation.x);
    json["Rotation"].push_back(rotation.y);
    json["Rotation"].push_back(rotation.z);

    json["Scale"] = nlohmann::json::array();
    json["Scale"].push_back(scale.x);
    json["Scale"].push_back(scale.y);
    json["Scale"].push_back(scale.z);
}

void Transform::Load(nlohmann::json &json) {
    Component::Load(json);

    SetLocalPosition({json["Position"][0], json["Position"][1], json["Position"][2]});
    SetLocalRotation({json["Rotation"][0], json["Rotation"][1], json["Rotation"][2]});
    SetLocalScale({json["Scale"][0], json["Scale"][1], json["Scale"][2]});
}
