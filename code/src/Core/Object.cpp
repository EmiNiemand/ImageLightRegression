#include "Core/Object.h"
#include "Application.h"
#include "Factories/ObjectFactory.h"
#include "Components/ComponentHeaders.h"

#include <utility>

Object::Object(std::string name, Object *parent, int id) : name(std::move(name)), parent(parent), id(id) {
    transform = this->AddComponent<Transform>();
}

Object::~Object() = default;

Object *Object::Instantiate(const std::string& name, Object *parent) {
    return ObjectFactory::GetInstance()->CreateObject(name, parent);
}

void Object::Destroy(Object* object) {
    object->DestroyAllComponents();
    object->DestroyAllChildren();
    Application::GetInstance()->AddObjectToDestroyBuffer(object->id);
}

void Object::SetParent(Object *newParent) {
    if (this == Application::GetInstance()->scene) {
        return;
    }
    glm::vec3 globalPosition = transform->GetGlobalPosition();
    parent->children.erase(id);
    parent = newParent;
    parent->AddChild(this);
    transform->SetLocalPosition(globalPosition - parent->transform->GetGlobalPosition());
}

void Object::AddChild(Object* child) {
    glm::vec3 globalPosition = child->transform->GetGlobalPosition();
    if (child->parent)
        child->parent->children.erase(child->id);
    child->parent = this;
    children.insert({child->id, child});
    child->transform->SetLocalPosition(globalPosition - transform->GetGlobalPosition());
}

void Object::RemoveChild(int childId) {
    if (!children.contains(childId)) return;
    Destroy(children.find(childId)->second);
}

void Object::RemoveAllChildren() {
    if (children.empty()) return;
    for (auto&& child : children) {
        Destroy(child.second);
    }
}

void Object::UpdateSelfAndChildren() {
    const int objectsSize = (int)Application::GetInstance()->objects.size() + 1;

    auto toCheck = new Object*[objectsSize];

    int checkIterator = 1;

    toCheck[0] = this;

    if (transform->GetDirtyFlag()) ForceUpdateSelfAndChildren();

    for (int i = 0; i < checkIterator; ++i) {
        for (const auto& child : toCheck[i]->children) {
            if (child.second->transform->GetDirtyFlag()) {
                child.second->ForceUpdateSelfAndChildren();
            }
            toCheck[checkIterator] = child.second;
            ++checkIterator;
        }
    }

    delete[] toCheck;
}

void Object::EnableSelfAndChildren() {
    if (enabled) return;

    for (auto&& component : components){
        component.second->SetEnabled(true);
    }
    for (auto&& child : children)
    {
        child.second->EnableSelfAndChildren();
    }
    enabled = true;
}

void Object::DisableSelfAndChildren() {
    if (!enabled) return;

    for (auto&& child : children)
    {
        child.second->DisableSelfAndChildren();
    }
    for (auto&& component : components){
        component.second->SetEnabled(false);
    }
    enabled = false;
}

void Object::DestroyAllComponents() {
    if (components.empty()) return;
    for (const auto& component : components) {
        Component::Destroy(component.second);
    }
    components.clear();
}

void Object::DestroyAllChildren() {
    if (children.empty()) return;

    for (const auto& child : children) {
        Object::Destroy(child.second);
    }
}

void Object::ForceUpdateSelfAndChildren() {
    if (parent == nullptr) {
        transform->ComputeModelMatrix();
    }
    else {
        transform->ComputeModelMatrix(parent->transform->GetModelMatrix());
    }
    OnTransformUpdateComponents();
}

void Object::OnTransformUpdateComponents() {
    for (auto& component : components) {
        component.second->OnUpdate();
    }
}

bool Object::GetEnabled() const {
    return enabled;
}

Object *Object::GetParent() const {
    return parent;
}

void Object::Save(nlohmann::json& json) {
    json["Name"] = name;
    json["VisibleInEditor"] = visibleInEditor;
    json["Enabled"] = enabled;

    transform->Save(json["Transform"]);

    json["Children"] = nlohmann::json::array();
    for (auto const &object : children) {
        if (!object.second->visibleInEditor) continue;
        json["Children"].push_back(nlohmann::json::object());
        object.second->Save(json["Children"].back());
    }

    json["Components"] = nlohmann::json::array();
    for (auto const &component : components) {
        json["Components"].push_back(nlohmann::json::object());
        component.second->Save(json["Components"].back());
    }
}

void Object::Load(nlohmann::json& json) {
    visibleInEditor = json["VisibleInEditor"];
    enabled = json["Enabled"];

    transform->Load(json["Transform"]);

    for (auto& child : json["Children"]) {
        Instantiate(child["Name"], this)->Load(child);
    }

    for (auto& component : json["Components"]) {
        if (component.contains("ComponentType")) {
            if (component["ComponentType"] == "DirectionalLight") {
                AddComponent<DirectionalLight>()->Load(component);
            }
            if (component["ComponentType"] == "PointLight") {
                AddComponent<PointLight>()->Load(component);
            }
            if (component["ComponentType"] == "SpotLight") {
                AddComponent<SpotLight>()->Load(component);
            }
            if (component["ComponentType"] == "Image") {
                AddComponent<Image>()->Load(component);
            }
            if (component["ComponentType"] == "Camera") {
                AddComponent<Camera>()->Load(component);
            }
            if (component["ComponentType"] == "EditorCamera") {
                AddComponent<EditorCamera>()->Load(component);
            }
            if (component["ComponentType"] == "Renderer") {
                AddComponent<Renderer>()->Load(component);
            }
            if (component["ComponentType"] == "Skybox") {
                AddComponent<Skybox>()->Load(component);
            }
        }
    }
}
