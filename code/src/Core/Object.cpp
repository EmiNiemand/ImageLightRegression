#include <utility>

#include "Core/Object.h"
#include "Application.h"
#include "Factories/ObjectFactory.h"
#include "Components/Transform.h"

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
    parent->children.erase(id);
    parent = newParent;
    newParent->AddChild(this);
}

void Object::AddChild(Object* child) {
    child->parent = this;
    children.insert({child->id, child});
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

    if (dirtyFlag) ForceUpdateSelfAndChildren();

    for (int i = 0; i < checkIterator; ++i) {
        for (const auto& child : toCheck[i]->children) {
            if (child.second->dirtyFlag) {
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
        component.second->enabled = true;
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
        component.second->enabled = false;
    }
    enabled = false;
}

void Object::RecalculateGlobalRotation() {
    globalRotation = parent->globalRotation + transform->GetLocalRotation();

    for (auto&& child : children) {
        child.second->RecalculateGlobalRotation();
    }
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
