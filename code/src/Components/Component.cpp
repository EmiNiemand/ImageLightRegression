#include "Components/Component.h"
#include "Application.h"
#include "Core/Object.h"

Component::Component(Object* parent, int id) : parent(parent), id(id) {}

Component::~Component() = default;

void Component::Destroy(Component* component) {
    Application::GetInstance()->AddComponentToDestroyBuffer(component->id);
}

void Component::OnDestroy() {
    parent->components.erase(id);
}

bool Component::GetEnabled() const {
    return enabled;
}

void Component::SetEnabled(bool inEnabled) {
    enabled = inEnabled;
    OnUpdate();
}

void Component::Save(nlohmann::json &json) {
    json["CallOnAwake"] = callOnAwake;
    json["CallOnStart"] = callOnStart;
    json["Enabled"] = enabled;
}

void Component::Load(nlohmann::json &json) {
    callOnAwake = json["CallOnAwake"];
    callOnStart = json["CallOnStart"];
    enabled = json["Enabled"];

    OnUpdate();
}
