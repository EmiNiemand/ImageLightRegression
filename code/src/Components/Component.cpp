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
