#include "Factories/ObjectFactory.h"
#include "Managers/SceneManager.h"
#include "Core/Object.h"

ObjectFactory::ObjectFactory() = default;

ObjectFactory::~ObjectFactory() {
    delete objectFactory;
}

ObjectFactory* ObjectFactory::GetInstance() {
    if (objectFactory == nullptr) {
        objectFactory = new ObjectFactory();
    }
    return objectFactory;
}

Object* ObjectFactory::CreateObject(const std::string& name, Object* parent) {
    Object* object = new Object(name, parent, id);
    Application::GetInstance()->objects.insert({id, object});
    if (SceneManager::GetInstance()->scene == nullptr) {
        ++id;
        return object;
    }
    if (parent == nullptr) {
        SceneManager::GetInstance()->scene->AddChild(object);
    }
    else {
        parent->AddChild(object);
    }
    ++id;
    return object;
}
