#ifndef IMAGELIGHTREGRESSION_OBJECT_H
#define IMAGELIGHTREGRESSION_OBJECT_H

#include "Factories/ComponentFactory.h"

#include <string>
#include <unordered_map>

#include "glm/glm.hpp"

class Component;
class Transform;

class Object {
    friend class Application;
    friend class ObjectFactory;
    friend class EditorCamera;
public:
    std::unordered_map<int, Component*> components;
    std::unordered_map<int, Object*> children;
    std::string name;

    glm::vec3 globalRotation = {0, 0, 0};

    Transform* transform;

    int id;
    bool dirtyFlag = true;
    bool enabled = true;
    bool showInEditor = true;

private:
    Object* parent;

public:
    virtual ~Object();

    static Object* Instantiate(const std::string& name, Object* parent = nullptr);
    static void Destroy(Object* object);

    template<class T>
    T* AddComponent() {
        T* component = ComponentFactory::GetInstance()->CreateComponent<T>(this);
        components.insert({component->id, component});
        component->OnCreate();
        return component;
    };

    template<class T>
    T* GetComponentByClass() {
        for (auto& component : components) {
            if (dynamic_cast<T*>(component.second) != nullptr) {
                return (T*)(component.second);
            }
        }
        return nullptr;
    };

    template<class T>
    std::vector<T*> GetComponentsByClass() {
        std::vector<T*> componentsOfClass;
        for (auto& component : components) {
            if (dynamic_cast<T*>(component.second) != nullptr) {
                componentsOfClass.push_back((T*)(component.second));
            }
        }
        return componentsOfClass;
    };


    void SetParent(Object* newParent);
    void AddChild(Object* child);
    void RemoveChild(int childId);
    void RemoveAllChildren();

    void UpdateSelfAndChildren();
    void EnableSelfAndChildren();
    void DisableSelfAndChildren();

    void RecalculateGlobalRotation();

private:
    Object(std::string name, Object *parent, int id);
    void DestroyAllComponents();
    void DestroyAllChildren();
    void ForceUpdateSelfAndChildren();
    void OnTransformUpdateComponents();
};


#endif //IMAGELIGHTREGRESSION_OBJECT_H
