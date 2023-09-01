#ifndef IMAGELIGHTREGRESSION_OBJECT_H
#define IMAGELIGHTREGRESSION_OBJECT_H

#include "Factories/ComponentFactory.h"
#include "Core/Interfaces/ISerializable.h"

#include <string>
#include <unordered_map>

#include "glm/glm.hpp"

class Component;
class Transform;

class Object : public ISerializable {
    friend class Application;
    friend class ObjectFactory;
    friend class EditorCamera;
public:
    std::unordered_map<int, Component*> components;
    std::unordered_map<int, Object*> children;
    std::string name;

    Transform* transform;

    int id;
    bool visibleInEditor = true;

private:
    Object* parent;

    bool enabled = true;

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
    [[nodiscard]] Object* GetParent() const;
    [[nodiscard]] bool GetEnabled() const;

    void AddChild(Object* child);
    void RemoveChild(int childId);
    void RemoveAllChildren();

    void UpdateSelfAndChildren();
    void EnableSelfAndChildren();
    void DisableSelfAndChildren();

    void Save(nlohmann::json& json) override;
    void Load(nlohmann::json& json) override;

private:
    Object(std::string name, Object *parent, int id);
    void DestroyAllComponents();
    void DestroyAllChildren();
    void ForceUpdateSelfAndChildren();
    void OnTransformUpdateComponents();
};


#endif //IMAGELIGHTREGRESSION_OBJECT_H
