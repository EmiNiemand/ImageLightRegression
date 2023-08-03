#ifndef IMAGELIGHTREGRESSION_COMPONENTFACTORY_H
#define IMAGELIGHTREGRESSION_COMPONENTFACTORY_H

#include "Application.h"

class Object;
class Camera;
class Spotlight;
class Pointlight;

class ComponentFactory {
private:
    int id = 0;
    inline static ComponentFactory* componentFactory;

public:
    virtual ~ComponentFactory();

    ComponentFactory(ComponentFactory &other) = delete;
    void operator=(const ComponentFactory&) = delete;

    static ComponentFactory* GetInstance();

    template<class T>
    T* CreateComponent(Object* parent) {

        id++;
        T* component = new T(parent, id);
        Application::GetInstance()->components.insert({component->id, component});
        return component;
    };

private:
    explicit ComponentFactory();
};


#endif //IMAGELIGHTREGRESSION_COMPONENTFACTORY_H
