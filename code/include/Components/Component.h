#ifndef IMAGELIGHTREGRESSION_COMPONENT_H
#define IMAGELIGHTREGRESSION_COMPONENT_H

class Object;

class Component {
public:
    Object* parent;
    int id;

    bool callOnAwake = true;
    bool callOnStart = true;
    bool enabled = true;

public:
    Component(Object *parent, int id);
    virtual ~Component() = 0;

    static void Destroy(Component* component);

    /// OnCreate called when Component is created, mainly use to subscribe component to managers
    inline virtual void OnCreate() {};
    /// OnDestroy called when Component is destroyed, mainly use to unsubscribe component from managers
    virtual void OnDestroy();
    /// Called if parent transform was changed
    inline virtual void OnUpdate() {};

    inline virtual void Awake() {};
    inline virtual void Start() {};
    inline virtual void Update() {};

};


#endif //IMAGELIGHTREGRESSION_COMPONENT_H
