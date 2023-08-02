#ifndef IMAGELIGHTREGRESSION_OBJECTFACTORY_H
#define IMAGELIGHTREGRESSION_OBJECTFACTORY_H

#include <string>

class Object;

class ObjectFactory {
private:
    int id = 0;
    inline static ObjectFactory* objectFactory = nullptr;

public:
    virtual ~ObjectFactory();

    ObjectFactory(ObjectFactory &other) = delete;
    void operator=(const ObjectFactory&) = delete;
    Object* CreateObject(const std::string& name, Object* parent = nullptr);

    static ObjectFactory* GetInstance();

private:
    explicit ObjectFactory();
};


#endif //IMAGELIGHTREGRESSION_OBJECTFACTORY_H
