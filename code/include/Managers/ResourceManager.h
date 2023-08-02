#ifndef IMAGELIGHTREGRESSION_RESOURCEMANAGER_H
#define IMAGELIGHTREGRESSION_RESOURCEMANAGER_H

#include "ApplicationTypes.h"
#include "Core/Resource.h"
#include "Macros.h"

#include <unordered_map>
#include <string>

struct SResource {
    uint64 resourceCounter;
    Resource* resource;
};

class ResourceManager {
private:
    inline static ResourceManager* resourceManager;
    inline static std::unordered_map<std::string, SResource> resources;

public:
    ResourceManager(ResourceManager &other) = delete;
    virtual ~ResourceManager();
    void operator=(const ResourceManager&) = delete;

    static ResourceManager* GetInstance();

    void StartUp();
    void ShutDown();

    template<typename T>
    static T* LoadResource(const std::string& path) {
        ILR_ASSERT_MSG((std::is_base_of<Resource, T>::value), "Resource of given path does not derived from Resource Class");

        if (!resources.contains(path)) {
            T* resource = new T(path);
            resource->Load();
            SResource sResource = {1, resource};
            resources.insert({path, sResource});
            return resource;
        }

        return (T*)resources.find(path)->second.resource;
    };

    void UnloadResource(const std::string& path);

private:
    ResourceManager();
};


#endif //IMAGELIGHTREGRESSION_RESOURCEMANAGER_H
