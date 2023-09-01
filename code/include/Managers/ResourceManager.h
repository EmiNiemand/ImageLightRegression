#ifndef IMAGELIGHTREGRESSION_RESOURCEMANAGER_H
#define IMAGELIGHTREGRESSION_RESOURCEMANAGER_H

#include "Core/Resource.h"
#include "ApplicationTypes.h"
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

    void Startup();
    void Shutdown();

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

        SResource& resource = resources.find(path)->second;
        ++resource.resourceCounter;

        return (T*)resource.resource;
    };

    static void UnloadResource(std::string path);

private:
    explicit ResourceManager();
};


#endif //IMAGELIGHTREGRESSION_RESOURCEMANAGER_H
