#include "Managers/ResourceManager.h"

ResourceManager::ResourceManager() = default;

ResourceManager::~ResourceManager() = default;

ResourceManager *ResourceManager::GetInstance() {
    if (resourceManager == nullptr) {
        resourceManager = new ResourceManager();
    }
    return resourceManager;
}

void ResourceManager::StartUp() {

}

void ResourceManager::ShutDown() {
    resources.clear();
    delete resourceManager;
}

void ResourceManager::UnloadResource(const std::string &path) {
#ifdef DEBUG
    if (!resources.contains(path))ILR_ERROR_MSG("Given resource does no longer exist");
#endif

    if (resources.contains(path)) {
        SResource* resource = &resources.find(path)->second;
        --resource->resourceCounter;
        if(!resource->resourceCounter) {
            delete resource->resource;
            resources.erase(path);
        }
    }
}
