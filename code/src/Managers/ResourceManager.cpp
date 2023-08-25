#include "Managers/ResourceManager.h"

ResourceManager::ResourceManager() = default;

ResourceManager::~ResourceManager() = default;

ResourceManager* ResourceManager::GetInstance() {
    if (resourceManager == nullptr) {
        resourceManager = new ResourceManager();
    }
    return resourceManager;
}

void ResourceManager::Startup() {
}

void ResourceManager::Shutdown() {
    std::vector<std::string> paths;
    paths.reserve(resources.size());

    for (const auto& resource : resources) {
        paths.push_back(resource.second.resource->GetPath());
    }

    for (const auto& path : paths) {
        UnloadResource(path);
    }

    resources.clear();
    paths.clear();
    delete resourceManager;
}

void ResourceManager::UnloadResource(std::string path) {
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
