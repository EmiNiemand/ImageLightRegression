#include "Core/Resource.h"
#include "Managers/ResourceManager.h"
#include "Macros.h"

Resource::Resource(std::string inPath) : path(std::move(inPath)){
    if (!exists(std::filesystem::path(path))) {
        ILR_ASSERT_MSG(exists(std::filesystem::path(path)), "Path does not exist: " + path);
    }
}

Resource::~Resource() {
    Unload();
}

void Resource::Unload() {
    ResourceManager::GetInstance()->UnloadResource(path);
}

const std::string &Resource::GetPath() {
    return path;
}
