#include "Core/Resource.h"
#include "Macros.h"

Resource::Resource(std::string inPath) : path(std::move(inPath)) {
    if (!exists(std::filesystem::path(path))) {
        ILR_ASSERT_MSG(exists(std::filesystem::path(path)), "Path does not exist: " + path);
    }
}

Resource::~Resource() = default;

const std::string &Resource::GetPath() {
    return path;
}
