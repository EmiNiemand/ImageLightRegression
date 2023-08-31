#include "CUM.h"
#include "Application.h"

uint64 CUM::Hash(const std::string &path) {
    std::hash<std::string> hash;
    return hash(path);
}

bool CUM::IsInViewport(glm::ivec2 position, Viewport* viewport) {
    return position.x >= viewport->position.x &&
    position.x <= viewport->position.x + viewport->resolution.x &&
    position.y <= Application::resolution.y - viewport->position.y &&
    position.y >= Application::resolution.y - (viewport->position.y + viewport->resolution.y);
}
