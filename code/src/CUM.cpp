#include "CUM.h"

uint64 CUM::Hash(const std::string &path) {
    std::hash<std::string> hash;
    return hash(path);
}
