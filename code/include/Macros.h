#pragma once
#include "spdlog/spdlog.h"
#include "effolkronium/random.hpp"

#define ILR_ERROR_MSG(message)        \
    spdlog::error(message)

#define ILR_INFO_MSG(message)        \
    spdlog::info(message)

#define ILR_WARN_MSG(message)        \
    spdlog::warn(message)

#define ILR_ASSERT_MSG(condition, message)                                     \
    do {                                                                       \
        if (!(condition)) {                                                    \
            spdlog::error("Assertion: " #condition ", failed: " message);      \
            std::terminate();                                                  \
        }                                                                      \
    } while (false)

#define ILR_ASSERT(condition)                                      \
    do {                                                           \
        if (!(condition)) {                                        \
            spdlog::error("Assertion " + #condition + " failed");  \
            std::terminate();                                      \
        }                                                          \
    } while (false)

#define ILR_UNIMPLEMENTED                            \
    do {                                             \
        spdlog::error("Unimplemented Assertion");    \
        std::terminate();                            \
    } while (false)

#define ILR_UNIMPLEMENTED_SOFT             \
    do {                                   \
        spdlog::error("Unimplemented");    \
    } while (false)

#define RNG(min, max) effolkronium::random_static::get(min, max)

#define STRING(val) std::format("{:g}", (float)val)
#define STRING_VEC2(val) STRING(val.x) + ", " + STRING(val.y)
#define STRING_VEC3(val) STRING(val.x) + ", " + STRING(val.y) + ", " + STRING(val.z)

#define DELETE_VECTOR_VALUES(vector)                                  \
    for (int iterator = 0; iterator < vector.size(); ++iterator) {    \
         delete vector[iterator];                                     \
    }                                                                 \
    vector.clear();