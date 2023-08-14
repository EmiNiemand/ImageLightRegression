#pragma once
#include "spdlog/spdlog.h"

#define ILR_ERROR_MSG(message)        \
    spdlog::error(message)

#define ILR_INFO_MSG(message)        \
    spdlog::info(message)

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
