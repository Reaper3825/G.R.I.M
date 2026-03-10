#pragma once

#ifndef FLASH_NAMESPACE_CONFIG_H
#define FLASH_NAMESPACE_CONFIG_H

// Compatibility header for flash-attention v2.5.9 (before namespace_config.h was added upstream).
// Provides FLASH_NAMESPACE macros. Define FLASH_NAMESPACE before including this header.

#ifndef FLASH_NAMESPACE
#define FLASH_NAMESPACE flash
#endif

#define FLASH_NAMESPACE_ALIAS(name) FLASH_NAMESPACE::name

#define FLASH_NAMESPACE_SCOPE(content)                                         \
  namespace FLASH_NAMESPACE {                                                  \
  content                                                                      \
  }

#endif  // FLASH_NAMESPACE_CONFIG_H
