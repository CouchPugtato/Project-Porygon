#ifndef ARTIFACT_IO_H
#define ARTIFACT_IO_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

/* Native-format corruption detection, not a cryptographic provenance hash. */
uint64_t artifact_fingerprint_bytes(uint64_t hash, const void* data, size_t count);
int artifact_finish_file(FILE* file);
int artifact_publish_file(const char* temporary, const char* path);

#endif
