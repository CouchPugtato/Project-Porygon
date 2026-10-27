#include "artifact_io.h"
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif

uint64_t artifact_fingerprint_bytes(uint64_t hash, const void* data, size_t count) {
    const unsigned char* bytes = data;
    size_t i;
    for (i = 0; i < count; ++i) { hash ^= bytes[i]; hash *= UINT64_C(1099511628211); }
    return hash;
}

int artifact_publish_file(const char* temporary, const char* path) {
#ifdef _WIN32
    return MoveFileExA(temporary, path, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return rename(temporary, path) == 0;
#endif
}

int artifact_finish_file(FILE* file) {
    int ok = fflush(file) == 0;
#ifdef _WIN32
    if (ok) ok = _commit(_fileno(file)) == 0;
#else
    if (ok) ok = fsync(fileno(file)) == 0;
#endif
    if (fclose(file) != 0) ok = 0;
    return ok;
}
