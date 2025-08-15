#include "showdown_client.h"
#include <stdio.h>

int main(void) {
    const char* host = "sim3.psim.us"; // Showdown sim host (rotates)
    const int   port = 443; // TLS (wss://)
    const char* path = "/showdown/websocket"; // Raw WS endpoint

    struct ShowdownClient* cli = showdown_client_create(host, port, path);
    if (!cli) {
        fprintf(stderr, "Failed to create ShowdownClient\n");
        return 1;
    }

    int rc = showdown_client_run(cli);
    showdown_client_destroy(cli);
    return rc;
}

