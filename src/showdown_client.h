#ifndef SHOWDOWN_CLIENT_H
#define SHOWDOWN_CLIENT_H

struct ShowdownClient;

// Create a client configured to connect to a given WS endpoint
struct ShowdownClient* showdown_client_create(const char* host, int port, const char* path);

// Run the event loop until disconnect or control c, returns 0 on clean exit
int showdown_client_run(struct ShowdownClient* cli);

// Destroy and free resources, safe to call with NULL
void showdown_client_destroy(struct ShowdownClient* cli);

#endif

