// #include <stdio.h>
// #include <stdlib.h>
// #include "pokemon.h"
#include "mongoose.h"

// Event handler called on new HTTP message
static void event_handler(struct mg_connection *c, int ev, void *ev_data) {
  if (ev == MG_EV_HTTP_MSG) {
    // Send a plain-text response
    mg_http_reply(c, 200, "Content-Type: text/plain\r\n", "Hello, World!\n");
  }
}

int main(void) {
  struct mg_mgr mgr;
  mg_mgr_init(&mgr);  // Initialize event manager

  // Start listening on port 8000
  if (mg_http_listen(&mgr, "http://0.0.0.0:8000", event_handler, NULL) == NULL) {
    fprintf(stderr, "Failed to open listener on port 8000\n");
    return 1;
  }

  printf("HTTP server listening on http://localhost:8000\n");

  // Event loop
  for (;;) {
    mg_mgr_poll(&mgr, 1000);
  }

  mg_mgr_free(&mgr);  // cleanup (never reached)
  return 0;
}