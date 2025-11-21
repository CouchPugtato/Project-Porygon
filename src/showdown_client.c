#include "showdown_client.h"
#include "game_state.h"
#include <libwebsockets.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <signal.h>

#define MAX_LINE 900

struct scs {
    int logged_in;
    char username[32];
    struct {
        size_t len;
        char* data;
    } out;
    struct BattleState bs;
    char current_room[64];
};

struct ShowdownClient {
    struct lws_context *ctx;
    struct lws_client_connect_info cc;
    struct lws_protocols protocols[2];
};

static volatile int g_quit = 0;
static void sigint_handler(int sig) { (void)sig; g_quit = 1; }

static char* build_line(const char* room, const char* msg, size_t* out_len) {
    char tmp[MAX_LINE];
    int n = room && room[0] ? snprintf(tmp, sizeof tmp, "%s|%s", room, msg)
        					: snprintf(tmp, sizeof tmp, "|%s", msg);
    if (n < 0 || (size_t)n >= sizeof tmp) return NULL;
    char* s = malloc((size_t)n);
    if (!s) return NULL;
    memcpy(s, tmp, (size_t)n);
    *out_len = (size_t)n;
    return s;
}

static void queue_message(struct lws* wsi, struct scs* s, const char* room, const char* msg) {
    if (s->out.data) { 
	    free(s->out.data);
	    s->out.data = NULL;
	    s->out.len = 0; 
    }
    s->out.data = build_line(room, msg, &s->out.len);
    if (s->out.data) lws_callback_on_writable(wsi);
}

static int callback_sc(struct lws* wsi, enum lws_callback_reasons reason, void* user, void* in, size_t len) {
    struct scs* s = (struct scs*)user;
    switch (reason) {
        case LWS_CALLBACK_CLIENT_ESTABLISHED:
            lwsl_user("[sc] connected; waiting for |challstr|\n");
            battle_state_init(&s->bs);
            break;
        case LWS_CALLBACK_CLIENT_RECEIVE: {
            const char* data = (const char*)in;
            lwsl_user("[recv]\n%.*s\n", (int)len, data);
            // track current room if present
            if (len > 2 && ((const char*)in)[0] == '>' ) {
                const char* nl = memchr(in, '\n', len);
                size_t roomlen = nl ? (size_t)(nl - (const char*)in) - 1 : 0;
                if (roomlen > 0 && roomlen < sizeof s->current_room) {
                    memcpy(s->current_room, (const char*)in + 1, roomlen);
                    s->current_room[roomlen] = '\0';
                }
            }
            if (!s->logged_in && memmem(data, len, "|challstr|", 10)) {
                if (!s->username[0]) {
                    srand((unsigned)time(NULL));
                    snprintf(s->username, sizeof s->username, "ArchGuest%04u", (unsigned)(rand() % 10000));
                }
                char cmd[128];
                snprintf(cmd, sizeof cmd, "/trn %s,0", s->username);
                queue_message(wsi, s, "", cmd);
                lwsl_user("[sc] sent guest /trn '%s'\n", s->username);
            }
            if (memmem(data, len, "|updateuser|", 11) && !s->logged_in) {
                s->logged_in = 1;
                lwsl_user("[sc] logged in as %s — joining lobby\n", s->username);
                queue_message(wsi, s, "", "/join lobby");
                queue_message(wsi, s, "lobby", "/cmd rooms");
            }
            // parse request JSON if present
            const char* req = memmem(data, len, "|request|", 9);
            if (req) {
                const char* json = req + 9;
                // skip leading '|' if present
                if (*json == '|') json++;
                battle_state_update_from_request(&s->bs, json);
                lwsl_user("[state] weather=%d terrain=%d\n", s->bs.weather, s->bs.terrain);
            }
            // split by lines and feed battle stream updates
            {
                const char* p = data;
                const char* end = data + len;
                while (p < end) {
                    const char* nl = memchr(p, '\n', (size_t)(end - p));
                    size_t linelen = nl ? (size_t)(nl - p) : (size_t)(end - p);
                    if (linelen > 0 && p[0] == '|') {
                        char tmp[256];
                        size_t cpy = linelen < sizeof tmp - 1 ? linelen : sizeof tmp - 1;
                        memcpy(tmp, p, cpy);
                        tmp[cpy] = '\0';
                        battle_state_update_from_line(&s->bs, tmp);
                    }
                    if (!nl) break;
                    p = nl + 1;
                }
            }
            break;
        }
        case LWS_CALLBACK_CLIENT_WRITEABLE: {
            if (s->out.data && s->out.len > 0) {
                unsigned char* buf = malloc(LWS_PRE + s->out.len);
                if (buf) {
                    memcpy(buf + LWS_PRE, s->out.data, s->out.len);
                    int n = lws_write(wsi, buf + LWS_PRE, s->out.len, LWS_WRITE_TEXT);
                    free(buf);
                    if (n < 0) lwsl_err("[sc] write failed\n");
                }
                free(s->out.data);
                s->out.data = NULL;
                s->out.len = 0;
            }
            break;
        }
        case LWS_CALLBACK_CLIENT_CLOSED:
        case LWS_CALLBACK_CLIENT_CONNECTION_ERROR:
            lwsl_user("[sc] connection closed or error\n");
            g_quit = 1;
            break;
        default:
            break;
    }
    return 0;
}

struct ShowdownClient* showdown_client_create(const char* host, int port, const char* path) {
    signal(SIGINT, sigint_handler);
    struct ShowdownClient* cli = calloc(1, sizeof *cli);
    if (!cli) return NULL;
    cli->protocols[0].name = "sc-proto";
    cli->protocols[0].callback = callback_sc;
    cli->protocols[0].per_session_data_size = sizeof(struct scs);
    cli->protocols[1].name = NULL;
    struct lws_context_creation_info info;
    memset(&info, 0, sizeof info);
    info.port = CONTEXT_PORT_NO_LISTEN;
    info.protocols = cli->protocols;
    info.options = LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT;
    lws_set_log_level(LLL_USER | LLL_ERR | LLL_WARN | LLL_NOTICE, NULL);
    cli->ctx = lws_create_context(&info);
    if (!cli->ctx) {
	    free(cli);
	    return NULL;
    }
    memset(&cli->cc, 0, sizeof cli->cc);
    cli->cc.context = cli->ctx;
    cli->cc.address = host;
    cli->cc.port = port;
    cli->cc.path = path;
    cli->cc.host = host;
    cli->cc.origin = "https://play.pokemonshowdown.com";
    cli->cc.protocol = cli->protocols[0].name;
    cli->cc.ssl_connection = LCCSCF_USE_SSL;
    if (!lws_client_connect_via_info(&cli->cc)) {
        lws_context_destroy(cli->ctx);
        free(cli);
        return NULL;
    }
    return cli;
}

int showdown_client_run(struct ShowdownClient* cli) {
    if (!cli || !cli->ctx) return 1;
    while (!g_quit) lws_service(cli->ctx, 50);
    return 0;
}

void showdown_client_destroy(struct ShowdownClient* cli) {
    if (!cli) return;
    if (cli->ctx) lws_context_destroy(cli->ctx);
    free(cli);
}

