#include "showdown_client.h"
#include "game_state.h"
#include <libwebsockets.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <signal.h>
#include <curl/curl.h>

#define MAX_LINE 4096

struct scs {
    int logged_in;
    int sent_trn;
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

static void send_global(struct lws* wsi, struct scs* s, const char* cmd) {
    queue_message(wsi, s, "", cmd);
}

static void join_room(struct lws* wsi, struct scs* s, const char* room) {
    char cmd[96];
    snprintf(cmd, sizeof cmd, "/join %s", room);
    send_global(wsi, s, cmd);
}

static void leave_room(struct lws* wsi, struct scs* s, const char* room) {
    char cmd[96];
    snprintf(cmd, sizeof cmd, "/leave %s", room);
    send_global(wsi, s, cmd);
}

static void choose_default_move(struct lws* wsi, struct scs* s, const char* room) {
    // Minimal policy: choose first move; refine later once parsing request fully
    (void)s;
    queue_message(wsi, s, room, "/choose move 1");
}

static void gen_id_name(const char* prefix, char* out, size_t outsz) {
    if (!prefix) prefix = "pory";
    size_t p = strlen(prefix);
    if (outsz <= p + 1) { if (out && outsz) out[0] = '\0'; return; }
    memcpy(out, prefix, p);
    size_t i = p;
    int max_total = 18; // Showdown requires <= 18 chars
    int max_digits = max_total - (int)p;
    if (max_digits < 4) max_digits = 4; // ensure some randomness
    if (max_digits > 12) max_digits = 12; // cap digits
    for (int j = 0; j < max_digits && i + 1 < outsz; j++) {
        int d = rand() % 10;
        out[i++] = (char)('0' + d);
    }
    out[i] = '\0';
}

struct bufaccum { char* data; size_t len; };
static size_t curl_write_cb(void* ptr, size_t size, size_t nmemb, void* userdata) {
    struct bufaccum* b = (struct bufaccum*)userdata;
    size_t total = size * nmemb;
    char* nd = (char*)realloc(b->data, b->len + total + 1);
    if (!nd) return 0;
    memcpy(nd + b->len, ptr, total);
    b->data = nd;
    b->len += total;
    b->data[b->len] = '\0';
    return total;
}

static int extract_assertion_from_body(const char* body, char* out, size_t outsz) {
    if (!body) return 0;
    // Server error responses begin with ';;' — treat as failure
    if (body[0] == ';' && body[1] == ';') return 0;
    const char* k = strstr(body, "assertion");
    if (k) {
        const char* q = strchr(k, '"');
        if (!q) return 0;
        q = strchr(q + 1, '"'); // after key
        if (!q) return 0;
        const char* vstart = strchr(q + 1, '"');
        if (!vstart) return 0;
        vstart++;
        size_t i = 0; while (vstart[i] && vstart[i] != '"' && i + 1 < outsz) { out[i] = vstart[i]; i++; }
        out[i] = '\0';
        return i > 0; // empty assertion is failure
    }
    // Otherwise assume body is the assertion string itself
    size_t n = strlen(body);
    if (n >= 2 && body[0] == ';' && body[1] == ';') return 0;
    if (n == 0) return 0;
    if (n >= outsz) n = outsz - 1;
    memcpy(out, body, n);
    out[n] = '\0';
    return 1;
}

static void sanitize_userid(const char* name, char* out, size_t outsz) {
    size_t i = 0;
    for (const char* p = name; *p && i + 1 < outsz; p++) {
        unsigned char c = (unsigned char)*p;
        if (c >= 'A' && c <= 'Z') c = (unsigned char)(c - 'A' + 'a');
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
            out[i++] = (char)c;
        }
    }
    out[i] = '\0';
}

static int fetch_assertion(const char* username, const char* challstr, char* out, size_t outsz) {
    if (!username || !challstr || !out || outsz < 8) return 0;
    const char* ps_user = getenv("PS_USER");
    const char* ps_pass = getenv("PS_PASSWORD");
    const char* ps_server = getenv("PS_SERVER");
    const char* server = (ps_server && *ps_server) ? ps_server : "sim3.psim.us";
    char serverid[32] = {0};
    {
        const char* env_sid = getenv("PS_SERVERID");
        if (env_sid && *env_sid) {
            size_t n = strlen(env_sid);
            if (n >= sizeof serverid) n = sizeof serverid - 1;
            memcpy(serverid, env_sid, n);
            serverid[n] = '\0';
        } else if (strstr(server, ".psim.us")) {
            snprintf(serverid, sizeof serverid, "%s", "showdown");
        } else {
            const char* dot = strchr(server, '.');
            size_t n = dot ? (size_t)(dot - server) : strlen(server);
            if (n >= sizeof serverid) n = sizeof serverid - 1;
            memcpy(serverid, server, n);
            serverid[n] = '\0';
        }
    }
    // Split challstr of the form "<keyid>|<challenge>" into components
    const char* bar = strchr(challstr, '|');
    char keyid[32] = {0};
    char challenge[512] = {0};
    if (bar) {
        size_t klen = (size_t)(bar - challstr);
        if (klen >= sizeof keyid) klen = sizeof keyid - 1;
        memcpy(keyid, challstr, klen);
        keyid[klen] = '\0';
        size_t clen = strlen(bar + 1);
        if (clen >= sizeof challenge) clen = sizeof challenge - 1;
        memcpy(challenge, bar + 1, clen);
        challenge[clen] = '\0';
    } else {
        // Fallback: treat whole string as challenge with default keyid "4"
        snprintf(keyid, sizeof keyid, "%s", "4");
        size_t clen = strlen(challstr);
        if (clen >= sizeof challenge) clen = sizeof challenge - 1;
        memcpy(challenge, challstr, clen);
        challenge[clen] = '\0';
    }
    const char* name = ps_user && *ps_user ? ps_user : username;
    CURL* curl = curl_easy_init();
    if (!curl) return 0;
    struct bufaccum resp = {0};
    char* form = NULL;
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "Project-Porygon/0.1");
    struct curl_slist* hdrs = NULL;
    hdrs = curl_slist_append(hdrs, "Origin: https://play.pokemonshowdown.com");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);
    char* enc_keyid = curl_easy_escape(curl, keyid, 0);
    char* enc_chal = curl_easy_escape(curl, challenge, 0);
    char* enc_challstr = curl_easy_escape(curl, challstr, 0);
    char url[768];
    if (ps_user && *ps_user && ps_pass && *ps_pass) {
        // Registered login via API with challenge params
        char* enc_name = curl_easy_escape(curl, name, 0);
        char* enc_pass = curl_easy_escape(curl, ps_pass, 0);
        char* enc_server = curl_easy_escape(curl, serverid, 0);
        // POST form: name, pass, challengekeyid, challenge
        snprintf(url, sizeof url, "https://play.pokemonshowdown.com/api/login");
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        size_t flen = strlen("name=") + strlen(enc_name) + strlen("&pass=") + strlen(enc_pass) + strlen("&challstr=") + strlen(enc_challstr) + strlen("&serverid=") + strlen(enc_server) + 1;
        form = (char*)malloc(flen);
        if (!form) { curl_free(enc_name); curl_free(enc_pass); curl_free(enc_server); curl_free(enc_keyid); curl_free(enc_chal); curl_free(enc_challstr); curl_easy_cleanup(curl); curl_slist_free_all(hdrs); return 0; }
        snprintf(form, flen, "name=%s&pass=%s&challstr=%s&serverid=%s", enc_name, enc_pass, enc_challstr, enc_server);
        hdrs = curl_slist_append(hdrs, "Content-Type: application/x-www-form-urlencoded");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, form);
        curl_free(enc_name);
        curl_free(enc_pass);
        curl_free(enc_server);
        // form freed after perform
    } else {
        // guest getassertion via API with challenge params
        char userid[64];
        sanitize_userid(name, userid, sizeof userid);
        char* enc_userid = curl_easy_escape(curl, userid, 0);
        snprintf(url, sizeof url, "https://play.pokemonshowdown.com/api/getassertion?userid=%s&challstr=%s&serverid=%s", enc_userid, enc_challstr, serverid);
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
        curl_free(enc_userid);
    }
    hdrs = curl_slist_append(hdrs, "Origin: https://play.pokemonshowdown.com");
    long code = 0;
    CURLcode cc = curl_easy_perform(curl);
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
    curl_easy_cleanup(curl);
    curl_slist_free_all(hdrs);
    if (enc_keyid) curl_free(enc_keyid);
    if (enc_chal) curl_free(enc_chal);
    if (enc_challstr) curl_free(enc_challstr);
    if (form) free(form);
    int ok = 0;
    if (resp.data) {
        size_t preview = resp.len > 200 ? 200 : resp.len;
        lwsl_user("[auth] HTTP %ld, body: %.*s\n", code, (int)preview, resp.data);
    }
    if (cc == CURLE_OK && code == 200 && resp.data) {
        ok = extract_assertion_from_body(resp.data, out, outsz);
    }
    free(resp.data);
    return ok;
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
            if (!s->sent_trn && memmem(data, len, "|challstr|", 10)) {
                // Prefer explicit username via env; else generate guestXXXXXXXXXXXX
                const char* env_user = getenv("PS_USER");
                if (env_user && *env_user) {
                    snprintf(s->username, sizeof s->username, "%s", env_user);
                } else if (!s->username[0]) {
                    srand((unsigned)time(NULL));
                    gen_id_name("porygon", s->username, sizeof s->username);
                }
                // Extract full challstr payload from the line
                const char* cpos = memmem(data, len, "|challstr|", 10);
                char* cbuf = NULL;
                if (cpos) {
                    const char* p = cpos + 10;
                    size_t remain = (size_t)(data + len - p);
                    const char* nl = memchr(p, '\n', remain);
                    size_t clen = nl ? (size_t)(nl - p) : remain;
                    cbuf = (char*)malloc(clen + 1);
                    if (cbuf) { memcpy(cbuf, p, clen); cbuf[clen] = '\0'; }
                }
                char assertion[2048] = {0};
                int fa = 0;
                if (cbuf && cbuf[0]) fa = fetch_assertion(s->username, cbuf, assertion, sizeof assertion);
                if (fa == 0) {
                    // fallback guest to non-guest prefix if server disallows
                    gen_id_name("porygon", s->username, sizeof s->username);
                    if (cbuf && cbuf[0]) fa = fetch_assertion(s->username, cbuf, assertion, sizeof assertion);
                }
                if (fa > 0) {
                    char cmd[2048];
                    snprintf(cmd, sizeof cmd, "/trn %s,0,%s", s->username, assertion);
                    send_global(wsi, s, cmd);
                    lwsl_user("[sc] sent /trn with assertion for '%s'\n", s->username);
                } else {
                    char cmd[128];
                    snprintf(cmd, sizeof cmd, "/trn %s,0", s->username);
                    send_global(wsi, s, cmd);
                    lwsl_user("[sc] fallback /trn '%s'\n", s->username);
                }
                s->sent_trn = 1;
                if (cbuf) { free(cbuf); cbuf = NULL; }
            }
            if (memmem(data, len, "|updateuser|", 11) && !s->logged_in) {
                s->logged_in = 1;
                lwsl_user("[sc] logged in as %s — joining lobby\n", s->username);
                join_room(wsi, s, "lobby");
                send_global(wsi, s, "/cmd rooms");
            }
            // parse request JSON if present
            const char* req = memmem(data, len, "|request|", 9);
            if (req) {
                const char* json = req + 9;
                // skip leading '|' if present
                if (*json == '|') json++;
                battle_state_update_from_request(&s->bs, json);
                lwsl_user("[state] weather=%d terrain=%d\n", s->bs.weather, s->bs.terrain);
                // If this is a battle room, make a default choice
                if (s->current_room[0] && !strncmp(s->current_room, "battle-", 7)) {
                    choose_default_move(wsi, s, s->current_room);
                }
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

