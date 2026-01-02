#include "showdown_client.h"
#include "game_state.h"
#include <libwebsockets.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <signal.h>
#include <curl/curl.h>
#include <stdio.h>
#include <sys/stat.h>

#define MAX_LINE 4096

struct scs {
    int logged_in;
    int sent_trn;
    char username[32];
    struct { size_t len; char* data; } outq[16];
    int out_head;
    int out_tail;
    int in_battle;
    time_t last_search_retry;
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

static int outq_is_empty(struct scs* s) { return s->out_head == s->out_tail; }
static int outq_is_full(struct scs* s) { return ((s->out_tail + 1) & 15) == s->out_head; }
static void outq_push(struct scs* s, char* data, size_t len) {
    if (!data) return;
    int next = (s->out_tail + 1) & 15;
    if (next == s->out_head) { free(data); return; }
    s->outq[s->out_tail].data = data;
    s->outq[s->out_tail].len = len;
    s->out_tail = next;
}
static int outq_pop(struct scs* s, char** data, size_t* len) {
    if (s->out_head == s->out_tail) return 0;
    *data = s->outq[s->out_head].data;
    *len = s->outq[s->out_head].len;
    s->outq[s->out_head].data = NULL;
    s->outq[s->out_head].len = 0;
    s->out_head = (s->out_head + 1) & 15;
    return 1;
}

static void queue_message(struct lws* wsi, struct scs* s, const char* room, const char* msg) {
    size_t len = 0;
    char* data = build_line(room, msg, &len);
    if (data && len > 0) lwsl_user("[sc] queued: %.*s\n", (int)len, data);
    outq_push(s, data, len);
    lws_callback_on_writable(wsi);
}

static void send_global(struct lws* wsi, struct scs* s, const char* cmd) {
    queue_message(wsi, s, "", cmd);
}

static void join_room(struct lws* wsi, struct scs* s, const char* room) {
    char cmd[96];
    snprintf(cmd, sizeof cmd, "/join %s", room);
    lwsl_user("[sc] joining room: %s\n", room);
    send_global(wsi, s, cmd);
}

static void leave_room(struct lws* wsi, struct scs* s, const char* room) {
    char cmd[96];
    snprintf(cmd, sizeof cmd, "/leave %s", room);
    lwsl_user("[sc] leaving room: %s\n", room);
    send_global(wsi, s, cmd);
}

static void choose_default_move(struct lws* wsi, struct scs* s, const char* room) {
    (void)s;
    lwsl_user("[battle] choosing default move 1 in %s\n", room);
    queue_message(wsi, s, room, "/choose move 1");
}

static void choose_from_request(struct lws* wsi, struct scs* s, const char* room, const char* request_json) {
    const char* tp = strstr(request_json, "\"teamPreview\":true");
    if (tp) {
        const char* ms = strstr(request_json, "\"maxChosenTeamSize\":");
        int msz = 0;
        if (ms) msz = atoi(ms + 22);
        if (msz <= 0) msz = 2;
        char cmd[64];
        if (msz >= 2) snprintf(cmd, sizeof cmd, "/choose team 1, 2"); else snprintf(cmd, sizeof cmd, "/choose team 1");
        lwsl_user("[battle] choosing team: %s in %s\n", cmd, room);
        queue_message(wsi, s, room, cmd);
        return;
    }
    if (strstr(request_json, "\"forceSwitch\":true")) {
        int switches = 0;
        const char* p = request_json;
        while ((p = strstr(p, "\"forceSwitch\":true"))) { switches++; p += 18; }
        char cmd[64];
        if (switches >= 2) snprintf(cmd, sizeof cmd, "/choose switch 1, switch 2");
        else snprintf(cmd, sizeof cmd, "/choose switch 1");
        lwsl_user("[battle] choosing switches: %s in %s\n", cmd, room);
        queue_message(wsi, s, room, cmd);
        return;
    }
    int idxs[2] = {1, 1};
    int found = 0;
    const char* p = request_json;
    for (int a = 0; a < 2; ++a) {
        const char* m = strstr(p, "\"moves\"");
        if (!m) break;
        const char* arr = strchr(m, '[');
        if (!arr) break;
        const char* end = strchr(arr, ']');
        if (!end) break;
        int mi = 1;
        const char* q = arr;
        while (q && q < end) {
            const char* dis = strstr(q, "\"disabled\":");
            if (!dis || dis >= end) break;
            const char* val = dis + 12;
            if (!strncmp(val, "false", 5)) { idxs[a] = mi; found++; break; }
            mi++;
            q = strstr(dis + 12, "\"disabled\":");
        }
        p = end + 1;
    }
    char cmd[64];
    if (found >= 2) {
        snprintf(cmd, sizeof cmd, "/choose move %d, move %d", idxs[0], idxs[1]);
    } else {
        snprintf(cmd, sizeof cmd, "/choose move %d", idxs[0]);
    }
    lwsl_user("[battle] choosing moves: %s in %s\n", cmd, room);
    queue_message(wsi, s, room, cmd);
}

static void ensure_dir(const char* path) {
    struct stat st;
    if (stat(path, &st) != 0) {
        mkdir(path, 0777);
    }
}

static void write_state_json(const char* room, const struct BattleState* bs) {
    ensure_dir("matches");
    char fname[256];
    time_t t = time(NULL);
    snprintf(fname, sizeof fname, "matches/%s_%ld.json", room && *room ? room : "global", (long)t);
    FILE* f = fopen(fname, "w");
    if (!f) return;
    fprintf(f, "{\n");
    fprintf(f, "\"room\":\"%s\",\n", room && *room ? room : "");
    fprintf(f, "\"weather\":%d,\n", bs->weather);
    fprintf(f, "\"terrain\":%d,\n", bs->terrain);
    fprintf(f, "\"friendly\":[");
    for (int i = 0; i < 6; i++) {
        const struct Pokemon* p = &bs->friendly_pokemon[i];
        fprintf(f, "{\"hp\":%d,\"max\":%d,\"status\":%d}", p->current_hp, p->max_hp, p->status_condition);
        if (i != 5) fprintf(f, ",");
    }
    fprintf(f, "],\n\"opponent\":[");
    for (int i = 0; i < 6; i++) {
        const struct Pokemon* p = &bs->opponent_pokemon[i];
        fprintf(f, "{\"hp\":%d,\"max\":%d,\"status\":%d}", p->current_hp, p->max_hp, p->status_condition);
        if (i != 5) fprintf(f, ",");
    }
    fprintf(f, "]\n}\n");
    fclose(f);
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
            s->out_head = 0; s->out_tail = 0;
            s->in_battle = 0;
            s->last_search_retry = 0;
            break;
        case LWS_CALLBACK_CLIENT_RECEIVE: {
            const char* data = (const char*)in;
            lwsl_user("SERVER - %.*s\n", (int)len, data);
            // track current room if present
            if (len > 2 && ((const char*)in)[0] == '>' ) {
                const char* nl = memchr(in, '\n', len);
                size_t roomlen = nl ? (size_t)(nl - (const char*)in) - 1 : 0;
                if (roomlen > 0 && roomlen < sizeof s->current_room) {
                    memcpy(s->current_room, (const char*)in + 1, roomlen);
                    s->current_room[roomlen] = '\0';
                    if (!strncmp(s->current_room, "battle-", 7) && !s->in_battle) {
                        s->in_battle = 1;
                        lwsl_user("[battle] entered match: %s\n", s->current_room);
                    }
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
            {
                const char* up = memmem(data, len, "|updateuser|", 11);
                if (up && !s->logged_in) {
                    const char* p = up + 11;
                    const char* bar1 = memchr(p, '|', (size_t)(data + len - p));
                    if (bar1) {
                        const char* bar2 = memchr(bar1 + 1, '|', (size_t)(data + len - (bar1 + 1)));
                        if (bar2 && bar2 > bar1 + 1 && *(bar1 + 1) == '1') {
                            s->logged_in = 1;
                            lwsl_user("[sc] logged in as %s — joining lobby\n", s->username);
                            join_room(wsi, s, "lobby");
                            send_global(wsi, s, "/cmd rooms");
                            const char* rs = getenv("PS_ROOMS");
                            if (rs && *rs) {
                                const char* rp = rs;
                                while (*rp) {
                                    while (*rp == ' ' || *rp == ',') rp++;
                                    if (!*rp) break;
                                    char rbuf[64];
                                    size_t i = 0;
                                    while (rp[i] && rp[i] != ',' && i + 1 < sizeof rbuf) { rbuf[i] = rp[i]; i++; }
                                    rbuf[i] = '\0';
                                    join_room(wsi, s, rbuf);
                                    rp += i;
                                    while (*rp && *rp != ',') rp++;
                                }
                            } else {
                                const char* r1 = getenv("PS_ROOM");
                                if (r1 && *r1) join_room(wsi, s, r1);
                            }
                            const char* ic = getenv("PS_INIT_CMD");
                            if (ic && *ic) send_global(wsi, s, ic);
                            const char* fmt = getenv("PS_SEARCH");
                            char scmd[96];
                            if (fmt && *fmt) snprintf(scmd, sizeof scmd, "/search %s", fmt);
                            else snprintf(scmd, sizeof scmd, "/search %s", "gen9randomdoublesbattle");
                            lwsl_user("[sc] sending search: %s\n", scmd);
                            send_global(wsi, s, scmd);
                        }
                    }
                }
            }
            // parse request JSON if present
            const char* req = memmem(data, len, "|request|", 9);
            if (req) {
                const char* json = req + 9;
                // skip leading '|' if present
                if (*json == '|') json++;
                lwsl_user("SERVER REQUEST - %.*s\n", (int)(data + len - json), json);
                battle_state_update_from_request(&s->bs, json);
                lwsl_user("[state] weather=%d terrain=%d\n", s->bs.weather, s->bs.terrain);
                if (s->current_room[0]) write_state_json(s->current_room, &s->bs);
                if (s->current_room[0] && !strncmp(s->current_room, "battle-", 7)) {
                    choose_from_request(wsi, s, s->current_room, json);
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
                        lwsl_user("SERVER - %s\n", tmp);
                        if (!strncmp(tmp, "|move|", 6)) {
                            lwsl_user("[battle] move: %s\n", tmp);
                        }
                        if (!strncmp(tmp, "|popup|", 7) && strstr(tmp, "not ladderable")) {
                            const char* fb = getenv("PS_SEARCH_FALLBACK");
                            char scmd[96];
                            snprintf(scmd, sizeof scmd, "/search %s", (fb && *fb) ? fb : "gen8randomdoubles");
                            lwsl_user("[sc] fallback search due to popup: %s\n", scmd);
                            send_global(wsi, s, scmd);
                        }
                        if (!strncmp(tmp, "|updatesearch|", 13) && s->logged_in && strstr(tmp, "\"searching\":[]")) {
                            const char* fb = getenv("PS_SEARCH_FALLBACK");
                            char scmd[96];
                            time_t now = time(NULL);
                            if (s->last_search_retry == 0 || (now - s->last_search_retry) >= 3) {
                                snprintf(scmd, sizeof scmd, "/search %s", (fb && *fb) ? fb : "gen8randomdoubles");
                                lwsl_user("[sc] updatesearch empty; retry with: %s\n", scmd);
                                send_global(wsi, s, scmd);
                                s->last_search_retry = now;
                            }
                        }
                        battle_state_update_from_line(&s->bs, tmp);
                    }
                    if (!nl) break;
                    p = nl + 1;
                }
            }
            break;
        }
        case LWS_CALLBACK_CLIENT_WRITEABLE: {
            char* data = NULL; size_t dlen = 0;
            if (outq_pop(s, &data, &dlen) && data && dlen > 0) {
                unsigned char* buf = malloc(LWS_PRE + dlen);
                if (buf) {
                    memcpy(buf + LWS_PRE, data, dlen);
                    lwsl_user("CLIENT - %.*s\n", (int)dlen, data);
                    if (memmem(data, dlen, "/trn ", 5)) {
                        lwsl_user("[sc] sending challenge: %.*s\n", (int)dlen, data);
                    }
                    int n = lws_write(wsi, buf + LWS_PRE, dlen, LWS_WRITE_TEXT);
                    free(buf);
                    if (n < 0) lwsl_err("[sc] write failed\n");
                }
                free(data);
                if (!outq_is_empty(s)) lws_callback_on_writable(wsi);
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
