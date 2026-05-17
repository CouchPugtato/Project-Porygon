#include "episode.h"
#include "gru_model.h"
#include "observation.h"
#include "showdown_client.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int main(void) {
    if (getenv("PORYGON_DEMO_GRU")) {
        Observation obs;
        size_t obs_dim = observation_flat_size();
        float* flat = (float*)malloc(obs_dim * sizeof(float));
        float* hidden = NULL;
        float* policy = NULL;
        float value = 0.0f;
        GruModel* model = NULL;
        Episode episode;
        int action = -1;

        if (!flat) {
            fprintf(stderr, "Failed to allocate observation buffer\n");
            return 1;
        }

        model = gru_model_create(obs_dim, 128, OBS_NUM_ACTIONS);
        hidden = (float*)malloc(gru_model_hidden_dim(model) * sizeof(float));
        policy = (float*)malloc(OBS_NUM_ACTIONS * sizeof(float));
        if (!model || !hidden || !policy || !episode_init(&episode, 4, obs_dim)) {
            fprintf(stderr, "Failed to initialize GRU demo\n");
            gru_model_destroy(model);
            free(hidden);
            free(policy);
            free(flat);
            return 1;
        }

        gru_model_zero_state(model, hidden);
        observation_fill_demo(&obs);
        for (int t = 0; t < 3; ++t) {
            obs.turn_norm = 0.1f + (0.1f * (float)t);
            obs.self_team[0].hp_frac = 0.82f - (0.08f * (float)t);
            obs.opp_team[0].hp_frac = 0.66f - (0.05f * (float)t);
            if (observation_flatten(flat, obs_dim, &obs) != obs_dim) {
                fprintf(stderr, "Failed to flatten observation\n");
                episode_free(&episode);
                gru_model_destroy(model);
                free(hidden);
                free(policy);
                free(flat);
                return 1;
            }
            episode_append(&episode, flat, -1, 0.0f, (uint8_t)(t == 2));
        }

        gru_model_forward_sequence(model, episode.observations, episode.count, hidden, policy, &value);
        action = gru_model_select_action(policy, obs.legal_mask, OBS_NUM_ACTIONS);
        episode.actions[episode.count - 1] = action;

        printf("obs_dim=%zu hidden_dim=%zu episode_steps=%zu action=%d value=%.4f\n",
            obs_dim,
            gru_model_hidden_dim(model),
            episode.count,
            action,
            value);

        episode_free(&episode);
        gru_model_destroy(model);
        free(hidden);
        free(policy);
        free(flat);
        return 0;
    }

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

