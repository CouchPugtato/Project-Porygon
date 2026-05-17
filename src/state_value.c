#include "state_value.h"

void tracked_int_reset(TrackedInt* tracked) {
    if (!tracked) {
        return;
    }
    tracked->value = 0;
    tracked->knowledge = KNOW_UNKNOWN;
}

void tracked_int_set_unknown(TrackedInt* tracked) {
    tracked_int_reset(tracked);
}

void tracked_int_set_inferred(TrackedInt* tracked, int value) {
    if (!tracked) {
        return;
    }
    tracked->value = value;
    tracked->knowledge = KNOW_INFERRED;
}

void tracked_int_set_confirmed(TrackedInt* tracked, int value) {
    if (!tracked) {
        return;
    }
    tracked->value = value;
    tracked->knowledge = KNOW_CONFIRMED;
}

void tracked_int_promote_confirmed(TrackedInt* tracked, int value) {
    if (!tracked) {
        return;
    }
    if (tracked->knowledge != KNOW_CONFIRMED || tracked->value == 0) {
        tracked_int_set_confirmed(tracked, value);
    }
}
