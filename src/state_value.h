#ifndef STATE_VALUE_H
#define STATE_VALUE_H

typedef enum {
    KNOW_UNKNOWN = 0,
    KNOW_INFERRED = 1,
    KNOW_CONFIRMED = 2
} KnowledgeLevel;

typedef struct {
    int value;
    KnowledgeLevel knowledge;
} TrackedInt;

void tracked_int_reset(TrackedInt* tracked);
void tracked_int_set_unknown(TrackedInt* tracked);
void tracked_int_set_inferred(TrackedInt* tracked, int value);
void tracked_int_set_confirmed(TrackedInt* tracked, int value);
void tracked_int_promote_confirmed(TrackedInt* tracked, int value);

#endif
