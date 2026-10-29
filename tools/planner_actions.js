'use strict';

const {expandedMoveActions} = require('./opening_actions');
const {replacementActions} = require('./replacement_actions');
const {selectProposals} = require('./opening_proposals');

class UnsupportedPlannerRequest extends Error {}

function decisionActions(view) {
    if (view.ended || view.request?.wait) return {turns: [], training_eligible: false};
    try {
        if (view.phase === 'switch') return replacementActions(view, true);
        return expandedMoveActions(view);
    } catch (error) {
        if (/ordinary action generation requires|move action space outside|replacement actions require|only fainted slot|replacement request has/.test(error.message))
            throw new UnsupportedPlannerRequest(error.message);
        throw error;
    }
}

function selectCommands(view, count, rng) {
    const pool = decisionActions(view).turns;
    if (view.phase === 'switch') return pool.slice(0, count).map(turn => turn.command);
    return selectProposals(view, pool, count, rng, 'coverage').commands;
}

module.exports = {decisionActions, selectCommands, UnsupportedPlannerRequest};
