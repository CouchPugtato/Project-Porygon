'use strict';

const {TeamValidator} = require('../external/pokemon-showdown/dist/sim/team-validator');

const FORMAT = 'gen9randomdoublesbattle', validators = new Map();

function levelValidator(level) {
    if (!Number.isInteger(level) || level < 1 || level > 100) throw new Error('invalid hypothetical battle level');
    if (!validators.has(level)) {
        const format = new TeamValidator(FORMAT).format;
        // acquisition legality at the native maximum before battle level scaling
        // clear the cached table so the added native adjustment rule takes effect
        validators.set(level, new TeamValidator({...format, name: `Hypothetical level ${level}`,
            id: `hypotheticallevel${level}`, team: undefined, ruleTable: null,
            ruleset: [...format.ruleset, `Adjust Level = ${level}`]}));
    }
    return validators.get(level);
}

function validateHypotheticalSet(set) {
    return levelValidator(set.level).validateSet(set, {});
}

function validateHypotheticalTeam(team) {
    // each member already has set legality checked at its own observed battle level
    // team clauses checked on detached acquisition level copies without changing battle stats
    return levelValidator(100).validateTeam(team.map(set => ({...structuredClone(set), level: 100})));
}

module.exports = {validateHypotheticalSet, validateHypotheticalTeam};
