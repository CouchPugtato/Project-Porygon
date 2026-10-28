'use strict';

// shared mechanics boundary for reconstruction and later turn proposals
const MOVES = new Set(['tackle', 'quickattack', 'watergun', 'thunderbolt', 'flamethrower',
    'protect', 'thunderwave', 'willowisp', 'poisongas']);

function isOrdinaryMove(id) { return MOVES.has(id); }

module.exports = {isOrdinaryMove};
