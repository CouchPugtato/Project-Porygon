'use strict';

function solveBeliefGame(worlds, iterations = 512) {
    const rows = worlds?.[0]?.length;
    if (!Array.isArray(worlds) || !worlds.length || !rows || worlds.some(payoffs =>
            !Array.isArray(payoffs) || payoffs.length !== rows || !payoffs[0]?.length ||
            payoffs.some(row => row.length !== payoffs[0].length || row.some(value => !Number.isFinite(value)))) ||
            !Number.isInteger(iterations) || iterations < 1 || iterations > 8192)
        throw new Error('finite rectangular payoff matrix and bounded iterations required');
    const rowRegret = Array(rows).fill(0), colRegret = worlds.map(matrix => Array(matrix[0].length).fill(0));
    const rowTotal = Array(rows).fill(0), colTotal = colRegret.map(row => row.map(() => 0));
    const policy = regrets => {
        const positive = regrets.map(value => Math.max(0, value)), sum = positive.reduce((a, b) => a + b, 0);
        return positive.map(value => sum ? value / sum : 1 / positive.length);
    };
    for (let iteration = 0; iteration < iterations; ++iteration) {
        const p = policy(rowRegret), rowValues = Array(rows).fill(0);
        worlds.forEach((matrix, world) => {
            const q = policy(colRegret[world]);
            const values = matrix.map(row => row.reduce((sum, value, j) => sum + value * q[j], 0));
            values.forEach((value, i) => { rowValues[i] += value / worlds.length; });
            const value = values.reduce((sum, score, i) => sum + score * p[i], 0);
            for (let j = 0; j < q.length; ++j) {
                const responseValue = matrix.reduce((sum, row, i) => sum + row[j] * p[i], 0);
                colRegret[world][j] += value - responseValue; colTotal[world][j] += q[j];
            }
        });
        const value = rowValues.reduce((sum, score, i) => sum + score * p[i], 0);
        for (let i = 0; i < rows; ++i) { rowRegret[i] += rowValues[i] - value; rowTotal[i] += p[i]; }
    }
    const policyP = rowTotal.map(n => n / iterations), policyQ = colTotal.map(row => row.map(n => n / iterations));
    const lower = worlds.reduce((sum, matrix) => sum + Math.min(...matrix[0].map((_, j) =>
        matrix.reduce((value, row, i) => value + row[j] * policyP[i], 0))) / worlds.length, 0);
    const upper = Math.max(...policyP.map((_, i) => worlds.reduce((sum, matrix, world) =>
        sum + matrix[i].reduce((value, cell, j) => value + cell * policyQ[world][j], 0) / worlds.length, 0)));
    return {policy: policyP, opponent_policy: policyQ, lower, upper, gap: Math.max(0, upper - lower), iterations};
}

function solveMatrix(matrix, iterations) { return solveBeliefGame([matrix], iterations); }

module.exports = {solveMatrix, solveBeliefGame};
