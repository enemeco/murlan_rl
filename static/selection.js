'use strict';
// Match exact card selections to the server's legal rank actions.
// The server remains authoritative and validates suits, ownership and opening rules.
const MurlanSelection = (() => {
  const rankOf = card => card === 'BJ' || card === 'RJ' ? card : card.slice(0, -1);
  const counts = {single: 1, pair: 2, triple: 3, quad: 4, swap_give: 1};
  function expectedRanks(action) {
    return action.kind === 'straight' ? [...action.ranks] : Array(counts[action.kind] || 0).fill(action.ranks[0]);
  }
  function matchesRanks(action, cards) {
    return JSON.stringify(expectedRanks(action).sort()) === JSON.stringify(cards.map(rankOf).sort());
  }
  function matchAction(state, cards) {
    if (!state || state.done || !state.your_turn || !cards.length || new Set(cards).size !== cards.length) return null;
    if (cards.some(c => !state.your_hand.includes(c))) return null;
    if (state.phase === 'play' && state.require_three_spades && !cards.includes('3S')) return null;
    return state.actions.find(a => a.kind !== 'pass' && state.legal_actions.includes(a.id)
      && matchesRanks(a, cards) && (a.required_cards || []).every(c => cards.includes(c))) || null;
  }
  return {rankOf, expectedRanks, matchesRanks, matchAction};
})();
if (typeof module !== 'undefined' && module.exports) module.exports = MurlanSelection;
