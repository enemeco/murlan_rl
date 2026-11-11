const test = require('node:test');
const assert = require('node:assert/strict');
const {matchAction} = require('../static/selection.js');
const state = {
  phase:'play', your_turn:true, done:false, require_three_spades:false,
  your_hand:['3C','3S','4C','4H','4S','5C','6C','7C','AC','2C','BJ','RJ'],
  legal_actions:[0,2,17,55],
  actions:[{id:0,kind:'pass',ranks:[]},{id:2,kind:'single',ranks:['4']},
    {id:17,kind:'pair',ranks:['4']},{id:55,kind:'straight',ranks:['3','4','5','6','7']}]
};
test('exact suits may vary but the rank action must be legal',()=>{
  assert.equal(matchAction(state,['4H','4S']).id,17);
  for(const cards of [[],['4H','5C'],['4H','4H'],['4H','4D'],['RJ']])assert.equal(matchAction(state,cards),null);
});
test('opening must include the actual three of spades',()=>{
  const opening={...state,require_three_spades:true};
  assert.equal(matchAction(opening,['3C','4C','5C','6C','7C']),null);
  assert.equal(matchAction(opening,['7C','6C','5C','4S','3S']).id,55);
});
test('no move can be selected out of turn or after the match',()=>{
  assert.equal(matchAction({...state,your_turn:false},['4C']),null);
  assert.equal(matchAction({...state,done:true},['4C']),null);
});
test('exchange fallback requires the exact lowest card',()=>{
  const exchange={...state,phase:'swap',legal_actions:[92],actions:[{id:92,kind:'swap_give',ranks:['4'],required_cards:['4C']}]};
  assert.equal(matchAction(exchange,['4C']).id,92);
  assert.equal(matchAction(exchange,['4H']),null);
});
test('special low straight works in any selection order',()=>{
  const low={...state,legal_actions:[91],actions:[{id:91,kind:'straight',ranks:['A','2','3','4','5']}]};
  assert.equal(matchAction(low,['3S','AC','4H','2C','5C']).id,91);
});
