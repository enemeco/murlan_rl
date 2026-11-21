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
  const exchange={...state,phase:'swap',legal_actions:[111],actions:[{id:111,kind:'swap_give',ranks:['4'],required_cards:['4C']}]};
  assert.equal(matchAction(exchange,['4C']).id,111);
  assert.equal(matchAction(exchange,['4H']),null);
});
test('A-low Kolors extend through 14 cards and need two different aces',()=>{
  const allRanks=['A','2','3','4','5','6','7','8','9','10','J','Q','K','A'];
  for(let length=6;length<=14;length++){
    const ranks=allRanks.slice(0,length);
    const cards=ranks.map((r,i)=>r==='3'?'3S':r+(i===13?'D':'C'));
    const id=101+length-6;
    const long={...state,your_hand:cards,require_three_spades:true,
      legal_actions:[id],actions:[{id,kind:'straight',ranks}]};
    assert.equal(matchAction(long,[...cards].reverse()).id,id);
    assert.equal(matchAction(long,cards.slice(1)),null);
    assert.equal(matchAction({...long,legal_actions:[0]},cards),null);
    if(length===14)assert.equal(matchAction(long,[...cards.slice(0,13),'AC']),null);
  }
});
test('ambiguous 13-card selection uses the server action order and legal mask',()=>{
  const high=['2','3','4','5','6','7','8','9','10','J','Q','K','A'];
  const low=['A','2','3','4','5','6','7','8','9','10','J','Q','K'];
  const cards=high.map(r=>r+'C');
  const both={...state,your_hand:cards,legal_actions:[100,108],actions:[
    {id:100,kind:'straight',ranks:high},{id:108,kind:'straight',ranks:low}]};
  assert.equal(matchAction(both,[...cards].reverse()).id,100);
  assert.equal(matchAction({...both,legal_actions:[108]},cards).id,108);
});
test('special low straight works in any selection order',()=>{
  const low={...state,legal_actions:[91],actions:[{id:91,kind:'straight',ranks:['A','2','3','4','5']}]};
  assert.equal(matchAction(low,['3S','AC','4H','2C','5C']).id,91);
});
test('2-3-4-5-6 is selectable only when the server permits that Kolor',()=>{
  const low={...state,legal_actions:[92],actions:[{id:92,kind:'straight',ranks:['2','3','4','5','6']}]};
  const cards=['6C','3S','2C','4H','5C'];
  assert.equal(matchAction(low,cards).id,92);
  assert.equal(matchAction({...low,legal_actions:[0]},cards),null);
  assert.equal(matchAction(low,['3S','4H','5C','6C','7C']),null);
});
test('longer 2-low Kolors through the full 13-card run accept manual selection',()=>{
  const allRanks=['2','3','4','5','6','7','8','9','10','J','Q','K','A'];
  for(let length=6;length<=13;length++){
    const ranks=allRanks.slice(0,length);
    const cards=ranks.map(r=>r==='3'?'3S':r+'C');
    const id=92+length-5;
    const long={...state,your_hand:[...cards,'RJ'],require_three_spades:true,
      legal_actions:[id],actions:[{id,kind:'straight',ranks}]};
    assert.equal(matchAction(long,[...cards].reverse()).id,id);
    assert.equal(matchAction(long,cards.slice(1)),null);
    assert.equal(matchAction({...long,legal_actions:[0]},cards),null);
  }
});
