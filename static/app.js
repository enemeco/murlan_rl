'use strict';
const $ = id => document.getElementById(id);
const KINDS = {single:'Letër teke',pair:'Dyshe',triple:'Treshe',quad:'Katërshe',straight:'Kolor',swap_give:'Shkëmbim',pass:'Pas'};
let game = null, selected = new Set(), busy = false, pending = null;
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const storage = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key,value) { try { localStorage.setItem(key,value); } catch {} },
  remove(key) { try { localStorage.removeItem(key); } catch {} }
};
const name = seat => seat === game?.human_seat ? 'Ti' : game?.player_names?.[seat] || 'Roboti';
const cardCount = n => n === 1 ? '1 letër' : `${n} letra`;
const chosenCards = () => game ? game.your_hand.filter(c => selected.has(c)) : [];
const chosenAction = () => MurlanSelection.matchAction(game, chosenCards());
function pretty(card) {
  if (card === 'BJ' || card === 'RJ') return {rank:'Xh',suit:'✦',red:card==='RJ',joker:true,label:card==='RJ'?'Xholi i kuq':'Xholi i zi'};
  const suit = card.slice(-1), symbol = {S:'♠',H:'♥',D:'♦',C:'♣'}[suit];
  return {rank:symbol?card.slice(0,-1):card,suit:symbol||'♠',red:suit==='H'||suit==='D',joker:false,label:card.slice(0,-1)+' '+(symbol||suit)};
}
function cardHTML(card, clickable = false) {
  const c = pretty(card), tag = clickable ? 'button' : 'div';
  const attributes = clickable ? `type="button" data-card="${esc(card)}" aria-pressed="false"` : '';
  return `<${tag} class="playing-card ${c.red?'red':''} ${c.joker?'joker':''}" ${attributes} aria-label="${esc(c.label)}"><span class="corner">${esc(c.rank)}</span><span class="suit">${c.suit}</span><span class="corner lower">${esc(c.rank)}</span></${tag}>`;
}
function renderHand() {
  $('hand').innerHTML = game.your_hand.length ? game.your_hand.map(c => cardHTML(c,true)).join('') : '<p class="muted">I mbarove letrat!</p>';
  $('hand').querySelectorAll('button[data-card]').forEach(button => {
    button.onclick = () => {
      if (busy || !game.your_turn || game.done) return;
      const card = button.dataset.card;
      selected.has(card) ? selected.delete(card) : selected.add(card);
      $('error').hidden = true;
      updateButtons();
    };
  });
}
function selectionMessage(action) {
  const cards = chosenCards();
  if (!game) return 'Loja po nis…';
  if (game.done) return 'Nis një lojë të re për të luajtur përsëri.';
  if (!game.your_turn) return 'Tani luajnë të tjerët.';
  if (!cards.length) return game.phase === 'swap' ? 'Zgjidh letrën që do të japësh.' : 'Zgjidh letrat që do të hedhësh në tavolinë.';
  if (action) return action.kind === 'swap_give' ? 'Mund ta japësh këtë letër.' : `${KINDS[action.kind]} · ${cardCount(cards.length)}. Mund ${cards.length===1?'ta luash':'t’i luash'}.`;
  if (game.phase === 'swap') return 'Zgjidh vetëm një nga letrat që lejohet të japësh.';
  if (game.require_three_spades && !cards.includes('3S')) return 'Letrat që luan në fillim duhet të përfshijnë 3♠.';
  if (game.trick && cards.length !== game.trick.size && cards.length !== 4) return `Në tavolinë ${game.trick.size===1?'është 1 letër':'janë '+cardCount(game.trick.size)}. Hidh të njëjtin lloj kombinimi ose një katërshe.`;
  return game.trick ? 'Këto letra nuk mund të luhen mbi ato në tavolinë.' : 'Këto letra nuk formojnë një kombinim të lejuar.';
}
function updateButtons() {
  const action = chosenAction();
  $('newBtn').disabled = busy;
  $('passBtn').disabled = busy || !game?.your_turn || !game?.legal_actions.includes(0) || game?.done;
  $('playBtn').disabled = busy || action === null;
  $('playBtn').innerHTML = busy ? 'Prit pak…' : `${game?.phase==='swap'?'Jep letrën':'Luaj letrat'} <span>→</span>`;
  $('hand').querySelectorAll('button[data-card]').forEach(button => {
    const active = selected.has(button.dataset.card);
    button.classList.toggle('selected',active);
    button.setAttribute('aria-pressed',String(active));
    button.disabled = busy || !game?.your_turn || game?.done;
  });
  $('selectedCount').textContent = selected.size ? `${cardCount(selected.size)} ${selected.size===1?'e zgjedhur':'të zgjedhura'}` : 'Asnjë letër e zgjedhur';
  $('selectionHint').textContent = busy ? 'Lëvizja po kryhet…' : selectionMessage(action);
}
function render(state) {
  game = state; selected.clear(); pending = null;
  storage.set('murlan_game',state.game_id);
  $('error').hidden = true;
  $('handNumber').textContent = `DORA ${state.hand_number}`;
  $('goal').textContent = state.goal_target;
  $('cardCount').textContent = cardCount(state.your_hand.length);
  $('modelLabel').textContent = state.bot_kind === 'strategic' ? 'Robotë me strategji të paracaktuar' : `Robotët · versioni ${state.bot_version}`;
  $('yourScore').textContent = `${state.scores[state.human_seat]} pikë`;
  for (let offset=1; offset<4; offset++) {
    const seat=(state.human_seat+offset)%4, finished=state.finished[seat], passed=state.passed[seat], player=name(seat);
    $(`player${offset}`).innerHTML = `<span class="avatar">${esc(player[0])}</span><div class="opponent-name">${esc(player)}</div><div class="meta">${state.scores[seat]} pikë · ${cardCount(state.hand_sizes[seat])}</div><div class="mini-cards">${Array(Math.min(state.hand_sizes[seat],5)).fill('<span></span>').join('')}</div>${finished||passed?`<div class="passed-label">${finished?'Mbaroi letrat':'Pas'}</div>`:''}`;
  }
  const trick = state.trick?.cards || [];
  $('trickCards').classList.toggle('many',trick.length>5);
  $('trickCards').innerHTML = trick.length ? trick.map(c=>cardHTML(c)).join('') : '<span class="empty-suit">♠</span>';
  $('trickLabel').textContent = state.trick ? `${KINDS[state.trick.kind]} · ${cardCount(state.trick.size)}` : 'Në tavolinë';
  $('tableHint').textContent = state.done ? (state.winner===state.human_seat?'Fitove lojën!':`Loja përfundoi. Fitues: ${name(state.winner)}.`) : state.phase==='swap' ? 'Shkëmbimi i letrave para dorës së re.' : state.trick ? 'Hidh letra më të larta ose thuaj pas.' : state.require_three_spades ? 'Hape lojën me 3♠.' : 'Ti hap. Zgjidh letrat që do të luash.';
  $('turnStatus').textContent = state.done ? 'Loja përfundoi' : state.your_turn ? (state.phase==='swap'?'Radha jote për shkëmbim':'Radha jote') : 'Tani luajnë të tjerët';
  $('actionHelp').textContent = state.done ? 'Nis një lojë të re kur të jesh gati.' : state.phase==='swap' ? 'Kliko letrën që do t’i japësh lojtarit që doli i fundit.' : 'Kliko letrat që do të luash.';
  renderHand();
  const events=[...state.events].reverse();
  $('history').innerHTML = events.length ? events.map(e=>`<li><span class="event-avatar">${esc(name(e.player).slice(0,2).toUpperCase())}</span><div class="event-text"><strong>${esc(name(e.player))}</strong><span>${e.kind==='pass'?'Pas':e.kind==='swap_give'?'Shkëmbim letrash':`${KINDS[e.kind]} · ${e.cards.map(c=>{const p=pretty(c);return esc(p.rank+p.suit)}).join(' ')}`}${e.finish_order?`<br>Renditja e kësaj dore: ${e.finish_order.map(name).map(esc).join(' → ')}`:''}</span></div><span class="event-hand">Dora ${e.hand}</span></li>`).join('') : '<li class="empty-history">Ende nuk është luajtur asnjë letër.</li>';
  updateButtons(); refreshLearning();
}
const ERRORS = {
  'Game not found or expired':'Kjo lojë nuk është më aktive. Nis një lojë të re.',
  'Stale turn; refresh the game':'Loja ka vazhduar. Po rifreskojmë tavolinën.',
  'It is not your turn':'Nuk është radha jote.',
  'Illegal action':'Kjo lëvizje nuk lejohet. Zgjidh letra të tjera.',
  'Unknown selected card':'Një nga letrat e zgjedhura nuk njihet. Rifresko lojën.',
  'Selected cards are not in your hand':'Një nga letrat e zgjedhura nuk është në dorën tënde.',
  'A card cannot be selected twice':'E njëjta letër nuk mund të zgjidhet dy herë.',
  'Selected cards do not match the action':'Këto letra nuk formojnë kombinimin e zgjedhur.',
  'The opening play must include 3S':'Letrat që luan në fillim duhet të përfshijnë 3♠.',
  'Select your lowest card for the exchange':'Për shkëmbimin duhet të japësh letrën më të ulët.',
  'Request ID already used for a different action':'Kjo kërkesë është përdorur tashmë. Rifresko lojën.',
  'Server is at game capacity; try again later':'Ka shumë lojëra në këtë çast. Provo sërish pak më vonë.'
};
async function api(path,options={}) {
  let response;
  try { response=await fetch(path,{...options,headers:{'Content-Type':'application/json',...(options.headers||{})}}); }
  catch { throw new Error('Nuk u lidhëm dot me serverin. Kontrollo lidhjen dhe provo sërish.'); }
  let data;
  try { data=await response.json(); } catch { throw new Error('Serveri nuk u përgjigj siç duhet. Provo sërish.'); }
  if (!response.ok) { const error=new Error(ERRORS[data.detail] || (response.status===422?'Kërkesa nuk është e vlefshme. Rifresko lojën dhe provo sërish.':'Nuk u krye veprimi. Provo sërish.'));error.status=response.status;throw error; }
  return data;
}
function showError(error) { $('error').hidden=false; $('error').textContent=error.message||'Ndodhi një gabim. Provo sërish.'; }
async function play(action,cards) {
  if (busy || !game) return;
  busy=true;updateButtons();
  const signature=JSON.stringify([...cards].sort());
  if (!pending || pending.action!==action || pending.revision!==game.revision || JSON.stringify([...pending.cards].sort())!==signature) pending={action,cards:[...cards],revision:game.revision,request_id:crypto.randomUUID()};
  try { render(await api(`/api/act/${game.game_id}`,{method:'POST',body:JSON.stringify(pending)})); }
  catch(error) {
    showError(error);
    // Resynchronize if a response was lost after the server accepted the move.
    try { const current=await api(`/api/state/${game.game_id}`);if(current.revision!==game.revision)render(current); } catch {}
  } finally { busy=false;updateButtons(); }
}
async function newGame() {
  if(busy)return;busy=true;updateButtons();
  const previous=game?.game_id;
  try {
    const state=await api('/api/new',{method:'POST',body:JSON.stringify({mode:$('modeSelect').value,learning_enabled:$('learnToggle').checked,previous_game_id:previous||null})});
    render(state);
    if(previous)api(`/api/game/${previous}`,{method:'DELETE'}).catch(()=>{});
  } catch(error) { showError(error); } finally { busy=false;updateButtons(); }
}
async function refreshLearning() {
  try {
    const s=await api('/api/learning');
    $('humanMoves').textContent=s.human_moves.toLocaleString();$('completedHands').textContent=s.completed_hands.toLocaleString();
    const labels={waiting:'Në pritje të lojërave të reja',training:'Po trajnohet një version i ri',evaluating:'Versioni i ri po testohet',promoted:'Robotët u përditësuan',retained:'Testimi përfundoi. Vazhdojmë me versionin e mëparshëm.',disabled:'Trajnimi automatik është çaktivizuar',error:'Trajnimi nuk përfundoi. Loja mund të vazhdojë.'};
    $('learningStatus').textContent=game&&!game.learning_enabled?'Lëvizjet e kësaj loje nuk ruhen':labels[s.phase]||'Në pritje';
    const active=['training','evaluating'].includes(s.phase);$('learningDot').className=active?'active':'';
    $('learningProgress').style.width=`${active?75:Math.min(100,100*s.pending_hands/Math.max(1,s.min_hands))}%`;
    $('learningDetail').textContent=s.error?'Versioni i ri nuk u përgatit dot. Robotët vazhdojnë me versionin aktual.':game&&!game.learning_enabled?'Këtë lojë mund ta luash pa ruajtur lëvizjet e tua.':!s.enabled?'Lëvizjet ruhen, por trajnimi automatik nuk është aktiv.':`${s.pending_hands} ${s.pending_hands===1?'dorë në pritje':'duar në pritje'} · ${s.promotions} përditësime. Trajnimi nis kur mblidhen mjaftueshëm lëvizje.`;
  } catch { /* Gameplay does not depend on the training-status request. */ }
}
function updateModeHelp() { $('modeHelp').textContent=$('modeSelect').value==='learn'?'Robotët provojnë herë pas here lëvizje të ndryshme.':'Robotët zgjedhin lëvizjen që vlerësojnë si më të mirën.'; }
$('playBtn').onclick=()=>{const action=chosenAction();if(action)play(action.id,chosenCards());};
$('passBtn').onclick=()=>{if(!$('passBtn').disabled)play(0,[]);};
$('newBtn').onclick=newGame;
$('rulesBtn').onclick=()=>$('rulesDlg').showModal();$('closeRules').onclick=()=>$('rulesDlg').close();
$('rulesDlg').addEventListener('click',e=>{if(e.target===$('rulesDlg')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}});
for(const id of ['learnToggle','modeSelect']) $(id).addEventListener('change',()=>{updateModeHelp();storage.set('murlan_settings',JSON.stringify({learn:$('learnToggle').checked,mode:$('modeSelect').value}));});
(async()=>{
  try {const settings=JSON.parse(storage.get('murlan_settings')||'null');if(settings){$('learnToggle').checked=settings.learn;$('modeSelect').value=settings.mode;}} catch {}
  updateModeHelp();
  const saved=storage.get('murlan_game');
  if(saved){try{render(await api(`/api/state/${saved}`));return;}catch(error){if(error.status!==404){showError(error);return;}storage.remove('murlan_game');}}
  await newGame();
})();
setInterval(refreshLearning,8000);
