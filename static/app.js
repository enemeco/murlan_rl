'use strict';
const $=id=>document.getElementById(id);
const NAMES=['Dritan','Ardi','Bora','Luan'];
const KINDS={single:'Tek',pair:'Dyshe',triple:'Treshe',quad:'Katërshe',straight:'Kolor',swap_give:'Shkëmbim',pass:'Kalon'};
let game=null,selected=null,busy=false,pending=null;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function name(seat){return seat===game?.human_seat?'Ti':NAMES[seat]||`Boti ${seat}`}
function pretty(card){
  if(card==='BJ'||card==='RJ')return {rank:'JOK',suit:'✦',red:card==='RJ',joker:true,label:card==='RJ'?'Xhol i kuq':'Xhol i zi'};
  const suit=card.slice(-1),symbol={S:'♠',H:'♥',D:'♦',C:'♣'}[suit];
  return {rank:symbol?card.slice(0,-1):card,suit:symbol||'♠',red:suit==='H'||suit==='D',joker:false,label:card};
}
function cardHTML(card,highlight=false){const c=pretty(card);return `<div class="playing-card ${c.red?'red':''} ${c.joker?'joker':''} ${highlight?'selected':''}" aria-label="${esc(c.label)}"><span class="corner">${esc(c.rank)}</span><span class="suit">${c.suit}</span><span class="corner lower">${esc(c.rank)}</span></div>`}
function rankOf(card){return card==='BJ'||card==='RJ'?card:card.slice(0,-1)}
function chosenCards(){
  const option=game?.actions.find(a=>a.id===selected);if(!option)return [];
  const count={single:1,pair:2,triple:3,quad:4,swap_give:1}[option.kind];
  const needed=option.kind==='straight'?option.ranks:Array(count||0).fill(option.ranks[0]);
  const hand=[...game.your_hand],chosen=[];
  // The opening rank-three action uses the forced 3 of spades.
  if(game.hand_number===1&&game.events.length===0&&needed.includes('3')&&hand.includes('3S')){hand.splice(hand.indexOf('3S'),1);hand.unshift('3S')}
  for(const rank of needed){const i=hand.findIndex(c=>rankOf(c)===rank);if(i>=0)chosen.push(hand.splice(i,1)[0])}
  return chosen;
}
function renderHand(){const chosen=chosenCards();$('hand').innerHTML=game.your_hand.length?game.your_hand.map(c=>cardHTML(c,chosen.includes(c))).join(''):'<p class="muted">I mbarove letrat për këtë dorë.</p>';}
function updateButtons(){
  $('newBtn').disabled=busy;
  $('passBtn').disabled=busy||!game?.legal_actions.includes(0)||game?.done;
  $('playBtn').disabled=busy||selected===null||game?.done;
  $('playBtn').innerHTML=busy?'Duke luajtur…':`${game?.phase==='swap'?'Shkëmbe letrën':'Luaj letrat'} <span>→</span>`;
  document.querySelectorAll('.option').forEach(b=>b.disabled=busy);
}
function render(state){
  game=state;selected=null;pending=null;localStorage.setItem('murlan_game',state.game_id);
  $('error').hidden=true;$('handNumber').textContent=`DORA ${String(state.hand_number).padStart(2,'0')}`;
  $('goal').textContent=state.goal_target;$('cardCount').textContent=`${state.your_hand.length} letra`;
  $('modelLabel').textContent=state.bot_kind==='strategic'?'Botë strategjikë':`Botë · versioni ${state.bot_version}`;
  $('yourScore').textContent=`${state.scores[state.human_seat]} pikë`;
  for(let offset=1;offset<4;offset++){
    const seat=(state.human_seat+offset)%4;
    const finished=state.finished[seat],passed=state.passed[seat];
    $(`player${offset}`).innerHTML=`<span class="avatar">${NAMES[seat][0]}</span><div class="opponent-name">${NAMES[seat]}</div><div class="meta">${state.scores[seat]} pikë · ${state.hand_sizes[seat]} letra</div><div class="mini-cards">${Array(Math.min(state.hand_sizes[seat],5)).fill('<span></span>').join('')}</div>${finished||passed?`<div class="passed-label">${finished?'Përfundoi':'Kaloi'}</div>`:''}`;
  }
  let trick=[];
  if(state.trick){
    if(state.trick.cards?.length){trick=state.trick.cards}
    else if(state.trick.kind==='straight'){trick=state.trick.ranks;if(trick.join(',')==='3,4,5,A,2')trick=['A','2','3','4','5']}
    else trick=Array(state.trick.size).fill(state.trick.ranks[0]);
  }
  $('trickCards').classList.toggle('many',trick.length>5);
  $('trickCards').innerHTML=trick.length?trick.map(c=>cardHTML(c)).join(''):'<span class="empty-suit">♠</span>';
  $('trickLabel').textContent=state.trick?`${KINDS[state.trick.kind]} · ${state.trick.size} letra`:'TAVOLINA ËSHTË E LIRË';
  $('tableHint').textContent=state.done?`${name(state.winner)} ${state.winner===state.human_seat?'fitove':'fitoi'} lojën!`:state.phase==='swap'?'Shkëmbimi mes të parit dhe të fundit.':state.trick?'Hidh një kombinim më të lartë ose kalo.':'Hap me një kombinim nga dora jote.';
  $('turnStatus').textContent=state.done?'Loja përfundoi':state.your_turn?(state.phase==='swap'?'Zgjidh letrën për shkëmbim':'Është radha jote'):'Botët po luajnë…';
  $('actionHelp').textContent=state.done?'Nis një lojë të re për të luajtur përsëri.':state.phase==='swap'?'Zgjidh letrën që i jep lojtarit të fundit.':'Zgjidh kombinimin që dëshiron të luash.';
  $('selectionHint').textContent='Zgjidh një kombinim';$('selectedCount').textContent='';
  renderHand();
  const options=state.actions.filter(a=>a.kind!=='pass');
  $('options').replaceChildren();
  for(const a of options){
    const b=document.createElement('button');b.className='option';b.setAttribute('aria-pressed','false');
    b.innerHTML=`<strong>${esc(a.ranks.join(' · '))}</strong>${KINDS[a.kind]||esc(a.kind)}`;
    b.onclick=()=>{selected=a.id;document.querySelectorAll('.option').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));renderHand();$('selectionHint').textContent=`${KINDS[a.kind]} · ${a.ranks.join(' · ')}`;$('selectedCount').textContent=`${chosenCards().length} letra të zgjedhura`;updateButtons()};
    $('options').append(b);
  }
  if(!options.length&&!state.done)$('actionHelp').textContent='Nuk ke kombinim më të lartë. Kalo radhën.';
  const events=[...state.events].reverse();
  $('history').innerHTML=events.length?events.map(e=>`<li><span class="event-avatar">${esc(name(e.player).slice(0,2).toUpperCase())}</span><div class="event-text"><strong>${esc(name(e.player))}</strong><span>${e.kind==='pass'?'Kaloi radhën':e.kind==='swap_give'?'Shkëmbeu një letër':`${KINDS[e.kind]} · ${e.cards.map(c=>{const p=pretty(c);return esc(p.rank+p.suit)}).join(' ')}`}${e.finish_order?`<br>Dora përfundoi: ${e.finish_order.map(name).map(esc).join(' → ')}`:''}</span></div><span class="event-hand">D${e.hand}</span></li>`).join(''):'<li class="empty-history">Historia e lojës shfaqet këtu.</li>';
  updateButtons();refreshLearning();
}
async function api(path,options={}){const response=await fetch(path,{...options,headers:{'Content-Type':'application/json',...(options.headers||{})}});let data;try{data=await response.json()}catch{throw new Error('Serveri nuk u përgjigj siç duhet. Provo përsëri.')}if(!response.ok){const error=new Error(typeof data.detail==='string'?data.detail:'Kërkesa nuk u pranua.');error.status=response.status;throw error}return data}
function showError(error){$('error').hidden=false;$('error').textContent=error.message||String(error)}
async function play(action){
  if(busy||!game)return;busy=true;updateButtons();
  if(!pending||pending.action!==action)pending={action,revision:game.revision,request_id:crypto.randomUUID()};
  try{render(await api(`/api/act/${game.game_id}`,{method:'POST',body:JSON.stringify(pending)}))}
  catch(error){
    showError(error);
    // A response can be lost after the server accepted the action. Resynchronize.
    try{const current=await api(`/api/state/${game.game_id}`);if(current.revision!==game.revision)render(current)}catch{}
  }finally{busy=false;updateButtons()}
}
async function newGame(){
  if(busy)return;busy=true;updateButtons();
  const previous=game?.game_id;
  try{const state=await api('/api/new',{method:'POST',body:JSON.stringify({mode:$('modeSelect').value,learning_enabled:$('learnToggle').checked})});render(state);if(previous)api(`/api/game/${previous}`,{method:'DELETE'}).catch(()=>{})}
  catch(error){showError(error)}finally{busy=false;updateButtons()}
}
async function refreshLearning(){
  try{
    const s=await api('/api/learning');
    $('humanMoves').textContent=s.human_moves.toLocaleString();$('completedHands').textContent=s.completed_hands.toLocaleString();
    const labels={waiting:'Në pritje të duarve të reja',training:'Po stërvitet një version i ri',evaluating:'Po vlerësohet versioni i ri',promoted:'Një version më i mirë u aktivizua',retained:'Vlerësimi përfundoi; botët u ruajtën',disabled:'Stërvitja automatike është e fikur',error:'Stërvitja kërkon kontroll'};
    $('learningStatus').textContent=game&&!game.learning_enabled?'Ruajtja është e fikur për këtë lojë':labels[s.phase]||s.phase;
    const active=['training','evaluating'].includes(s.phase);$('learningDot').className=active?'active':'';
    $('learningProgress').style.width=`${active?75:Math.min(100,100*s.pending_hands/Math.max(1,s.min_hands))}%`;
    $('learningDetail').textContent=s.error||(!s.enabled?'Lëvizjet ruhen për stërvitje të mëvonshme.':`${s.pending_hands} duar në pritje · ${s.promotions} versione të aktivizuara. Përditësimi nis kur mblidhen mjaftueshëm duar.`);
  }catch{/* Play remains available if the status request fails. */}
}
$('playBtn').onclick=()=>selected!==null&&play(selected);$('passBtn').onclick=()=>play(0);$('newBtn').onclick=newGame;
$('rulesBtn').onclick=()=>$('rulesDlg').showModal();$('closeRules').onclick=()=>$('rulesDlg').close();
$('rulesDlg').addEventListener('click',e=>{if(e.target===$('rulesDlg')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close()}});
for(const id of ['learnToggle','modeSelect']){$(id).addEventListener('change',()=>localStorage.setItem('murlan_settings',JSON.stringify({learn:$('learnToggle').checked,mode:$('modeSelect').value})))}
(async()=>{
  try{const settings=JSON.parse(localStorage.getItem('murlan_settings')||'null');if(settings){$('learnToggle').checked=settings.learn;$('modeSelect').value=settings.mode}}
  catch{}
  const saved=localStorage.getItem('murlan_game');
  if(saved){try{render(await api(`/api/state/${saved}`));return}catch(error){if(error.status!==404){showError(error);return}localStorage.removeItem('murlan_game')}}
  await newGame();
})();
setInterval(refreshLearning,8000);
