import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';

const manual = await readFile(new URL('./manual.js', import.meta.url), 'utf8');
const {ManualStream} = await import('data:text/javascript;base64,' + Buffer.from(manual).toString('base64'));
const source = (await readFile(new URL('./app.js', import.meta.url), 'utf8')).replace(/^import .*;\n/, '');
const flush = () => new Promise(resolve => setImmediate(resolve));

async function dashboard() {
 const nodes=new Map(), documentEvents={},windowEvents={},requests=[];
 function element(){return {listeners:{},children:[],checked:false,disabled:false,
  classList:{remove(){},toggle(){}},addEventListener(n,f){this.listeners[n]=f;},
  append(...v){this.children.push(...v);},replaceChildren(...v){this.children=v;},
  setPointerCapture(){},elements:{namedItem:id=>get('form-'+id)}};}
 function get(id){if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);}
 const motions=['forward','back','left','right','up','down','aux'].map(direction=>{const n=element();n.dataset={direction};return n;});
 const state={input_generation:'100',connected:true,age_ms:0,input_inhibited:false,pause_requested:false,estop_requested:false};
 const c=vm.createContext({ManualStream,Date,
  document:{getElementById:get,querySelectorAll:()=>motions,createElement:element,addEventListener(n,f){documentEvents[n]=f;}},
  window:{addEventListener(n,f){windowEvents[n]=f;}},setTimeout(){},setInterval(){},
  fetch:async(path,options)=>{requests.push([path,options]);return {ok:true,json:async()=>path==='/api/state'?state:path==='/api/captures'?{configured:false}:{receipt:{status:'QUEUED_BENCH'}}};}});
 vm.runInContext(source+'\nglobalThis.inspect=()=>({enabled,current,axes:stream.axes,inputGeneration,deadmanSource});globalThis.poll=pollState;',c);
 await flush();
 function enable(){get('enable').checked=true;get('enable').listeners.change();}
 function key(type,key,extra={}){documentEvents[type]({key,code:key===' '?'Space':key,target:{tagName:'BODY'},preventDefault(){},...extra});}
 function pointer(node,event,id){node.listeners[event]({pointerId:id,preventDefault(){}});}
 const payloads=()=>requests.filter(([p])=>p==='/api/manual').map(([,o])=>JSON.parse(o.body));
 return {c,get,motions,state,requests,enable,key,pointer,payloads,windowEvents};
}

test('a service generation change clears held input before using the new token',async()=>{
 const d=await dashboard();d.enable();d.key('keydown',' ');d.key('keydown','w');await flush();
 assert.equal(d.c.inspect().axes.forward,.2);
 await d.c.poll();assert.equal(d.c.inspect().enabled,true);
 d.state.input_generation='200';await d.c.poll();
 assert.equal(d.c.inspect().enabled,false);assert.equal(d.c.inspect().axes,null);
 assert.equal(d.c.inspect().deadmanSource,null);assert.ok(d.motions.every(n=>n.disabled));
 const count=d.payloads().length;await vm.runInContext('stream.tick()',d.c);assert.equal(d.payloads().length,count);
});
test('continuous deadman supports neutral -> movement -> neutral -> disarm',async()=>{
 const d=await dashboard();d.enable();
 d.key('keydown','r');await flush();assert.equal(d.payloads().length,0);
 d.key('keydown',' ');await flush();
 d.key('keydown','r');await flush();
 d.key('keyup','r');await flush();
 d.key('keyup',' ');await flush();
 assert.deepEqual(d.payloads().map(p=>[p.hold,p.up]),[[true,0],[true,.2],[true,0],[false,0]]);
 assert.equal(d.c.inspect().axes,null);assert.ok(d.motions.every(n=>n.disabled));
});
test('independent pointer ownership preserves deadman during direction release',async()=>{
 const d=await dashboard();d.enable();const arm=d.get('arm-neutral'),up=d.motions.find(n=>n.dataset.direction==='up');
 d.pointer(arm,'pointerdown',1);await flush();d.pointer(up,'pointerdown',2);await flush();
 d.pointer(up,'lostpointercapture',3);assert.equal(d.c.inspect().current,'up');
 d.pointer(up,'pointerup',2);await flush();assert.equal(d.c.inspect().deadmanSource,1);assert.equal(d.c.inspect().axes.up,0);
 d.pointer(arm,'lostpointercapture',1);await flush();assert.equal(d.payloads().at(-1).hold,false);
 d.pointer(up,'pointerup',2);await flush();assert.equal(d.c.inspect().axes,null);
});
test('blur and controller faults disable controls and require a fresh deadman press',async()=>{
 const d=await dashboard();d.enable();d.key('keydown',' ');d.key('keydown','w');await flush();
 d.windowEvents.blur();await flush();assert.equal(d.payloads().at(-1).hold,false);
 d.enable();d.key('keydown',' ',{repeat:true});assert.equal(d.c.inspect().axes,null);
 d.key('keyup',' ');d.key('keydown',' ');await flush();
 d.state.telemetry={controller_status:'status_unavailable input_reset_required=true'};await d.c.poll();await flush();
 assert.equal(d.c.inspect().enabled,false);assert.equal(d.payloads().at(-1).hold,false);
});
test('Pause bypasses held input and Resume never restores it',async()=>{
 const d=await dashboard();d.enable();d.key('keydown',' ');d.key('keydown','w');await flush();
 d.get('pause').onclick();await flush();assert.equal(d.c.inspect().axes,null);assert.equal(d.c.inspect().enabled,false);
 assert.ok(d.requests.some(([p])=>p==='/api/pause'));
 d.state.input_inhibited=true;d.state.pause_requested=true;await d.c.poll();
 d.get('resume').onclick();await flush();assert.equal(d.c.inspect().axes,null);
 assert.ok(d.requests.some(([p])=>p==='/api/resume'));
});

test('keyboard deadman works immediately after clicking Enable controls',async()=>{
 const d=await dashboard();d.enable();const target=d.get('enable');target.tagName='INPUT';
 d.key('keydown',' ',{target});await flush();d.key('keydown','r',{target});await flush();
 assert.equal(d.c.inspect().axes.up,.2);
 d.key('keyup',' ',{target});await flush();assert.equal(d.payloads().at(-1).hold,false);
 d.key('keydown',' ',{target:{tagName:'INPUT'}});await flush();assert.equal(d.c.inspect().axes,null);
});
