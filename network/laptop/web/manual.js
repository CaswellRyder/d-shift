// One in-flight command; holds are sampled, never queued.
export class ManualStream {
 constructor(send, report, cancel) { this.send=send; this.report=report; this.cancel=cancel; this.axes=null; this.busy=false; this.releasePending=false; this.safety=null; this.generation=0; }
 hold(axes) { this.axes=axes; }
 release() { this.axes=null; this.releasePending=true; this.tick(); }
 stop(path) {
  this.axes=null; this.releasePending=false; this.safety=null;
  const generation=++this.generation;
  // Safety bypasses an outstanding manual HTTP exchange so the backend can
  // interrupt it. A late manual receipt cannot overwrite the safety result.
  this.report('Pending bench receipt; local input cleared.');
  Promise.resolve(this.send(path,{})).then(result=>{
   if(generation===this.generation)this.report(`Bench receipt: ${result.receipt?.status??'received'}. No motor acknowledgement.`);
  }).catch(error=>{if(generation===this.generation){this.cancel();this.report(`Safety receipt unknown: ${error.message}. Local input remains cleared.`);}});
 }
 async tick() {
  if(this.busy) return;
  let path='/api/manual', payload;
  if(this.safety) { path=this.safety; this.safety=null; payload={}; }
  else if(this.releasePending) {this.releasePending=false; payload={forward:0,right:0,up:0,yaw:0,hold:false};}
  else if(this.axes) payload={...this.axes,hold:true};
  else return;
  const generation=this.generation; this.busy=true;
  try { this.report('Pending bench receipt'); const result=await this.send(path,payload); if(generation!==this.generation)return; this.report(`Bench receipt: ${result.receipt?.status ?? 'received'}${result.receipt?.code ? ' / '+result.receipt.code : ''}. No motor acknowledgement.`); }
  catch(error) { if(generation!==this.generation)return; this.axes=null; this.cancel(); this.report(`Receipt error: ${error.message}. Controls disabled; no automatic resume.`); if(payload.hold) this.releasePending=true; }
  finally { this.busy=false; if(this.safety || this.releasePending) this.tick(); }
 }
}
