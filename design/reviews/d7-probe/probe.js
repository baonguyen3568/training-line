(() => {
  const vis = el => { const r = el.getBoundingClientRect(); const cs = getComputedStyle(el); return r.width>0 && r.height>0 && cs.visibility!=='hidden' && cs.display!=='none'; };
  const name = el => (el.getAttribute('aria-label')||el.getAttribute('aria-labelledby')&&document.getElementById(el.getAttribute('aria-labelledby'))?.textContent||el.textContent||el.getAttribute('title')||el.getAttribute('placeholder')||'').trim();
  const sel = el => { let s = el.tagName.toLowerCase(); if (el.className && typeof el.className==='string') s += '.' + el.className.trim().split(/\s+/).slice(0,2).join('.'); return s; };
  const inter = [...document.querySelectorAll('button,a[href],input,select,textarea,[role=button],[role=tab],[role=switch],[tabindex]:not([tabindex="-1"])')].filter(vis);
  const noName = inter.filter(e => !name(e) && !(e.labels && e.labels.length)).map(e => sel(e));
  const small = inter.filter(e => { const r=e.getBoundingClientRect(); return r.width<24||r.height<24; }).map(e => sel(e)+' '+Math.round(e.getBoundingClientRect().width)+'x'+Math.round(e.getBoundingClientRect().height)+' "'+name(e).slice(0,20)+'"');
  const parse = c => { const m=c.match(/[\d.]+/g); return m? m.map(Number):[0,0,0,0]; };
  const lum = ([r,g,b]) => { const f=v=>{v/=255;return v<=.03928?v/12.92:Math.pow((v+.055)/1.055,2.4)}; return .2126*f(r)+.7152*f(g)+.0722*f(b); };
  const bgOf = el => { let e=el; let layers=[]; while(e){ const c=parse(getComputedStyle(e).backgroundColor); if(c.length===3) c.push(1); if(c[3]>0){ layers.push(c); if(c[3]>=1) break;} e=e.parentElement; } let base=[0,0,0]; for(const c of layers.reverse()){ base=base.map((v,i)=>v*(1-c[3])+c[i]*c[3]); } return base; };
  const low = []; const seen=new Set();
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) { const t=walker.currentNode; if(!t.textContent.trim()) continue; const el=t.parentElement; if(!vis(el)||seen.has(el)) continue; seen.add(el);
    const cs=getComputedStyle(el); let fg=parse(cs.color); if(fg.length===3) fg.push(1); const op=parseFloat(cs.opacity); const bg=bgOf(el); const a=fg[3]*op; const f=fg.slice(0,3).map((v,i)=>v*a+bg[i]*(1-a));
    const L1=lum(f),L2=lum(bg); const ratio=(Math.max(L1,L2)+.05)/(Math.min(L1,L2)+.05); const px=parseFloat(cs.fontSize); const big=px>=24||(px>=18.66&&parseInt(cs.fontWeight)>=700);
    if(ratio < (big?3:4.5) && !el.closest('[disabled],[aria-disabled=true]')) low.push(sel(el)+' "'+t.textContent.trim().slice(0,24)+'" '+ratio.toFixed(2)+' '+px+'px'); }
  const heads=[...document.querySelectorAll('h1,h2,h3,h4')].filter(vis).map(h=>h.tagName+':'+h.textContent.trim().slice(0,24));
  const lm={main:document.querySelectorAll('main,[role=main]').length,nav:document.querySelectorAll('nav,[role=navigation]').length,header:document.querySelectorAll('header,[role=banner]').length};
  const tiny=[...new Set([...document.querySelectorAll('body *')].filter(e=>vis(e)&&e.childNodes.length&&[...e.childNodes].some(n=>n.nodeType===3&&n.textContent.trim())&&parseFloat(getComputedStyle(e).fontSize)<11).map(e=>sel(e)+' '+getComputedStyle(e).fontSize))].slice(0,15);
  return {interactive:inter.length,noName,small:small.slice(0,25),smallCount:small.length,lowContrast:low.slice(0,25),lowCount:low.length,heads,lm,tiny,lang:document.documentElement.lang};
})()
