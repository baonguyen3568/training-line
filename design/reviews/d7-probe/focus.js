(async () => { const out=[]; const els=[...document.querySelectorAll('button,a[href],input,select,[role=tab],[tabindex="0"]')].filter(e=>{const r=e.getBoundingClientRect();return r.width>0&&r.height>0}).slice(0,60);
 for (const e of els){ e.focus({focusVisible:true}); await new Promise(r=>setTimeout(r,10)); const cs=getComputedStyle(e); const has=(cs.outlineStyle!=='none'&&parseFloat(cs.outlineWidth)>0)||cs.boxShadow!=='none'; if(!has) out.push((e.getAttribute('aria-label')||e.textContent||e.tagName).trim().slice(0,24)); }
 return {checked:els.length,noFocusRing:out}; })()
