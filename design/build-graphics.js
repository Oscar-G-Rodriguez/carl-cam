// Original Carl-Cam vector art. Run only in the selected owner's Figma file.
const page = await figma.getNodeByIdAsync('0:1');
await figma.setCurrentPageAsync(page);
await Promise.all(['Regular', 'Bold', 'Bold Italic'].map(style => figma.loadFontAsync({family:'Arimo',style})));
const variables = await figma.variables.getLocalVariablesAsync();
const vars = Object.fromEntries(variables.map(v => [v.name, v]));
const styles = Object.fromEntries((await figma.getLocalTextStylesAsync()).map(s => [s.name, s]));
const created = [], components = {}, menuVariants = [];
const paint = name => figma.variables.setBoundVariableForPaint({type:'SOLID',color:{r:0,g:0,b:0}},'color',vars['color/'+name]);
const track = node => {created.push(node.id); return node;};
const svg = content => '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" viewBox="0 0 640 360">'+content+'</svg>';
const art = {};
art.red = svg('<path d="M32 32 L178 25 L332 31 L480 28 L608 32 L615 166 L609 328 L456 335 L312 328 L160 334 L32 328 L24 202 Z" fill="none" stroke="#111014" stroke-width="20"/>'+ '<path d="M35 26 L210 29 L420 31 L608 32 M35 328 L206 330 L448 329 L609 335" fill="none" stroke="#F1ECE2" stroke-width="7"/>'+ '<path d="M65 36 L280 37 M337 336 L557 336 M28 70 L25 184 M614 216 L615 301" fill="none" stroke="#C21E29" stroke-width="5"/>'+ '<path d="M14 17 L50 12 L53 47 L19 53 Z M593 13 L628 23 L620 56 L587 47 Z M592 309 L629 321 L621 355 L585 342 Z M14 309 L50 315 L46 350 L9 341 Z" fill="#F1ECE2" stroke="#111014" stroke-width="4"/>');
let outer='M20 32 L20 20 L40 20';
for(let x=40;x<572;x+=28) outer+=' Q'+(x+14)+' -2 '+(x+28)+' 20';
outer+=' L620 20 L620 340 L600 340';
for(let x=600;x>68;x-=28) outer+=' Q'+(x-14)+' 362 '+(x-28)+' 340';
outer+=' L20 340 Z M44 44 L44 316 L596 316 L596 44 Z';
art.yellow=svg('<path d="'+outer+'" fill="#F5C915" fill-rule="evenodd" stroke="#25162F" stroke-width="3"/>'+ '<path d="M34 94 L34 122 M606 226 L606 254" stroke="#4AC9E8" stroke-width="8"/>'+[ [32,32],[608,32],[608,328],[32,328]].map(([x,y])=>'<circle cx="'+x+'" cy="'+y+'" r="16" fill="#F16AA5" stroke="#25162F" stroke-width="5"/>').join(''));
art.blue=svg('<path d="M32 32 L240 32 M278 32 L435 32 M467 32 L608 32 M608 32 L608 155 M608 189 L608 328 M608 328 L449 328 M412 328 L253 328 M219 328 L32 328 M32 328 L32 197 M32 161 L32 32" fill="none" stroke="#4AC9E8" stroke-width="12"/>'+ '<path d="M278 32 L435 32 M32 91 L32 168 M608 198 L608 277 M260 328 L420 328" stroke="#8BEAF4" stroke-width="9"/>'+[ [32,32],[608,32],[608,328],[32,328]].map(([x,y])=>'<path d="M'+(x-16)+' '+(y-16)+' H'+(x+16)+' V'+(y-1)+' H'+x+' V'+(y+16)+' H'+(x-16)+' Z" fill="#4AC9E8" stroke="#092457" stroke-width="4"/>').join(''));
function bindArt(root){
 for(const n of root.findAll(()=>true)){
  created.push(n.id);
  for(const field of ['fills','strokes']) if(field in n && Array.isArray(n[field])) n[field]=n[field].map(p=>{
   if(p.type!=='SOLID')return p;
   const match=variables.find(v=>v.resolvedType==='COLOR'&&Object.values(v.valuesByMode).some(c=>c&&typeof c==='object'&&'r'in c&&Math.abs(c.r-p.color.r)<.004&&Math.abs(c.g-p.color.g)<.004&&Math.abs(c.b-p.color.b)<.004));
   return match?figma.variables.setBoundVariableForPaint(p,'color',match):p;
  });
 }
}
for(const [name,source]of Object.entries(art)){
 let component=page.children.find(n=>n.type==='COMPONENT'&&n.name==='Border/'+name);
 if(!component){component=track(figma.createComponent());component.name='Border/'+name;component.resize(640,360);component.x=100+Object.keys(components).length*700;component.y=100;component.fills=[];component.description='Original '+name+' frame. Canonical hand-quad anchors: (32,32), (608,32), (608,328), (32,328). Warp artwork, not camera pixels. No decorative text.';
 const vectors=track(figma.createNodeFromSvg(source));component.appendChild(vectors);vectors.x=0;vectors.y=0;vectors.name='Editable border paths';bindArt(vectors);}
 components[name]=component.id;
}
for(const [theme,subtitle,accent]of [['red','COLLAGE','red'],['yellow','POP','yellow'],['blue','PANELS','cyan']])for(const state of ['Default','Hover','Focused','Selected']){
 const name='Theme='+theme+', State='+state;
 let c=page.children.find(n=>n.type==='COMPONENT'&&n.name===name);
 if(!c){c=track(figma.createComponent());c.name=name;c.resize(300,100);c.x=2300+(menuVariants.length%4)*340;c.y=100+Math.floor(menuVariants.length/4)*140;c.fills=[];c.layoutMode='VERTICAL';c.primaryAxisSizingMode='FIXED';c.counterAxisSizingMode='FIXED';c.setBoundVariable('paddingLeft',vars['spacing/lg']);c.setBoundVariable('paddingTop',vars['spacing/md']);c.setBoundVariable('itemSpacing',vars['spacing/sm']);
 const shape=track(figma.createNodeFromSvg('<svg width="300" height="100" viewBox="0 0 300 100" xmlns="http://www.w3.org/2000/svg"><path d="M0 8 L275 0 L300 84 L18 100 Z" fill="#111014"/><path d="M0 8 L275 0 L277 8 L2 16 Z" fill="'+({red:'#C21E29',yellow:'#F5C915',cyan:'#4AC9E8'})[accent]+'"/></svg>'));c.appendChild(shape);shape.layoutPositioning='ABSOLUTE';shape.x=0;shape.y=0;bindArt(shape);
 for(const [text,role]of [[theme.toUpperCase(),'Menu title'],[subtitle,'Caption']]){const t=track(figma.createText());t.name=role;t.textStyleId=styles['Carl-Cam/'+role].id;t.characters=text;t.fills=[paint('cream')];c.appendChild(t);}
 if(state!=='Default'){const cue=track(figma.createRectangle());c.appendChild(cue);cue.layoutPositioning='ABSOLUTE';cue.resize(292,92);cue.x=4;cue.y=4;cue.fills=[];cue.strokeWeight=3;cue.strokes=[paint(state==='Focused'?'yellow':state==='Hover'?'cyan':accent)];cue.name=state+' interaction cue';}
 }
 menuVariants.push(c.id);
}
return {createdNodeIds:created,borders:components,menuVariants,counts:{borders:3,menuVariants:menuVariants.length},anchorRect:[32,32,576,296]};
