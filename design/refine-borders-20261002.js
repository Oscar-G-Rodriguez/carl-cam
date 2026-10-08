const page=await figma.getNodeByIdAsync('0:1');
await figma.setCurrentPageAsync(page);
const ids=['7:8','7:9','7:10','7:11','7:12','7:13','7:14','7:15','7:16','7:20','7:21','7:22','7:23'];
const fetched=await Promise.all(ids.map(id=>figma.getNodeByIdAsync(id)));
if(fetched.some(n=>!n)) throw new Error('A recorded border node is missing; no changes made.');
const nodes=Object.fromEntries(fetched.map(n=>[n.id,n]));
const cream=await figma.variables.getVariableByIdAsync('VariableID:6:9');
if(!cream) throw new Error('Existing cream token is missing; no changes made.');
const mutated=[]; const created=[];
let outer='M 0 20 Q 0 8 12 8 L 40 8';
for(let x=40;x<560;x+=52) outer+=' C '+(x+9)+' 8 '+(x+9)+' 0 '+(x+26)+' 0 C '+(x+43)+' 0 '+(x+43)+' 8 '+(x+52)+' 8';
outer+=' L 588 8 Q 600 8 600 20 L 600 316 Q 600 328 588 328 L 560 328';
for(let x=560;x>40;x-=52) outer+=' C '+(x-9)+' 328 '+(x-9)+' 336 '+(x-26)+' 336 C '+(x-43)+' 336 '+(x-43)+' 328 '+(x-52)+' 328';
outer+=' L 12 328 Q 0 328 0 316 Z M 32 32 L 568 32 Q 576 32 576 40 L 576 296 Q 576 304 568 304 L 32 304 Q 24 304 24 296 L 24 40 Q 24 32 32 32 Z';
const band=nodes['7:10'];
band.vectorPaths=[{windingRule:'EVENODD',data:outer}];
band.x=20; band.y=12; band.strokeWeight=3; band.strokeJoin='ROUND'; band.name='Pop / soft scalloped yellow band';
mutated.push(band.id);
const accent=nodes['7:11'];
accent.vectorPaths=[{windingRule:'NONZERO',data:'M 0 0 L 0 22 M 576 120 L 576 142'}];
accent.x=32; accent.y=96; accent.fills=[]; accent.strokeCap='ROUND'; accent.name='Pop / cyan capsule interruptions';
mutated.push(accent.id);
const circle='M 28 14 C 28 21.732 21.732 28 14 28 C 6.268 28 0 21.732 0 14 C 0 6.268 6.268 0 14 0 C 21.732 0 28 6.268 28 14 Z';
const corners=[['7:12',18,18,'top-left'],['7:13',594,18,'top-right'],['7:14',594,314,'bottom-right'],['7:15',18,314,'bottom-left']];
for(const [id,x,y,label] of corners){const n=nodes[id];n.vectorPaths=[{windingRule:'NONZERO',data:circle}];n.x=x;n.y=y;n.strokeWeight=4;n.name='Pop / pink '+label+' bubble';mutated.push(id);}
let keyline=nodes['7:9'].children.find(n=>n.name==='Pop / cream inner keyline');
if(!keyline){keyline=figma.createVector();nodes['7:9'].appendChild(keyline);created.push(keyline.id);}
keyline.name='Pop / cream inner keyline';keyline.vectorPaths=[{windingRule:'NONZERO',data:'M 10 0 L 550 0 Q 560 0 560 10 L 560 270 Q 560 280 550 280 L 10 280 Q 0 280 0 270 L 0 10 Q 0 0 10 0 Z'}];
keyline.x=40;keyline.y=40;keyline.fills=[];keyline.strokes=[figma.variables.setBoundVariableForPaint({type:'SOLID',color:{r:1,g:0.95686275,b:0.78431373}},'color',cream)];keyline.strokeWeight=2;keyline.strokeJoin='ROUND';mutated.push(keyline.id);
const blueCorners=[['7:20',false,false,'top-left'],['7:21',true,false,'top-right'],['7:22',true,true,'bottom-right'],['7:23',false,true,'bottom-left']];
const points=[[0,0],[32,0],[32,15],[16,15],[16,32],[0,32]];
for(const [id,flipX,flipY,label] of blueCorners){const n=nodes[id];const p=points.map(([x,y])=>[flipX?32-x:x,flipY?32-y:y]);n.vectorPaths=[{windingRule:'NONZERO',data:p.map(([x,y],i)=>(i?'L ':'M ')+x+' '+y).join(' ')+' Z'}];n.name='Blue / '+label+' outward bracket';mutated.push(id);}
nodes['7:8'].description+=' October 2 refinement: ten shallow smooth scallops per horizontal edge, cream inner keyline, balanced round cyan accents and 28px pink corner bubbles. Fixed canonical anchors retained.';
nodes['7:16'].description+=' October 2 correction: brackets mirror independently on both axes so each corner points outward and opens toward the live-image interior.';
mutated.push('7:8','7:16');
await nodes['7:8'].screenshot({scale:1});
await nodes['7:16'].screenshot({scale:1});
return {createdNodeIds:created,mutatedNodeIds:mutated,components:[{id:'7:8',width:nodes['7:8'].width,height:nodes['7:8'].height,childCount:nodes['7:9'].children.length},{id:'7:16',width:nodes['7:16'].width,height:nodes['7:16'].height,cornerOrientations:blueCorners.map(c=>({id:c[0],flipX:c[1],flipY:c[2]}))}],canonicalAnchors:[[32,32],[608,32],[608,328],[32,328]]};
