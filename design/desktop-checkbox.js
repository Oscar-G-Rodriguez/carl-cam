// Scoped Checkbox family; variant-grid workflow adapted from the bundled
// createComponentWithVariants helper. Existing Carl-Cam foundations are reused.
const page=await figma.getNodeByIdAsync('0:1');
await figma.setCurrentPageAsync(page);
const existing=page.findAllWithCriteria({types:['COMPONENT_SET']})
  .find(n=>n.name==='Desktop/Checkbox');
if(existing) return {status:'already_exists',componentSetId:existing.id};
await figma.loadFontAsync({family:'Arimo',style:'Regular'});
const allVars=await figma.variables.getLocalVariablesAsync();
const vars=new Map(allVars.map(v=>[v.name,v]));
const created=[],createdVariables=[];
const collection=await figma.variables.getVariableCollectionByIdAsync('VariableCollectionId:6:2');
if(!vars.has('radius/checkbox')) {
  const v=figma.variables.createVariable('radius/checkbox',collection,'FLOAT');
  v.scopes=['CORNER_RADIUS'];v.setValueForMode(collection.defaultModeId,4);
  v.setVariableCodeSyntax('WEB','CONTROL_CHECKBOX_RADIUS');
  vars.set(v.name,v);createdVariables.push(v.id);
}
function paint(name){return figma.variables.setBoundVariableForPaint(
  {type:'SOLID',color:{r:0,g:0,b:0}},'color',vars.get(name));}
function remember(n){created.push(n.id);return n;}
// The source SVG is original Carl-Cam artwork, not a remote UI-kit icon.
const check=remember(figma.createComponent());check.name='Icon/check';
check.description='Original 14px checkbox tick; source lives in design/checkmark.svg.';
check.resize(14,14);check.fills=[];check.x=3900;check.y=1190;
const svg=remember(figma.createNodeFromSvg('<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M3 7L6 10L11 4" stroke="#F1ECE2" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'));
svg.name='Original checkbox tick';check.appendChild(svg);
for(const node of svg.findAll(()=>true)) {
  created.push(node.id);
  if(Array.isArray(node.strokes)&&node.strokes.length)
    node.strokes=node.strokes.map(p=>p.type==='SOLID'?figma.variables.setBoundVariableForPaint(p,'color',vars.get('color/cream')):p);
}
const variants=[];
for(const state of ['Default','Focused','Disabled']) for(const checked of ['False','True']) {
  const c=remember(figma.createComponent());c.name=`State=${state}, Checked=${checked}`;
  c.resize(400,44);c.layoutMode='HORIZONTAL';c.primaryAxisSizingMode='FIXED';
  c.counterAxisSizingMode='FIXED';c.counterAxisAlignItems='CENTER';
  c.setBoundVariable('itemSpacing',vars.get('spacing/sm'));c.fills=[];
  if(state==='Focused'){c.strokes=[paint('color/yellow')];c.setBoundVariable('strokeWeight',vars.get('stroke/focus'));}
  else c.strokes=[];
  const box=remember(figma.createAutoLayout('HORIZONTAL'));box.name='Native checkbox indicator';
  box.resize(18,18);box.primaryAxisSizingMode='FIXED';box.counterAxisSizingMode='FIXED';
  box.primaryAxisAlignItems='CENTER';box.counterAxisAlignItems='CENTER';
  box.fills=[paint('color/surface')];
  box.strokes=[paint(state==='Disabled'?'color/control/disabled-border':'color/cyan')];
  box.setBoundVariable('strokeWeight',vars.get('stroke/control'));
  box.setBoundVariable('cornerRadius',vars.get('radius/checkbox'));c.appendChild(box);
  const tick=remember(check.createInstance());box.appendChild(tick);tick.visible=checked==='True';
  created.push(...tick.findAll(()=>true).map(n=>n.id));
  const label=remember(figma.createText());label.name='Label';label.fontName={family:'Arimo',style:'Regular'};
  await label.setTextStyleIdAsync('S:a75a4ef41cf4e33bcc82d3c6483027f9d7c70424,');
  label.characters='I understand local live camera processing';
  label.textAutoResize='HEIGHT';label.resize(374,18);
  label.fills=[paint(state==='Disabled'?'color/control/disabled-text':'color/cream')];
  c.appendChild(label);label.layoutSizingHorizontal='FILL';variants.push(c);
}
const set=remember(figma.combineAsVariants(variants,page));set.name='Desktop/Checkbox';
set.description='Native consent/option checkbox; 44px row, shared text, editable label and checked/focus/disabled states. Behavior belongs to Qt.';
const labelProperty=set.addComponentProperty('Label','TEXT','I understand local live camera processing');
variants.forEach((c,i)=>{c.findAllWithCriteria({types:['TEXT']})[0].componentPropertyReferences={characters:labelProperty};
  c.x=16+(i%2)*416;c.y=16+Math.floor(i/2)*60;});
set.resize(848,196);set.x=3900;set.y=950;
const review=remember(figma.createAutoLayout('VERTICAL'));review.name='Desktop/Checkbox States Review';
review.resize(848,100);review.layoutSizingHorizontal='FIXED';review.x=3900;review.y=1260;
review.fills=[paint('color/chrome')];
for(const k of ['paddingLeft','paddingRight','paddingTop','paddingBottom'])review.setBoundVariable(k,vars.get('spacing/md'));
review.setBoundVariable('itemSpacing',vars.get('spacing/sm'));
for(let i=0;i<3;i++) {
  const row=remember(figma.createAutoLayout('HORIZONTAL'));row.name=['Default','Focused','Disabled'][i];
  row.fills=[];row.setBoundVariable('itemSpacing',vars.get('spacing/md'));review.appendChild(row);
  for(const c of variants.slice(i*2,i*2+2)) {const instance=remember(c.createInstance());row.appendChild(instance);
    instance.setProperties({[labelProperty]:c.name});created.push(...instance.findAll(()=>true).map(n=>n.id));}
}
return {createdNodeIds:created,createdVariableIds:createdVariables,componentSetId:set.id,
  labelProperty,checkIconId:check.id,reviewId:review.id,
  variants:variants.map(c=>({id:c.id,name:c.name,width:c.width,height:c.height})),
  token:{id:vars.get('radius/checkbox').id,scopes:vars.get('radius/checkbox').scopes,codeSyntax:vars.get('radius/checkbox').codeSyntax}};
