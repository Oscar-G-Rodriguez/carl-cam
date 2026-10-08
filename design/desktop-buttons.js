// Carl-Cam desktop family. Adapted from figma-generate-library's
// createComponentWithVariants.js; existing approved foundations are reused.
const page = await figma.getNodeByIdAsync('0:1');
await figma.setCurrentPageAsync(page);
const existing = page.findAllWithCriteria({types:['COMPONENT_SET']})
  .find(n => n.name === 'Desktop/Button');
if (existing) return {status:'already_exists', componentSetId:existing.id};
const [allVars, collections, fonts] = await Promise.all([
  figma.variables.getLocalVariablesAsync(),
  figma.variables.getLocalVariableCollectionsAsync(),
  figma.listAvailableFontsAsync()
]);
if (!fonts.some(f=>f.fontName.family==='Arimo' && f.fontName.style==='Regular'))
  throw new Error('Approved Arimo Regular is unavailable');
await figma.loadFontAsync({family:'Arimo',style:'Regular'});
const collection = collections.find(c=>c.id==='VariableCollectionId:6:2');
if (!collection) throw new Error('Approved token collection is missing');
const mode = collection.defaultModeId;
const vars = new Map(allVars.map(v=>[v.name,v]));
const createdVariables = [];
function token(name,type,value,scopes,syntax) {
  if (vars.has(name)) return vars.get(name);
  const v = figma.variables.createVariable(name,collection,type);
  v.scopes = scopes; v.setValueForMode(mode,value);
  v.setVariableCodeSyntax('WEB',syntax);
  vars.set(name,v); createdVariables.push(v.id); return v;
}
const disabledText = token('primitive/disabled-text','COLOR',
  {r:174/255,g:179/255,b:189/255,a:1},[],'CONTROL_DISABLED_TEXT');
const disabledBorder = token('primitive/disabled-border','COLOR',
  {r:108/255,g:115/255,b:129/255,a:1},[],'CONTROL_DISABLED_BORDER');
token('color/control/disabled-text','COLOR',{type:'VARIABLE_ALIAS',id:disabledText.id},
  ['TEXT_FILL'],'CONTROL_DISABLED_TEXT');
token('color/control/disabled-border','COLOR',{type:'VARIABLE_ALIAS',id:disabledBorder.id},
  ['STROKE_COLOR'],'CONTROL_DISABLED_BORDER');
token('spacing/xs','FLOAT',4,['GAP'],'CONTROL_PADDING_VERTICAL');
token('stroke/focus','FLOAT',3,['STROKE_FLOAT'],'CONTROL_FOCUS_STROKE');
function paint(name) {
  const v = vars.get(name); if (!v) throw new Error('Missing '+name);
  return figma.variables.setBoundVariableForPaint(
    {type:'SOLID',color:{r:0,g:0,b:0}},'color',v);
}
const created = [];
const variants = [];
for (const state of ['Default','Hover','Focused','Disabled']) {
  const c = figma.createComponent(); created.push(c.id); page.appendChild(c);
  c.name = 'State='+state; c.resize(184,46); c.layoutMode='HORIZONTAL';
  c.primaryAxisSizingMode='AUTO'; c.counterAxisSizingMode='FIXED';
  c.primaryAxisAlignItems='CENTER'; c.counterAxisAlignItems='CENTER';
  c.setBoundVariable('paddingLeft',vars.get('spacing/sm'));
  c.setBoundVariable('paddingRight',vars.get('spacing/sm'));
  c.setBoundVariable('paddingTop',vars.get('spacing/xs'));
  c.setBoundVariable('paddingBottom',vars.get('spacing/xs'));
  c.setBoundVariable('itemSpacing',vars.get('spacing/sm'));
  c.setBoundVariable('cornerRadius',vars.get('radius/control'));
  c.setBoundVariable('strokeWeight',vars.get(state==='Focused'?'stroke/focus':'stroke/control'));
  c.strokeAlign='INSIDE'; c.fills=[paint('color/surface')];
  c.strokes=[paint(state==='Disabled'?'color/control/disabled-border':
    state==='Focused'?'color/yellow':state==='Hover'?'color/pink':'color/cyan')];
  const label = figma.createText(); created.push(label.id);
  label.name='Label'; label.fontName={family:'Arimo',style:'Regular'};
  await label.setTextStyleIdAsync('S:a75a4ef41cf4e33bcc82d3c6483027f9d7c70424,');
  label.characters='Start camera'; label.textAutoResize='WIDTH_AND_HEIGHT';
  label.fills=[paint(state==='Disabled'?'color/control/disabled-text':'color/cream')];
  c.appendChild(label); variants.push(c);
}
const set = figma.combineAsVariants(variants,page); created.push(set.id);
set.name='Desktop/Button';
set.description='Accessible desktop action button. Label is editable; four states mirror Qt controls. Camera/menu/output behavior remains in Python.';
const labelProperty = set.addComponentProperty('Label','TEXT','Start camera');
variants.forEach((c,index)=>{
  const label=c.findAllWithCriteria({types:['TEXT']})[0];
  label.componentPropertyReferences={characters:labelProperty};
  c.x=24+index*208; c.y=24;
});
set.resize(880,94); set.x=3900; set.y=100;
const review=figma.createAutoLayout('VERTICAL'); created.push(review.id);
review.name='Desktop/Button States Review'; review.resize(880,100);
review.x=3900; review.y=260; review.layoutSizingHorizontal='FIXED';
review.fills=[paint('color/chrome')];
for(const field of ['paddingLeft','paddingRight','paddingTop','paddingBottom'])
  review.setBoundVariable(field,vars.get('spacing/lg'));
review.setBoundVariable('itemSpacing',vars.get('spacing/md'));
const row=figma.createAutoLayout('HORIZONTAL');created.push(row.id);
row.name='Normal / Hover / Keyboard focus / Disabled';row.fills=[];
row.setBoundVariable('itemSpacing',vars.get('spacing/md'));review.appendChild(row);
for(const variant of variants){
  const inst=variant.createInstance();row.appendChild(inst);
  inst.setProperties({[labelProperty]:variant.name.replace('State=','')});
  created.push(inst.id,...inst.findAll(()=>true).map(n=>n.id));
}
return {createdNodeIds:created,createdVariableIds:createdVariables,
  componentSetId:set.id,variants:variants.map(c=>({id:c.id,name:c.name,width:c.width,height:c.height})),
  labelProperty,reviewId:review.id,reviewBounds:{width:review.width,height:review.height},
  tokenScopes:createdVariables.map(id=>{const v=[...vars.values()].find(v=>v.id===id);return {id,name:v.name,scopes:v.scopes,codeSyntax:v.codeSyntax};})};
