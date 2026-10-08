// Editable desktop training views. The camera is off; no model or test result
// is fabricated. The collection guide receives the exact source crop separately.
const page=await figma.getNodeByIdAsync('0:1');await figma.setCurrentPageAsync(page);
await Promise.all(['Regular','Bold'].map(style=>figma.loadFontAsync({family:'Arimo',style})));
figma.skipInvisibleInstanceChildren=false;
const vars=new Map((await figma.variables.getLocalVariablesAsync()).map(v=>[v.name,v]));
const [button,title,selection,checkbox,preview,progress]=await Promise.all(
  ['23:50','29:55','29:52','33:121','29:83','42:259'].map(id=>figma.getNodeByIdAsync(id)));
const created=[],mutated=[],screens=[];
function paint(name){return figma.variables.setBoundVariableForPaint(
  {type:'SOLID',color:{r:0,g:0,b:0}},'color',vars.get(name));}
function remember(n){created.push(n.id);return n;}
function instance(component,parent,name,properties={}){
  const n=remember(component.createInstance());parent.appendChild(n);n.name=name;
  n.setProperties(properties);created.push(...n.findAll(()=>true).map(c=>c.id));return n;
}
function container(parent,name,direction='VERTICAL',width=null){
  const n=remember(figma.createAutoLayout(direction));n.name=name;n.fills=[];
  parent.appendChild(n);n.setBoundVariable('itemSpacing',vars.get('spacing/sm'));
  if(width===null)n.layoutSizingHorizontal='FILL';
  else{n.resize(width,26);n.layoutSizingVertical='HUG';}return n;
}
async function text(label,parent,name='Instructions',width=null){
  const n=remember(figma.createText());parent.appendChild(n);n.name=name;
  n.fontName={family:'Arimo',style:'Regular'};
  await n.setTextStyleIdAsync('S:a75a4ef41cf4e33bcc82d3c6483027f9d7c70424,');
  n.fills=[paint('color/cream')];n.resize(width??parent.width,18);
  n.textAutoResize='HEIGHT';n.characters=label;return n;
}
function action(label,parent){
  const n=instance(button,parent,label,{'Label#23:0':label});
  n.layoutSizingHorizontal='FILL';return n;
}
for(const [id,collect] of [['41:259',false],['41:260',true]]){
  const wrapper=await figma.getNodeByIdAsync(id);
  if(wrapper.children.length){screens.push({id,status:'already_composed'});continue;}
  mutated.push(wrapper.id);
  const header=instance(title,wrapper,'Native window controls',
    {'Title#29:1':'Carl-Cam — Train my gestures'});header.layoutSizingHorizontal='FILL';
  const body=container(wrapper,'Training workspace','HORIZONTAL');
  body.setBoundVariable('itemSpacing',vars.get('spacing/md'));body.layoutSizingVertical='FILL';
  const left=container(body,'Instruction viewport — vertical scrolling', 'VERTICAL',240);
  left.layoutSizingVertical='FILL';left.clipsContent=true;left.overflowDirection='VERTICAL';
  left.strokes=[paint('color/ink')];left.strokeWeight=1;left.strokeAlign='INSIDE';
  const instructions=container(left,'Scrollable instruction content','VERTICAL');
  for(const p of ['paddingTop','paddingBottom','paddingLeft','paddingRight'])
    instructions.setBoundVariable(p,vars.get('spacing/sm'));
  let guide=null;
  await text(collect?'Picture frame · round 1':'Choose which pose selects each filter',instructions,'Current training step',222);
  if(collect){
    const slot=container(instructions,'Owner-drawn picture-frame pose','HORIZONTAL');
    slot.resize(222,177);slot.layoutSizingVertical='FIXED';slot.primaryAxisAlignItems='CENTER';
    guide=remember(figma.createRectangle());slot.appendChild(guide);
    guide.name='Approved cropped pose guide — discrete content image';guide.resize(206,177);
    guide.fills=[];
    await text('Keep the required hand(s) clearly visible. Follow three small variations. Progress counts usable measurements, not waiting time. For the frame, both L-hands and a valid four-corner shape are required.',instructions,'Collection instructions',222);
  }else{
    await text('Frame and peace have fixed roles. Assign the other three poses to different filters.',instructions,'Pose roles',222);
    const choice=instance(selection,instructions,'Start or resume collection',{'Value#29:0':'Start a new collection'});
    choice.layoutSizingHorizontal='FILL';
    for(const [pose,theme] of [['Thumbs up','Red'],['Open palm','Yellow'],['I love you','Blue']]){
      await text(pose,instructions,'Assignable pose',222);
      const select=instance(selection,instructions,pose+' filter',{'Value#29:0':theme});
      select.layoutSizingHorizontal='FILL';
    }
    const consent=instance(checkbox,instructions,'Explicit numerical-data consent',
      {'Label#33:0':'Save hand measurements'});consent.layoutSizingHorizontal='FILL';
    await text('Only anonymous numerical hand measurements are saved on this computer. No photos or video are saved. You can delete measurements and models separately in Settings.',instructions,'Recording privacy',222);
  }
  const viewport=container(body,'Camera-off live preview','HORIZONTAL');
  viewport.layoutSizingVertical='FILL';viewport.primaryAxisAlignItems='CENTER';viewport.counterAxisAlignItems='CENTER';
  const live=instance(preview,viewport,'No camera image retained');
  const right=container(body,'Progress and quality viewport — vertical scrolling','VERTICAL',240);
  right.layoutSizingVertical='FILL';right.clipsContent=true;right.overflowDirection='VERTICAL';
  right.strokes=[paint('color/ink')];right.strokeWeight=1;right.strokeAlign='INSIDE';
  const quality=container(right,'Scrollable progress and recovery content');
  for(const p of ['paddingTop','paddingBottom','paddingLeft','paddingRight'])
    quality.setBoundVariable(p,vars.get('spacing/sm'));
  const bar=instance(progress,quality,'Usable-step progress',{'Value#42:0':'0%'});bar.layoutSizingHorizontal='FILL';
  await text(collect?'Recording 1 of 30 · 12 usable seconds. The app is learning your prompted label; an untrained classifier does not certify it.':'The camera is displayed live only. The full collection requires at least 6 minutes 15 seconds of usable measurements, plus instructions and tests.',quality,'Progress detail',222);
  action(collect?'Start round':'Begin guided collection',quality);
  const footer=container(wrapper,'Persistent safety notice and camera return','HORIZONTAL');
  footer.setBoundVariable('itemSpacing',vars.get('spacing/md'));footer.counterAxisAlignItems='CENTER';
  await text('Live commands stay disabled throughout training. A previous working model is kept unless you explicitly activate a tested replacement.',footer,'Training safety notice',892);
  const back=instance(button,footer,'Back to camera',{'Label#23:0':'Back to camera'});back.resize(240,46);
  const scale=Math.min(viewport.width/1280,viewport.height/720);live.resize(1280*scale,720*scale);
  wrapper.placeholder=false;
  const descendants=wrapper.findAll(()=>true);const counts={};for(const n of descendants)counts[n.type]=(counts[n.type]||0)+1;
  screens.push({id:wrapper.id,name:wrapper.name,guideId:guide?.id??null,descendantCount:descendants.length,counts,
    fontsMatch:descendants.filter(n=>n.type==='TEXT').every(n=>n.fontName.family==='Arimo'),
    imageFilledNodes:descendants.filter(n=>'fills' in n&&Array.isArray(n.fills)&&n.fills.some(p=>p.type==='IMAGE')).map(n=>({id:n.id,name:n.name,type:n.type,width:n.width,height:n.height})),
    cameraScene:{id:live.id,width:live.width,height:live.height},
    instructionsHeight:instructions.height,instructionViewportHeight:left.height});
}
return {createdNodeIds:created,mutatedNodeIds:mutated,screens};
