// Compose the four approved setup steps from the shared desktop controls.
// All text remains editable; the camera is off and no demo model is invented.
const page=await figma.getNodeByIdAsync('0:1');await figma.setCurrentPageAsync(page);
await Promise.all(['Regular','Bold'].map(style=>figma.loadFontAsync({family:'Arimo',style})));
figma.skipInvisibleInstanceChildren=false;
const vars=new Map((await figma.variables.getLocalVariablesAsync()).map(v=>[v.name,v]));
const [button,disabledButton,title,selection,checkboxOff,checkboxOn,preview]=await Promise.all(
  ['23:50','23:56','29:55','29:52','33:121','33:127','29:83'].map(id=>figma.getNodeByIdAsync(id)));
const ids=['34:147','34:148','34:149','34:150'];const created=[],mutated=[],screens=[];
function paint(name){return figma.variables.setBoundVariableForPaint(
  {type:'SOLID',color:{r:0,g:0,b:0}},'color',vars.get(name));}
function remember(n){created.push(n.id);return n;}
function instance(component,parent,name,properties={}){
  const n=remember(component.createInstance());parent.appendChild(n);n.name=name;
  n.setProperties(properties);created.push(...n.findAll(()=>true).map(c=>c.id));return n;
}
function container(parent,name,direction='VERTICAL') {
  const n=remember(figma.createAutoLayout(direction));n.name=name;n.fills=[];
  n.setBoundVariable('itemSpacing',vars.get(direction==='VERTICAL'?'spacing/md':'spacing/sm'));
  parent.appendChild(n);n.layoutSizingHorizontal='FILL';return n;
}
async function text(label,parent,name='Instructions') {
  const n=remember(figma.createText());n.name=name;n.fontName={family:'Arimo',style:'Regular'};
  await n.setTextStyleIdAsync('S:a75a4ef41cf4e33bcc82d3c6483027f9d7c70424,');
  n.fills=[paint('color/cream')];parent.appendChild(n);
  // resize/FILL can reset the text's auto-height mode. Apply the explicit
  // wrapping width first, then restore HEIGHT before assigning the content.
  n.resize(parent.width,18);n.textAutoResize='HEIGHT';n.characters=label;return n;
}
function action(label,parent,disabled=false,fill=false){
  const n=instance(disabled?disabledButton:button,parent,label,{'Label#23:0':label});
  if(fill)n.layoutSizingHorizontal='FILL';return n;
}
const headings=['Step 1 of 4 — Your camera stays on this computer',
  'Step 2 of 4 — Choose and test camera','Step 3 of 4 — Choose gesture controls',
  'Step 4 of 4 — Optional output'];
const instructions=[
  'Carl-Cam processes live camera images locally. It does not upload or save photos or video. Local models are bundled; hand controls require a verified offline vision runtime. If you choose personal gesture training, the app asks separately before saving numerical hand measurements and labels. Settings can delete those samples and personal models. Virtual-camera output is optional and starts only when you explicitly request it in this launch.',
  'Choose a physical camera. Stop other camera applications if it is busy, then test it here. OBS Virtual Camera is excluded from inputs.',
  'You can try all three filters immediately with mouse and keyboard. Personal training introduces the five supported poses, records prompted variations, fits local neural candidates, checks them, and asks before activation. There is no bundled creator demo in this development build.',
  'Optional output lets another desktop app receive Carl-Cam through OBS Virtual Camera. Install OBS yourself if needed, then return to the Camera screen and press Start virtual camera. Select OBS Virtual Camera in the receiving app. Carl-Cam never installs OBS or starts output automatically. Skip this step if you only want the desktop preview.'
];
for(let i=0;i<ids.length;i++) {
  const wrapper=await figma.getNodeByIdAsync(ids[i]);
  if(wrapper.children.length){screens.push({id:wrapper.id,status:'already_composed'});continue;}
  mutated.push(wrapper.id);
  const header=instance(title,wrapper,'Native window controls',{'Title#29:1':'Carl-Cam — Setup'});
  header.layoutSizingHorizontal='FILL';
  await text(headings[i],wrapper,'Current setup step');
  const body=container(wrapper,'Step content');body.layoutSizingVertical='FILL';
  await text(instructions[i],body);
  let viewport=null,live=null;
  if(i===0) {
    const consent=instance(checkboxOff,body,'Explicit local-processing acknowledgment',
      {'Label#33:0':'I understand local live camera processing'});consent.layoutSizingHorizontal='FILL';
    await text('Telemetry-free vision runtime is not verified. Mouse and keyboard still work.',body,'Development runtime notice');
  } else if(i===1) {
    const controls=container(body,'Physical camera controls','HORIZONTAL');controls.counterAxisAlignItems='CENTER';
    const select=instance(selection,controls,'Physical camera choice',{'Value#29:0':'Select camera'});select.resize(240,46);
    action('Refresh cameras',controls);action('Test selected camera',controls);action('Stop camera to change device',controls);
    viewport=container(body,'Camera-off 16:9 preview','HORIZONTAL');viewport.layoutSizingVertical='FILL';
    viewport.primaryAxisAlignItems='CENTER';viewport.counterAxisAlignItems='CENTER';
    live=instance(preview,viewport,'Camera off — no image retained');
    await text('Camera has not been started',body,'Camera readiness');
  } else if(i===2) {
    action('Train my gestures',body,false,true);
    action('Use mouse/keyboard for now',body,false,true);
  } else {
    const practice=instance(checkboxOn,body,'Optional control practice',
      {'Label#33:0':'Practice both filter-switching paths after setup (optional)'});
    practice.layoutSizingHorizontal='FILL';
  }
  if(i!==1){const space=container(body,'Flexible remaining space');space.layoutSizingVertical='FILL';}
  const footer=container(wrapper,'Setup navigation','HORIZONTAL');
  action('Previous step',footer,i===0,true);
  action(i===3?'Finish setup':'Continue',footer,false,true);
  action('Back to camera',footer,false,true);
  if(viewport){const scale=Math.min(viewport.width/1280,viewport.height/720);live.resize(1280*scale,720*scale);}
  wrapper.placeholder=false;
  const descendants=wrapper.findAll(()=>true);const counts={};
  for(const n of descendants)counts[n.type]=(counts[n.type]||0)+1;
  screens.push({id:wrapper.id,name:wrapper.name,descendantCount:descendants.length,counts,
    imageFilledNodes:descendants.filter(n=>Array.isArray(n.fills)&&n.fills.some(p=>p.type==='IMAGE')).map(n=>({id:n.id,name:n.name,type:n.type,width:n.width,height:n.height})),
    fontsMatch:descendants.filter(n=>n.type==='TEXT').every(n=>n.fontName.family==='Arimo'),
    cameraScene:live?{id:live.id,width:live.width,height:live.height,ratio:live.width/live.height}:null});
}
return {createdNodeIds:created,mutatedNodeIds:mutated,screens};
