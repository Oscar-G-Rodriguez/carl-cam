// Compose the approved native camera screen from editable local components.
// The black viewport is the actual camera-off state, not a UI screenshot.
const page=await figma.getNodeByIdAsync('0:1');
await figma.setCurrentPageAsync(page);
await Promise.all(['Regular','Bold'].map(style=>figma.loadFontAsync({family:'Arimo',style})));
const wrapper=await figma.getNodeByIdAsync('26:48');
if(wrapper.children.length) return {status:'already_composed',wrapperId:wrapper.id};
const [allVars,collections,button]=await Promise.all([
  figma.variables.getLocalVariablesAsync(),figma.variables.getLocalVariableCollectionsAsync(),
  figma.getNodeByIdAsync('23:50')
]);
const vars=new Map(allVars.map(v=>[v.name,v]));
const collection=collections.find(c=>c.id==='VariableCollectionId:6:2');
const created=[],mutated=[wrapper.id],createdVariables=[];
function paint(name){return figma.variables.setBoundVariableForPaint(
  {type:'SOLID',color:{r:0,g:0,b:0}},'color',vars.get(name));}
function remember(n){created.push(n.id);return n;}
function descendants(n){created.push(...n.findAll(()=>true).map(c=>c.id));}
function row(name,parent){const n=remember(figma.createAutoLayout('HORIZONTAL'));
  n.name=name;n.fills=[];n.counterAxisAlignItems='CENTER';
  n.setBoundVariable('itemSpacing',vars.get('spacing/sm'));parent.appendChild(n);
  n.layoutSizingHorizontal='FILL';return n;}
async function text(label,parent,style='Body'){
  const n=remember(figma.createText());n.name=label;n.fontName={family:'Arimo',style:style==='Body'?'Regular':'Bold'};
  await n.setTextStyleIdAsync(style==='Body'?'S:a75a4ef41cf4e33bcc82d3c6483027f9d7c70424,':
    'S:d98bc3b0d1e5f7eba78be50ad563d5e1b4b8bd4e,');
  n.characters=label;n.fills=[paint('color/cream')];parent.appendChild(n);return n;
}
function action(label,parent,icon=null,disabled=false){
  const main=disabled?disabledButton:button;const n=remember(main.createInstance());
  parent.appendChild(n);n.name=label;
  n.setProperties({'Label#23:0':label,'Show icon#27:5':!!icon,...(icon?{'Icon#27:0':icon}:{})});
  descendants(n);return n;
}
const disabledButton=await figma.getNodeByIdAsync('23:56');
let selection=page.findAllWithCriteria({types:['COMPONENT']}).find(n=>n.name==='Desktop/Selection');
if(!selection){
  selection=remember(figma.createComponent());selection.name='Desktop/Selection';
  selection.description='Native camera/theme selector; editable value and original dropdown glyph. Application actions remain in Qt.';
  selection.resize(216,46);selection.layoutMode='HORIZONTAL';selection.counterAxisAlignItems='CENTER';
  selection.primaryAxisSizingMode='FIXED';selection.counterAxisSizingMode='FIXED';
  selection.fills=[paint('color/surface')];selection.strokes=[paint('color/cyan')];
  selection.setBoundVariable('cornerRadius',vars.get('radius/control'));
  selection.setBoundVariable('strokeWeight',vars.get('stroke/control'));
  selection.setBoundVariable('itemSpacing',vars.get('spacing/sm'));
  for(const field of ['paddingLeft','paddingRight'])selection.setBoundVariable(field,vars.get('spacing/sm'));
  const label=await text('Select camera',selection);label.textAutoResize='HEIGHT';label.resize(160,18);
  label.layoutSizingHorizontal='FILL';
  await text('▾',selection);
  const key=selection.addComponentProperty('Value','TEXT','Select camera');
  label.componentPropertyReferences={characters:key};selection.x=3900;selection.y=450;
}
const selectionKey=Object.keys(selection.componentPropertyDefinitions).find(k=>k.startsWith('Value#'));
function select(label,parent,width){const n=remember(selection.createInstance());parent.appendChild(n);
  n.setProperties({[selectionKey]:label});n.resize(width,46);n.name=label;descendants(n);return n;}
let titleBar=page.findAllWithCriteria({types:['COMPONENT']}).find(n=>n.name==='Desktop/TitleBar');
if(!titleBar){
  titleBar=remember(figma.createComponent());titleBar.name='Desktop/TitleBar';
  titleBar.description='Native draggable title bar. Original Figma minimize/maximize/close icons; all controls also keyboard accessible in Python.';
  titleBar.resize(880,56);titleBar.layoutMode='HORIZONTAL';
  titleBar.primaryAxisSizingMode='FIXED';titleBar.counterAxisSizingMode='FIXED';
  titleBar.counterAxisAlignItems='CENTER';titleBar.fills=[paint('color/chrome')];
  titleBar.setBoundVariable('itemSpacing',vars.get('spacing/sm'));
  for(const field of ['paddingLeft','paddingRight'])titleBar.setBoundVariable(field,vars.get('spacing/sm'));
  const label=await text('carl-cam',titleBar,'Control');
  const key=titleBar.addComponentProperty('Title','TEXT','carl-cam');label.componentPropertyReferences={characters:key};
  const spacer=remember(figma.createAutoLayout('HORIZONTAL'));spacer.name='Flexible space';spacer.fills=[];
  titleBar.appendChild(spacer);spacer.layoutSizingHorizontal='FILL';
  for(const [name,icon] of [['Minimize','9:45'],['Maximize / restore','9:48'],['Close','9:42']]){
    const n=action('',titleBar,icon);n.name=name;
    for(const t of n.findAllWithCriteria({types:['TEXT']})){t.visible=false;mutated.push(t.id);}
    n.resize(44,44);n.layoutSizingHorizontal='FIXED';n.layoutSizingVertical='FIXED';
  }
  titleBar.x=3900;titleBar.y=560;
}
if(!vars.has('color/video-black')){
  const raw=figma.variables.createVariable('primitive/video-black',collection,'COLOR');
  raw.scopes=[];raw.setValueForMode(collection.defaultModeId,{r:0,g:0,b:0,a:1});
  raw.setVariableCodeSyntax('WEB','Qt.GlobalColor.black');createdVariables.push(raw.id);
  const semantic=figma.variables.createVariable('color/video-black',collection,'COLOR');
  semantic.scopes=['FRAME_FILL','SHAPE_FILL'];
  semantic.setValueForMode(collection.defaultModeId,{type:'VARIABLE_ALIAS',id:raw.id});
  semantic.setVariableCodeSyntax('WEB','Qt.GlobalColor.black');createdVariables.push(semantic.id);
  vars.set('color/video-black',semantic);
}
let preview=page.findAllWithCriteria({types:['COMPONENT']}).find(n=>n.name==='Desktop/CameraOffPreview');
if(!preview){
  preview=remember(figma.createComponent());preview.name='Desktop/CameraOffPreview';
  preview.description='Actual camera-off viewport: black, 16:9, no retained image and no simulated person. Live frames are rendered by Python, never provided by this component.';
  preview.resize(640,360);preview.fills=[paint('color/video-black')];preview.x=3900;preview.y=760;
}
const header=remember(titleBar.createInstance());wrapper.appendChild(header);
header.layoutSizingHorizontal='FILL';descendants(header);
const navigation=row('Desktop navigation — outside virtual output',wrapper);
for(const [label,icon] of [['Camera setup',null],['Train my gestures','9:57'],['Inspect','9:60'],
  ['Settings',null],['Start virtual camera',null],['Help',null]])action(label,navigation,icon);
const capture=row('Camera controls',wrapper);
select('Select camera',capture,240);action('Start camera',capture,'9:54');action('Stop camera',capture,null,true);
select('Red',capture,100);action('Refresh cameras',capture);
const framing=row('Frame and menu controls',wrapper);
for(const label of ['Open menu','Mouse frame','Clear frame'])action(label,framing);
action('Try higher input quality',framing,null,true);
const viewport=row('16:9 camera viewport — camera off',wrapper);
viewport.resize(1248,474);viewport.layoutSizingHorizontal='FILL';viewport.layoutSizingVertical='FIXED';
viewport.primaryAxisAlignItems='CENTER';
const camera=remember(preview.createInstance());viewport.appendChild(camera);
camera.resize(842.6666667,474);camera.name='Camera is off — no capture';descendants(camera);
const status=await text('Choose a camera, then press Start camera.',wrapper);
status.textAutoResize='HEIGHT';status.resize(1248,18);status.layoutSizingHorizontal='FILL';
const outputStatus=await text('Virtual camera is off. Output is optional.',wrapper);
outputStatus.textAutoResize='HEIGHT';outputStatus.resize(1248,18);outputStatus.layoutSizingHorizontal='FILL';
return {createdNodeIds:created,mutatedNodeIds:mutated,createdVariableIds:createdVariables,
  wrapperId:wrapper.id,components:{selection:selection.id,titleBar:titleBar.id,preview:preview.id},
  selectionKey,width:wrapper.width,height:wrapper.height,
  sections:wrapper.children.map(n=>({id:n.id,name:n.name,width:n.width,height:n.height})),
  previewAspect:camera.width/camera.height};
