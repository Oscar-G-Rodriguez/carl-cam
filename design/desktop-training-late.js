/**
 * bindVariablesToComponent
 *
 * Binds design token variables to the visual properties of a component node.
 * Supports fills, strokes, all padding directions, item spacing, and corner radius.
 * Only binds properties for which a variable ID is provided in `bindings`.
 *
 * This function should be called on each variant individually within a component
 * set, OR on the component set itself for properties shared by all variants.
 *
 * @param {ComponentNode | FrameNode | RectangleNode} component
 *   The Figma node to mutate. Usually a ComponentNode or one of its children.
 * @param {{
 *   fills?: string,
 *   strokes?: string,
 *   paddingTop?: string,
 *   paddingBottom?: string,
 *   paddingLeft?: string,
 *   paddingRight?: string,
 *   itemSpacing?: string,
 *   cornerRadius?: string
 * }} bindings
 *   Each key is a visual property name; each value is a Figma Variable ID
 *   (e.g. "VariableID:123:456"). Omit a key to skip binding that property.
 * @returns {Promise<{ mutatedNodeIds: string[] }>}
 *   List of node IDs that were mutated (for audit/validation purposes).
 */
async function bindVariablesToComponent(component, bindings) {
  const mutatedNodeIds = []

  if (!component) {
    return { mutatedNodeIds }
  }

  // Batch every getVariableByIdAsync call upfront in a single Promise.all rather
  // than awaiting per-property — the lookups are independent and IPC-bound.
  const floatBindings = [
    ['paddingTop', 'paddingTop'],
    ['paddingBottom', 'paddingBottom'],
    ['paddingLeft', 'paddingLeft'],
    ['paddingRight', 'paddingRight'],
    ['itemSpacing', 'itemSpacing'],
    ['cornerRadius', 'cornerRadius'],
  ]

  const requestedIds = []
  if (bindings.fills) requestedIds.push(['fills', bindings.fills])
  if (bindings.strokes) requestedIds.push(['strokes', bindings.strokes])
  for (const [bindingKey] of floatBindings) {
    if (bindings[bindingKey]) requestedIds.push([bindingKey, bindings[bindingKey]])
  }

  const resolved = await Promise.all(
    requestedIds.map(([, id]) => figma.variables.getVariableByIdAsync(id)),
  )
  const varByKey = {}
  for (let i = 0; i < requestedIds.length; i++) {
    varByKey[requestedIds[i][0]] = resolved[i]
  }

  const markMutated = () => {
    if (!mutatedNodeIds.includes(component.id)) {
      mutatedNodeIds.push(component.id)
    }
  }

  // --- Fills ---
  const fillVar = varByKey.fills
  if (fillVar) {
    const existingFills = component.fills
    if (Array.isArray(existingFills) && existingFills.length > 0) {
      // Bind the color of the first fill to the variable
      const boundFill = figma.variables.setBoundVariableForPaint(existingFills[0], 'color', fillVar)
      component.fills = [boundFill, ...existingFills.slice(1)]
    } else {
      // No existing fill — create a solid fill bound to the variable
      const boundFill = figma.variables.setBoundVariableForPaint(
        { type: 'SOLID', color: { r: 0.5, g: 0.5, b: 0.5 } },
        'color',
        fillVar,
      )
      component.fills = [boundFill]
    }
    markMutated()
  }

  // --- Strokes ---
  const strokeVar = varByKey.strokes
  if (strokeVar) {
    const existingStrokes = component.strokes
    if (Array.isArray(existingStrokes) && existingStrokes.length > 0) {
      const boundStroke = figma.variables.setBoundVariableForPaint(
        existingStrokes[0],
        'color',
        strokeVar,
      )
      component.strokes = [boundStroke, ...existingStrokes.slice(1)]
    } else {
      const boundStroke = figma.variables.setBoundVariableForPaint(
        { type: 'SOLID', color: { r: 0.5, g: 0.5, b: 0.5 } },
        'color',
        strokeVar,
      )
      component.strokes = [boundStroke]
    }
    markMutated()
  }

  // --- Spacing properties (FLOAT variables bound via setBoundVariable) ---
  for (const [bindingKey, figmaProp] of floatBindings) {
    const variable = varByKey[bindingKey]
    if (variable) {
      component.setBoundVariable(figmaProp, variable)
      markMutated()
    }
  }

  return { mutatedNodeIds }
}



const page=await figma.getNodeByIdAsync('0:1');await figma.setCurrentPageAsync(page);
figma.skipInvisibleInstanceChildren=false;
const variables=new Map((await figma.variables.getLocalVariablesAsync()).map(v=>[v.name,v]));
const [title,button,disabledButton,checked,disabledCheckbox,preview,progress,busy,originalGuide]=await Promise.all(
 ['29:55','23:50','23:56','33:127','33:145','29:83','42:259','42:260','43:352'].map(id=>figma.getNodeByIdAsync(id)));
const fonts=new Map();for(const root of [title,button,disabledButton,checked,disabledCheckbox,progress,busy]){
 for(const t of root.findAllWithCriteria({types:['TEXT']}))for(const s of t.getStyledTextSegments(['fontName']))fonts.set(JSON.stringify(s.fontName),s.fontName);}
await Promise.all([...fonts.values()].map(f=>figma.loadFontAsync(f)));
await figma.loadFontAsync({family:'Arimo',style:'Regular'});
const created=[],mutated=[],screens=[];const data=[{"mode":"review","requested_size":[1180,780],"actual_size":[1180,780],"camera_scene":[0,136.125,636,357.75],"presentation_fixture":true,"heading":"Review recordings, then train","instructions":"Three whole recording rounds train the model. The fourth checks each candidate; the fifth stays untouched until a winner is fixed. PyTorch is primary; logistic regression is a comparison only. Depth needs a retry. Keep hands visible and try another angle/lighting; if unavailable depth persists, explicitly uncheck depth for four 112/138-value candidates.","detail":"Picture frame: 0 measurements\nPeace sign — open menu: 0 measurements\nThumbs up: 0 measurements\nOpen palm: 0 measurements\nI love you: 0 measurements\nOrdinary movement: 0 measurements","back_bounds":[924,718,240,46],"left_scroll_bounds":[16,80,240,630],"right_scroll_bounds":[924,80,240,630],"controls":{"action":{"text":"Train and compare","visible":true,"enabled":true,"bounds":[933,259,222,46]},"depth":{"text":"Include depth (180 values)","visible":true,"enabled":true,"bounds":[25,297,222,44]},"cuda":{"text":"Train on NVIDIA GPU","visible":true,"enabled":false,"bounds":[25,349,222,44]},"retry_depth":{"text":"Retry depth rounds","visible":true,"enabled":true,"bounds":[933,313,222,46]},"redo":{"text":"Redo this round","visible":false,"enabled":true,"bounds":[925,81,640,480]},"skip":{"text":"Skip lighting change","visible":false,"enabled":true,"bounds":[925,81,640,480]}},"progress_range":[0,100],"render":"training-review-1180x780.png","render_sha256":"55fa218f9dfb8712696ebf9cb1f70a284c1b4e3836473165977eb181f93458be"},{"mode":"fitting","requested_size":[1180,780],"actual_size":[1180,780],"camera_scene":[0,136.125,636,357.75],"presentation_fixture":true,"heading":"Training local neural candidates","instructions":"This stays on your computer. You can cancel and keep your recordings and previous model.","detail":"Preparing local candidate training. Progress will appear here; Back to camera requests cancellation.","back_bounds":[924,718,240,46],"left_scroll_bounds":[16,80,240,630],"right_scroll_bounds":[924,80,240,630],"controls":{"action":{"text":"Training in progress…","visible":true,"enabled":false,"bounds":[933,179,222,46]},"depth":{"text":"Include depth (180 values)","visible":false,"enabled":true,"bounds":[17,81,640,480]},"cuda":{"text":"Train on NVIDIA GPU","visible":false,"enabled":false,"bounds":[17,81,640,480]},"retry_depth":{"text":"Retry depth rounds","visible":false,"enabled":true,"bounds":[925,81,640,480]},"redo":{"text":"Redo this round","visible":false,"enabled":true,"bounds":[925,81,640,480]},"skip":{"text":"Skip lighting change","visible":false,"enabled":true,"bounds":[925,81,640,480]}},"progress_range":[0,0],"render":"training-fitting-1180x780.png","render_sha256":"fc783d5254529bd5a22751f17ccc64808b291315687eba361bf05f31ae736dea"},{"mode":"comparison","requested_size":[1180,780],"actual_size":[1180,780],"camera_scene":[0,136.125,636,357.75],"presentation_fixture":true,"heading":"Candidate comparison: Picture frame · attempt 1/5","instructions":"Release the previous pose. Prepare this pose, then mark when it is formed. The same observations are checked for every candidate; no filter/menu action runs.","detail":"Timing uses your explicit Pose ready marker. It includes any extra hand-return time after a keyboard mark.","back_bounds":[924,718,240,46],"left_scroll_bounds":[16,80,240,630],"right_scroll_bounds":[924,80,240,630],"controls":{"action":{"text":"Prepare this attempt","visible":true,"enabled":true,"bounds":[933,195,222,46]},"depth":{"text":"Include depth (180 values)","visible":false,"enabled":true,"bounds":[17,81,640,480]},"cuda":{"text":"Train on NVIDIA GPU","visible":false,"enabled":false,"bounds":[17,81,640,480]},"retry_depth":{"text":"Retry depth rounds","visible":false,"enabled":true,"bounds":[925,81,640,480]},"redo":{"text":"Redo this round","visible":false,"enabled":true,"bounds":[925,81,640,480]},"skip":{"text":"Skip lighting change","visible":false,"enabled":true,"bounds":[925,81,640,480]}},"progress_range":[0,100],"render":"training-comparison-1180x780.png","render_sha256":"a2996b320140eb16cfd9f3608ca9404d1f8420c9cfa4133a34fb7db398260c2c"},{"mode":"movement","requested_size":[1180,780],"actual_size":[1180,780],"camera_scene":[0,136.125,636,357.75],"presentation_fixture":true,"heading":"Candidate comparison — ordinary movement","instructions":"Spend two usable minutes typing, touching your face, pointing/pinching, and moving hands into and out of view. Commands remain disabled. Camera gaps pause the timer.","detail":"Timing uses your explicit Pose ready marker. It includes any extra hand-return time after a keyboard mark.","back_bounds":[924,718,240,46],"left_scroll_bounds":[16,80,240,630],"right_scroll_bounds":[924,80,240,630],"controls":{"action":{"text":"Start movement check","visible":true,"enabled":true,"bounds":[933,195,222,46]},"depth":{"text":"Include depth (180 values)","visible":false,"enabled":true,"bounds":[17,81,640,480]},"cuda":{"text":"Train on NVIDIA GPU","visible":false,"enabled":false,"bounds":[17,81,640,480]},"retry_depth":{"text":"Retry depth rounds","visible":false,"enabled":true,"bounds":[925,81,640,480]},"redo":{"text":"Redo this round","visible":false,"enabled":true,"bounds":[925,81,640,480]},"skip":{"text":"Skip lighting change","visible":false,"enabled":true,"bounds":[925,81,640,480]}},"progress_range":[0,100],"render":"training-movement-1180x780.png","render_sha256":"8fd4b14087f80db9e188382eb034c12f234481c157dc1736816df23ae22694bd"},{"mode":"sealed","requested_size":[1180,780],"actual_size":[1180,780],"camera_scene":[0,136.125,636,357.75],"presentation_fixture":true,"heading":"Fixed winner — final recording check","instructions":"Selected Fixture-only candidate — not trained. The fifth recordings are checked once, with this model and these thresholds fixed. A failure needs a new final check, not tuning against these recordings.","detail":"","back_bounds":[924,718,240,46],"left_scroll_bounds":[16,80,240,630],"right_scroll_bounds":[924,80,240,630],"controls":{"action":{"text":"Run untouched final check","visible":true,"enabled":true,"bounds":[933,147,222,46]},"depth":{"text":"Include depth (180 values)","visible":false,"enabled":true,"bounds":[17,81,640,480]},"cuda":{"text":"Train on NVIDIA GPU","visible":false,"enabled":false,"bounds":[17,81,640,480]},"retry_depth":{"text":"Retry depth rounds","visible":false,"enabled":true,"bounds":[925,81,640,480]},"redo":{"text":"Redo this round","visible":false,"enabled":true,"bounds":[925,81,640,480]},"skip":{"text":"Skip lighting change","visible":false,"enabled":true,"bounds":[925,81,640,480]}},"progress_range":[0,100],"render":"training-sealed-1180x780.png","render_sha256":"133a7637c1264c62e048567a2bee7964581a11932b78c9bffa35ca9b61824498"},{"mode":"needs_examples","requested_size":[1180,780],"actual_size":[1180,780],"camera_scene":[0,136.125,636,357.75],"presentation_fixture":true,"heading":"More examples are needed","instructions":"No candidate commands have been activated. Keep using your previous model or mouse/keyboard. The retry targets weak cases and keeps earlier training recordings; a revealed final test is replaced.","detail":"","back_bounds":[924,718,240,46],"left_scroll_bounds":[16,80,240,630],"right_scroll_bounds":[924,80,240,630],"controls":{"action":{"text":"Collect extra examples","visible":true,"enabled":true,"bounds":[933,147,222,46]},"depth":{"text":"Include depth (180 values)","visible":false,"enabled":true,"bounds":[17,81,640,480]},"cuda":{"text":"Train on NVIDIA GPU","visible":false,"enabled":false,"bounds":[17,81,640,480]},"retry_depth":{"text":"Retry depth rounds","visible":false,"enabled":true,"bounds":[925,81,640,480]},"redo":{"text":"Redo this round","visible":false,"enabled":true,"bounds":[925,81,640,480]},"skip":{"text":"Skip lighting change","visible":false,"enabled":true,"bounds":[925,81,640,480]}},"progress_range":[0,100],"render":"training-needs_examples-1180x780.png","render_sha256":"bf4738d74a3a4e27decc086c3032660186a84949d5ba65e3b90cfc38ef2ebed9"},{"mode":"ready","requested_size":[1180,780],"actual_size":[1180,780],"camera_scene":[0,136.125,636,357.75],"presentation_fixture":true,"heading":"Personal checks passed","instructions":"All five commands can be activated together. These checks concern your own recordings and rehearsal; they do not certify recognition for other people. After activation, Help offers guided practice of direct switching and menu selection on the live camera.","detail":"","back_bounds":[924,718,240,46],"left_scroll_bounds":[16,80,240,630],"right_scroll_bounds":[924,80,240,630],"controls":{"action":{"text":"Activate tested gestures","visible":true,"enabled":true,"bounds":[933,147,222,46]},"depth":{"text":"Include depth (180 values)","visible":false,"enabled":true,"bounds":[17,81,640,480]},"cuda":{"text":"Train on NVIDIA GPU","visible":false,"enabled":false,"bounds":[17,81,640,480]},"retry_depth":{"text":"Retry depth rounds","visible":false,"enabled":true,"bounds":[925,81,640,480]},"redo":{"text":"Redo this round","visible":false,"enabled":true,"bounds":[925,81,640,480]},"skip":{"text":"Skip lighting change","visible":false,"enabled":true,"bounds":[925,81,640,480]}},"progress_range":[0,100],"render":"training-ready-1180x780.png","render_sha256":"aec141cf8f9e0dfc59867676c1362ad12ce4b759de04e2f8f61314a4f1649f6a"}];
const wrappers=[{"mode":"review","id":"65:652","name":"Desktop/Training/Review recordings","x":22340,"y":100,"width":1180,"height":780},{"mode":"fitting","id":"65:653","name":"Desktop/Training/Fitting","x":23680,"y":100,"width":1180,"height":780},{"mode":"comparison","id":"65:654","name":"Desktop/Training/Compare candidates","x":25020,"y":100,"width":1180,"height":780},{"mode":"movement","id":"65:655","name":"Desktop/Training/Ordinary movement","x":26360,"y":100,"width":1180,"height":780},{"mode":"sealed","id":"65:656","name":"Desktop/Training/Sealed check","x":27700,"y":100,"width":1180,"height":780},{"mode":"needs_examples","id":"65:657","name":"Desktop/Training/Extra examples","x":29040,"y":100,"width":1180,"height":780},{"mode":"ready","id":"65:658","name":"Desktop/Training/Approve activation","x":30380,"y":100,"width":1180,"height":780}];
function keep(n){created.push(n.id);return n;}
function paint(name){const v=variables.get(name);let c=v.valuesByMode[Object.keys(v.valuesByMode)[0]];
 while(c.type==='VARIABLE_ALIAS'){const ref=[...variables.values()].find(x=>x.id===c.id);c=ref.valuesByMode[Object.keys(ref.valuesByMode)[0]];}
 return figma.variables.setBoundVariableForPaint({type:'SOLID',color:{r:c.r,g:c.g,b:c.b}},'color',v);}
function all(n){return [n,...('findAll' in n?n.findAll(()=>true):[])];}
function instance(c,parent,name,props={}){const n=c.createInstance();parent.appendChild(n);n.name=name;n.setProperties(props);created.push(...all(n).map(x=>x.id));return n;}
function column(parent,name,direction='VERTICAL',width=null){const n=keep(figma.createAutoLayout(direction));parent.appendChild(n);n.name=name;n.fills=[];n.setBoundVariable('itemSpacing',variables.get('spacing/sm'));if(width===null)n.layoutSizingHorizontal='FILL';else{n.resize(width,20);n.layoutSizingVertical='HUG';}return n;}
async function text(parent,name,value,width=222){const n=keep(figma.createText());parent.appendChild(n);n.name=name;n.fontName={family:'Arimo',style:'Regular'};await n.setTextStyleIdAsync('S:b900cacd089357111e011385d9dba3b35f8eabec,');n.fills=[paint('color/cream')];n.characters=value;n.resize(width,16);n.textAutoResize=value?'HEIGHT':'NONE';return n;}
function action(parent,name,enabled=true){const n=instance(enabled?button:disabledButton,parent,name,{'Label#23:0':name});n.layoutSizingHorizontal='FILL';return n;}
let copy=page.children.find(n=>n.name==='Desktop/TrainingCopy');let properties;
if(!copy){copy=keep(figma.createComponent());page.appendChild(copy);copy.name='Desktop/TrainingCopy';copy.description='Editable current-step heading, optional exact owner frame drawing and wrapping instructions for the in-app training sidebar. No training result or camera image is represented by the illustration.';
 copy.layoutMode='VERTICAL';copy.resize(222,40);copy.layoutSizingVertical='HUG';copy.fills=[];copy.x=3900;copy.y=3600;
 const binding=await bindVariablesToComponent(copy,{itemSpacing:'VariableID:6:17'});mutated.push(...binding.mutatedNodeIds);
 const h=await text(copy,'Current training step','Training step');
 const slot=column(copy,'Optional owner-drawn frame pose','HORIZONTAL');slot.resize(222,177);slot.layoutSizingVertical='FIXED';slot.primaryAxisAlignItems='CENTER';slot.counterAxisAlignItems='CENTER';
 const g=originalGuide.clone();slot.appendChild(g);created.push(...all(g).map(x=>x.id));g.name='Exact 206×177 owner frame crop';g.resize(206,177);
 const b=await text(copy,'Training instructions','Follow the current prompt. Commands stay disabled.');
 const heading=copy.addComponentProperty('Heading','TEXT','Training step');h.componentPropertyReferences={characters:heading};
 const instructions=copy.addComponentProperty('Instructions','TEXT','Follow the current prompt. Commands stay disabled.');b.componentPropertyReferences={characters:instructions};
 const show=copy.addComponentProperty('Show frame guide','BOOLEAN',false);slot.componentPropertyReferences={visible:show};slot.visible=false;
 properties={heading,instructions,show};
}else{const p=copy.componentPropertyDefinitions;properties={heading:Object.keys(p).find(k=>k.startsWith('Heading#')),instructions:Object.keys(p).find(k=>k.startsWith('Instructions#')),show:Object.keys(p).find(k=>k.startsWith('Show frame guide#'))};}
let review=page.children.find(n=>n.name==='Desktop/TrainingCopy Review');
if(!review){review=column(page,'Desktop/TrainingCopy Review','HORIZONTAL',460);review.x=3900;review.y=3950;review.fills=[paint('color/surface')];review.setBoundVariable('itemSpacing',variables.get('spacing/md'));
 for(const mode of ['review','comparison']){const s=data.find(s=>s.mode===mode);const n=instance(copy,review,'Instruction specimen — '+mode,{[properties.heading]:s.heading,[properties.instructions]:s.instructions,[properties.show]:mode==='comparison'});n.resize(222,n.height);n.layoutSizingVertical='HUG';}}
for(const wrapperInfo of wrappers){const s=data.find(s=>s.mode===wrapperInfo.mode);const root=await figma.getNodeByIdAsync(wrapperInfo.id);
 if(root.children.length){screens.push({id:root.id,mode:s.mode,status:'already_composed'});continue;}
 mutated.push(root.id);const header=instance(title,root,'Native window controls',{'Title#29:1':'Carl-Cam — Train my gestures'});header.layoutSizingHorizontal='FILL';
 const body=column(root,'Training workspace','HORIZONTAL');body.setBoundVariable('itemSpacing',variables.get('spacing/md'));body.layoutSizingVertical='FILL';
 const left=column(body,'Instruction viewport — vertical scrolling','VERTICAL',240);left.layoutSizingVertical='FILL';left.clipsContent=true;left.overflowDirection='VERTICAL';left.strokes=[paint('color/ink')];left.strokeWeight=1;left.strokeAlign='INSIDE';
 const instructionPanel=column(left,'Scrollable instruction content');for(const p of ['paddingTop','paddingBottom','paddingLeft','paddingRight'])instructionPanel.setBoundVariable(p,variables.get('spacing/sm'));
 const copyInstance=instance(copy,instructionPanel,'Current phase and instructions',{[properties.heading]:s.heading,[properties.instructions]:s.instructions,[properties.show]:s.mode==='comparison'});
 if(s.mode==='review'){const d=instance(checked,instructionPanel,'Preferred depth measurements',{'Label#33:0':s.controls.depth.text});d.layoutSizingHorizontal='FILL';const gpu=instance(disabledCheckbox,instructionPanel,'CUDA unavailable in this reviewed environment',{'Label#33:0':s.controls.cuda.text});gpu.layoutSizingHorizontal='FILL';}
 const viewport=column(body,'Camera-off live preview','HORIZONTAL');viewport.layoutSizingVertical='FILL';viewport.primaryAxisAlignItems='CENTER';viewport.counterAxisAlignItems='CENTER';
 const live=instance(preview,viewport,'No camera image retained');
 const right=column(body,'Progress and quality viewport — vertical scrolling','VERTICAL',240);right.layoutSizingVertical='FILL';right.clipsContent=true;right.overflowDirection='VERTICAL';right.strokes=[paint('color/ink')];right.strokeWeight=1;right.strokeAlign='INSIDE';
 const quality=column(right,'Scrollable progress and recovery content');for(const p of ['paddingTop','paddingBottom','paddingLeft','paddingRight'])quality.setBoundVariable(p,variables.get('spacing/sm'));
 const completeStep=s.mode==='ready'||s.mode==='movement';
 const bar=instance(s.mode==='fitting'?busy:progress,quality,'Current phase progress',{'Value#42:0':completeStep?'100%':'0%'});bar.layoutSizingHorizontal='FILL';if(completeStep){bar.fills=[paint('color/cyan')];mutated.push(bar.id);}
 await text(quality,'Phase detail — fixture only',s.detail);
 const primary=action(quality,s.controls.action.text,s.controls.action.enabled);
 if(s.mode==='review')action(quality,s.controls.retry_depth.text);
 const footer=column(root,'Persistent safety notice and camera return','HORIZONTAL');footer.setBoundVariable('itemSpacing',variables.get('spacing/md'));footer.counterAxisAlignItems='CENTER';
 await text(footer,'Training safety notice','Live commands stay disabled throughout training. A previous working model is kept unless you explicitly activate a tested replacement.',892);
 const back=action(footer,'Back to camera');back.resize(240,46);back.layoutSizingHorizontal='FIXED';
 const scale=Math.min(viewport.width/1280,viewport.height/720);live.resize(1280*scale,720*scale);root.placeholder=false;
 const nodes=root.findAll(()=>true),counts={};for(const n of nodes)counts[n.type]=(counts[n.type]||0)+1;
 screens.push({id:root.id,mode:s.mode,name:root.name,descendantCount:nodes.length,counts,fontsMatch:nodes.filter(n=>n.type==='TEXT').every(n=>n.fontName.family==='Arimo'),imageFilledNodes:nodes.filter(n=>'fills'in n&&Array.isArray(n.fills)&&n.fills.some(p=>p.type==='IMAGE')).map(n=>({id:n.id,name:n.name,type:n.type,width:n.width,height:n.height,visible:n.visible,parentVisible:n.parent.visible,hash:n.fills.find(p=>p.type==='IMAGE').imageHash})),cameraScene:{id:live.id,width:live.width,height:live.height},primaryAction:{id:primary.id,label:s.controls.action.text,enabled:s.controls.action.enabled},majorBounds:root.children.map(n=>({id:n.id,name:n.name,x:n.x,y:n.y,width:n.width,height:n.height}))});
}
return {createdNodeIds:[...new Set(created)],mutatedNodeIds:[...new Set(mutated)],copy:{id:copy.id,properties,reviewId:review.id},screens,presentationFixtures:true,sourceEvidence:'build/desktop-training-review-fd2a5757f03d4a148eb0f12574e31852/evidence.json'};
