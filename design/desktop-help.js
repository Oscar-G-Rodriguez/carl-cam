/**
 * createComponentWithVariants
 *
 * Creates a component set by generating all combinations of `variantAxes`,
 * building one Figma component per combination, then calling
 * `figma.combineAsVariants` to produce the component set. After combining,
 * the variants are repositioned into a grid so they don't all stack at (0, 0).
 *
 * @param {{
 *   name: string,
 *   description?: string,
 *   variantAxes: Record<string, string[]>,
 *   baseProps: {
 *     width: number,
 *     height: number,
 *     fills?: Paint[],
 *     padding?: {top?: number, bottom?: number, left?: number, right?: number},
 *     radius?: number,
 *     layoutMode?: 'HORIZONTAL' | 'VERTICAL' | 'NONE',
 *     itemSpacing?: number
 *   },
 *   page: PageNode
 * }} config
 *   - `name`: Component set name (e.g. "Button").
 *   - `description`: Optional human-readable purpose and usage guidance.
 *   - `variantAxes`: Each key is a variant property name; each value is an array of
 *     allowed values. All combinations are generated (Cartesian product).
 *     Example: { Size: ['Small', 'Medium', 'Large'], Style: ['Primary', 'Ghost'] }
 *     produces 6 variants.
 *   - `baseProps`: Visual properties applied to every variant.
 *   - `page`: The PageNode to create components on (must be set as current page by caller).
 * @returns {Promise<{
 *   componentSet: ComponentSetNode,
 *   variants: ComponentNode[]
 * }>}
 */
async function createComponentWithVariants(config) {
  const { name, variantAxes, baseProps, page } = config

  // Ensure we are on the correct page
  await figma.setCurrentPageAsync(page)

  // Compute Cartesian product of variant axes
  const axisNames = Object.keys(variantAxes)
  const axisValues = axisNames.map((k) => variantAxes[k])
  const combinations = cartesianProduct(axisValues)

  // Build one component per combination
  const components = []
  for (const combo of combinations) {
    const comp = figma.createComponent()

    // Name: "Property=Value, Property=Value, ..."
    comp.name = axisNames.map((ax, i) => `${ax}=${combo[i]}`).join(', ')

    // Base geometry
    comp.resize(baseProps.width, baseProps.height)

    // Fills
    if (baseProps.fills !== undefined) {
      comp.fills = baseProps.fills
    } else {
      comp.fills = [{ type: 'SOLID', color: { r: 0.9, g: 0.9, b: 0.9 } }]
    }

    // Corner radius
    if (baseProps.radius !== undefined) {
      comp.cornerRadius = baseProps.radius
    }

    // Auto-layout
    if (baseProps.layoutMode && baseProps.layoutMode !== 'NONE') {
      comp.layoutMode = baseProps.layoutMode
      comp.primaryAxisAlignItems = 'CENTER'
      comp.counterAxisAlignItems = 'CENTER'
      if (baseProps.itemSpacing !== undefined) {
        comp.itemSpacing = baseProps.itemSpacing
      }
    }

    // Padding
    if (baseProps.padding) {
      comp.paddingTop = baseProps.padding.top ?? 0
      comp.paddingBottom = baseProps.padding.bottom ?? 0
      comp.paddingLeft = baseProps.padding.left ?? 0
      comp.paddingRight = baseProps.padding.right ?? 0
    }

    page.appendChild(comp)
    components.push(comp)
  }

  // Combine into a component set
  const componentSet = figma.combineAsVariants(components, page)
  componentSet.name = name
  if (config.description) {
    componentSet.description = config.description
  }

  // Grid layout — variants stack at (0, 0) after combineAsVariants; reposition them.
  const GRID_GAP = 16
  const cols = Math.max(1, axisValues[axisValues.length - 1]?.length ?? 1)
  const variantWidth = baseProps.width
  const variantHeight = baseProps.height

  componentSet.children.forEach((variant, idx) => {
    const col = idx % cols
    const row = Math.floor(idx / cols)
    variant.x = col * (variantWidth + GRID_GAP)
    variant.y = row * (variantHeight + GRID_GAP)
  })

  // Resize component set to wrap its children with padding
  const totalCols = Math.min(cols, combinations.length)
  const totalRows = Math.ceil(combinations.length / cols)
  const PADDING = 40
  componentSet.resize(
    totalCols * variantWidth + (totalCols - 1) * GRID_GAP + PADDING * 2,
    totalRows * variantHeight + (totalRows - 1) * GRID_GAP + PADDING * 2,
  )

  // Position component set at a safe canvas location
  componentSet.x = 480
  componentSet.y = 80

  return { componentSet, variants: componentSet.children }
}

/**
 * Computes the Cartesian product of multiple arrays.
 * cartesianProduct([[A, B], [1, 2]]) → [[A,1], [A,2], [B,1], [B,2]]
 *
 * @param {Array<string[]>} arrays
 * @returns {string[][]}
 */
function cartesianProduct(arrays) {
  return arrays.reduce(
    (acc, curr) => acc.flatMap((combo) => curr.map((val) => [...combo, val])),
    [[]],
  )
}


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



const page=await figma.getNodeByIdAsync('0:1');
// The supplied variant helper below switches the page exactly once.
if(page.children.some(n=>n.name==='Desktop/PoseGuide')) return {status:'inspect_existing_pose_components'};
const {componentSet,variants}=await createComponentWithVariants({
 name:'Desktop/PoseGuide',
 description:'Owner-drawn pose illustration and editable caption. Five content variants use the exact Qt Help crops; illustrations are instructional assets, never classifier training samples.',
 variantAxes:{Pose:["frame","peace","thumbs_up","open_palm","ily"]},
 baseProps:{width:304,height:220,layoutMode:'VERTICAL',fills:[]},page});
figma.skipInvisibleInstanceChildren=false;
const [instructions,illustrations,title,button]=await Promise.all(
 ['59:531','59:532','29:55','23:50'].map(id=>figma.getNodeByIdAsync(id)));
if(instructions.children.length||illustrations.children.length)throw new Error('Inspect partial Help composition before retrying');
const fonts=await figma.listAvailableFontsAsync();
for(const style of ['Regular','Bold']) {
 if(!fonts.some(f=>f.fontName.family==='Arimo'&&f.fontName.style===style))throw new Error('Required Arimo font unavailable');
 await figma.loadFontAsync({family:'Arimo',style});
}
for(const component of [title,button]) for(const text of component.findAllWithCriteria({types:['TEXT']}))
 for(const segment of text.getStyledTextSegments(['fontName']))await figma.loadFontAsync(segment.fontName);
const vars=new Map((await figma.variables.getLocalVariablesAsync()).map(v=>[v.name,v]));
const byId=new Map([...vars.values()].map(v=>[v.id,v]));
const data={"notice":"Hand controls and guided training are on hold in this development build until its telemetry-free vision runtime is verified. Use mouse/keyboard for now. Saved recordings and models are unchanged; reinstalling the same runtime does not resolve this hold.","sections":[{"title":"Start here","paragraph":"Open Setup. Read the privacy explanation, select a physical camera and test it. Choose personal training or mouse/keyboard; you can train later. Optional output needs an installed OBS Virtual Camera driver."},{"title":"Try a filter without training","paragraph":"Choose Red, Yellow or Blue. Press Mouse frame, drag its corners, or focus the preview and choose a corner with 1–4, then use arrow keys. The effect appears only inside a valid frame. Clear frame or Escape removes it. Open menu supports mouse clicks or arrow keys and Enter; Back or Escape cancels."},{"title":"Use trained hands","paragraph":"Train all five supported poses in one guided workflow. Assign thumbs up, open palm and I love you to the three filters. Peace opens the head-following menu. Point with the opening hand's index finger and pinch/release over a choice. Hovering or pinching pauses menu movement. A direct-selection pose chooses its assigned filter; release, then form the two-hand frame again to see it."},{"title":"Form the hand-defined frame","paragraph":"Use two opposing L-shaped hands. The thumb/palm junctions are opposite corners. Thumb and index directions extend as rays; their cross-hand intersections make the other corners. The four fingertips are not the corners. Keep both hands visible, separate them, and avoid crossing or almost parallel rays. Hold a valid frame to keep the filter visible; releasing it removes the effect."},{"title":"Training and breaks","paragraph":"Follow each drawn pose and the small position, angle, distance and lighting variations. Only usable hand measurements advance collection. Completed rounds can resume later; an interrupted round restarts. Training and checks do not execute live commands. A candidate activates only after its checks and your explicit approval. A failed replacement keeps the previous model."},{"title":"If something fails","paragraph":"Camera missing: connect it, refresh cameras, check Windows camera permission, and close other apps using it. Hands missing: improve steady lighting, move closer and keep fingers visible without crossing. Mask uncertain: the palette remains while background decorations are omitted. Output missing: install OBS's virtual-camera driver and close other senders. Lower detail or frame rate is not proof that a GPU or model is broken."},{"title":"Input quality","paragraph":"After five seconds of sustained slow processing, Carl-Cam may request a smaller camera input. It waits at least ten seconds before another reduction. The picture frame and outgoing canvas stay the same size. Actual input dimensions and reduced quality are shown below the preview. Use Try higher input quality to request the preferred size again; there is no automatic upgrade. Some cameras ignore unsupported sizes. Hand commands pause while formats change; form a fresh pose afterward. This fallback cannot guarantee 30 frames per second."},{"title":"Privacy and local data","paragraph":"Carl-Cam has no account, uploads, telemetry, image/video recording, or automatic output. Numerical training rows and model artifacts stay under your local Carl-Cam data folder. Settings can delete samples separately, reset personal models, or delete both after confirmation. Creator export contains model artifacts without recordings and remains a candidate pending release tests."}],"guides":[{"pose":"frame","label":"Picture frame","width":208,"height":180},{"pose":"peace","label":"Peace sign — open menu","width":138,"height":180},{"pose":"thumbs_up","label":"Thumbs up","width":141,"height":180},{"pose":"open_palm","label":"Open palm","width":153,"height":180},{"pose":"ily","label":"I love you","width":160,"height":180}]};
const created=[componentSet.id,...variants.map(v=>v.id)],mutated=['59:531','59:532'],createdStyleIds=[];
function remember(n){created.push(n.id);return n;}
function paint(name) {
 let value=Object.values(vars.get(name).valuesByMode)[0];
 while(value.type==='VARIABLE_ALIAS')value=Object.values(byId.get(value.id).valuesByMode)[0];
 return figma.variables.setBoundVariableForPaint({type:'SOLID',color:{r:value.r,g:value.g,b:value.b}},'color',vars.get(name));
}
function frame(parent,name,direction='VERTICAL',gap='spacing/sm'){
 const n=remember(figma.createAutoLayout(direction));parent.appendChild(n);n.name=name;n.fills=[];
 n.setBoundVariable('itemSpacing',vars.get(gap));n.layoutSizingHorizontal='FILL';return n;
}
function instance(component,parent,name,properties,width=null){
 const n=remember(component.createInstance());parent.appendChild(n);n.name=name;n.setProperties(properties);
 if(width!==null)n.resize(width,component.height);
 created.push(...n.findAll(()=>true).map(c=>c.id));return n;
}
async function text(parent,label,name,width,styleId){
 const n=remember(figma.createText());parent.appendChild(n);n.name=name;
 n.fontName={family:'Arimo',style:'Regular'};await n.setTextStyleIdAsync(styleId);
 n.fills=[paint('color/cream')];n.resize(width,16);n.textAutoResize='HEIGHT';n.characters=label;return n;
}
const styles=await figma.getLocalTextStylesAsync();
let headingStyle=styles.find(s=>s.name==='Carl-Cam/Section heading');
if(!headingStyle){
 headingStyle=figma.createTextStyle();headingStyle.name='Carl-Cam/Section heading';
 headingStyle.fontName={family:'Arimo',style:'Bold'};headingStyle.fontSize=18;
 headingStyle.lineHeight={unit:'PIXELS',value:20};createdStyleIds.push(headingStyle.id);
}
const bodyStyle='S:b900cacd089357111e011385d9dba3b35f8eabec,';
const section=remember(figma.createComponent());section.name='Desktop/HelpSection';
section.description='Editable heading and wrapping paragraph for the offline Help page; shared 18px heading and 14px body styles.';
section.layoutMode='VERTICAL';section.resize(944,44);section.x=3900;section.y=3436;section.fills=[];
const bind=await bindVariablesToComponent(section,{itemSpacing:vars.get('spacing/sm').id});mutated.push(...bind.mutatedNodeIds);
const heading=await text(section,'Help section','Section heading',944,headingStyle.id);
const paragraph=await text(section,'Instructions','Section paragraph',944,bodyStyle);
const headingProperty=section.addComponentProperty('Heading','TEXT','Help section');
const bodyProperty=section.addComponentProperty('Body','TEXT','Instructions');
heading.componentPropertyReferences={characters:headingProperty};paragraph.componentPropertyReferences={characters:bodyProperty};
section.layoutSizingVertical='HUG';
componentSet.x=3900;componentSet.y=2444;
const labelProperty=componentSet.addComponentProperty('Label','TEXT','Picture frame');
const imageSlots=[];
for(let i=0;i<variants.length;i++){
 const variant=variants[i],guide=data.guides[i];
 variant.primaryAxisAlignItems='MIN';variant.counterAxisAlignItems='MIN';
 const binding=await bindVariablesToComponent(variant,{
 paddingTop:vars.get('spacing/sm').id,paddingBottom:vars.get('spacing/sm').id,
 paddingLeft:vars.get('spacing/sm').id,paddingRight:vars.get('spacing/sm').id,itemSpacing:vars.get('spacing/sm').id});
 mutated.push(...binding.mutatedNodeIds);
 const slot=frame(variant,'Centered owner drawing','HORIZONTAL');slot.resize(288,180);
 slot.layoutSizingHorizontal='FILL';slot.primaryAxisAlignItems='CENTER';slot.counterAxisAlignItems='CENTER';slot.itemSpacing=0;
 const image=remember(figma.createRectangle());slot.appendChild(image);image.name='Original drawing — '+guide.pose;
 image.resize(guide.width,guide.height);image.fills=[];imageSlots.push({pose:guide.pose,id:image.id,width:image.width,height:image.height});
 const label=await text(variant,guide.label,'Pose caption',288,bodyStyle);
 label.componentPropertyReferences={characters:labelProperty};
 variant.resize(304,220);variant.layoutSizingHorizontal='FIXED';variant.layoutSizingVertical='FIXED';
}
function grid(parent,name){
 const n=frame(parent,name);n.setBoundVariable('itemSpacing',vars.get('spacing/md'));
 for(let row=0;row<2;row++){
  const line=frame(n,'Guide row '+(row+1),'HORIZONTAL','spacing/md');
  for(const guide of data.guides.slice(row*3,row*3+3)){
   const component=variants.find(v=>v.name==='Pose='+guide.pose);
   instance(component,line,guide.label,{[labelProperty]:guide.label},304);
  }
 }
 return n;
}
const review=remember(figma.createAutoLayout('VERTICAL'));review.name='Desktop/Help Components Review';
review.x=3900;review.y=2824;review.fills=[paint('color/surface')];review.setBoundVariable('itemSpacing',vars.get('spacing/md'));
const specimen=instance(section,review,'Editable instruction section',{
 [headingProperty]:data.sections[0].title,[bodyProperty]:data.sections[0].paragraph},944);specimen.layoutSizingVertical='HUG';
grid(review,'Original pose-card specimens');
const screens=[];
for(const [wrapper,scrolled] of [[instructions,false],[illustrations,true]]){
 const chrome=instance(title,wrapper,'Native window controls',{'Title#29:1':'Carl-Cam — Help and quick start'},992);
 chrome.layoutSizingHorizontal='FILL';
 await text(wrapper,data.notice,'Current runtime privacy hold',992,bodyStyle);
 const viewport=frame(wrapper,'Help vertical scroll viewport','HORIZONTAL');
 viewport.itemSpacing=0;viewport.layoutSizingVertical='FILL';viewport.clipsContent=true;viewport.overflowDirection='VERTICAL';
 viewport.paddingLeft=1;viewport.paddingRight=1;viewport.paddingTop=1;viewport.paddingBottom=1;
 const body=frame(viewport,'Complete offline Help content');
 for(const p of ['paddingTop','paddingBottom','paddingLeft','paddingRight'])body.setBoundVariable(p,vars.get('spacing/md'));
 for(const item of data.sections){
  const group=instance(section,body,item.title,{[headingProperty]:item.title,[bodyProperty]:item.paragraph},944);
  group.layoutSizingHorizontal='FILL';group.layoutSizingVertical='HUG';
 }
 grid(body,'Five original supported-pose guides');
 const gutter=frame(viewport,'Native Windows scrollbar reserve');gutter.resize(14,508);gutter.layoutSizingHorizontal='FIXED';gutter.layoutSizingVertical='FILL';
 const footer=frame(wrapper,'Persistent practice and camera return','HORIZONTAL','spacing/md');
 const practice=instance(button,footer,'Practice switching',{'Label#23:0':'Practice both switching paths on my live camera'},736);
 practice.layoutSizingHorizontal='FILL';
 instance(button,footer,'Back to camera',{'Label#23:0':'Back to camera'},240);
 if(scrolled){body.layoutPositioning='ABSOLUTE';body.x=1;body.y=1-(body.height-508);body.resize(976,body.height);}
 const nodes=wrapper.findAll(()=>true),counts={};
 for(const n of nodes)counts[n.type]=(counts[n.type]||0)+1;
 screens.push({id:wrapper.id,name:wrapper.name,scrolled,descendantCount:nodes.length,counts,
  fontsMatch:nodes.filter(n=>n.type==='TEXT').every(n=>n.fontName.family==='Arimo'),
  bodyBounds:{x:body.x,y:body.y,width:body.width,height:body.height},
  majorBounds:wrapper.children.map(n=>({id:n.id,name:n.name,x:n.x,y:n.y,width:n.width,height:n.height}))});
 // Drawings remain deliberately pending until their exact assets are uploaded.
}
return {createdNodeIds:[...new Set(created)],mutatedNodeIds:[...new Set(mutated)],createdStyleIds,
 guideSetId:componentSet.id,labelProperty,imageSlots,sectionId:section.id,headingProperty,bodyProperty,
 headingStyleId:headingStyle.id,reviewId:review.id,screens,assetsPending:true};

