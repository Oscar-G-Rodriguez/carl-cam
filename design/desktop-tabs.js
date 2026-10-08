// Original tab family. The supplied helper switches the page once.
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


figma.skipInvisibleInstanceChildren = false;

const page = await figma.getNodeByIdAsync('0:1');
const old = page.children.find(n => n.name === 'Desktop/Tab');
if (old) return {createdNodeIds: [], status: 'already_exists', tabSetId: old.id};
const fonts = await figma.listAvailableFontsAsync();
if (!fonts.some(f => f.fontName.family === 'Arimo' && f.fontName.style === 'Regular')) throw new Error('Arimo unavailable');
await figma.loadFontAsync({family:'Arimo',style:'Regular'});
const vars = new Map((await figma.variables.getLocalVariablesAsync()).map(v=>[v.name,v]));
const {componentSet, variants} = await createComponentWithVariants({
 name:'Desktop/Tab', description:'Settings navigation using native Qt keyboard behavior. Selected state has a cyan lower bar; keyboard focus adds a separate inset yellow ring.',
 variantAxes:{State:['Default','Selected','Focused']},
 baseProps:{width:216,height:44,layoutMode:'VERTICAL',itemSpacing:0},page});
const created=[componentSet.id,...variants.map(n=>n.id)];
componentSet.x=3900;componentSet.y=2200;
function paint(name){
 let value=Object.values(vars.get(name).valuesByMode)[0];
 const byId=new Map([...vars.values()].map(v=>[v.id,v]));
 while(value.type==='VARIABLE_ALIAS') value=Object.values(byId.get(value.id).valuesByMode)[0];
 return figma.variables.setBoundVariableForPaint({type:'SOLID',color:{r:value.r,g:value.g,b:value.b}},'color',vars.get(name));
}
const property=componentSet.addComponentProperty('Label','TEXT','Settings group');
for(const variant of variants){
 const selected=variant.name!=='State=Default';
 variant.fills=[paint(selected?'color/chrome':'color/surface')];
 variant.strokes=[paint('color/cyan')];variant.setBoundVariable('strokeWeight',vars.get('stroke/control'));variant.strokeAlign='INSIDE';
 variant.setBoundVariable('topLeftRadius',vars.get('radius/control'));variant.setBoundVariable('topRightRadius',vars.get('radius/control'));variant.bottomLeftRadius=0;variant.bottomRightRadius=0;
 const labelRow=figma.createAutoLayout('HORIZONTAL');variant.appendChild(labelRow);created.push(labelRow.id);labelRow.name='Centered tab label';labelRow.fills=[];
 labelRow.layoutSizingHorizontal='FILL';labelRow.layoutSizingVertical='FILL';labelRow.primaryAxisAlignItems='CENTER';labelRow.counterAxisAlignItems='CENTER';
 const text=figma.createText();labelRow.appendChild(text);created.push(text.id);text.name='Tab label';text.fontName={family:'Arimo',style:'Regular'};
 await text.setTextStyleIdAsync('S:b900cacd089357111e011385d9dba3b35f8eabec,');text.characters='Settings group';text.fills=[paint('color/cream')];text.componentPropertyReferences={characters:property};
 const line=figma.createRectangle();variant.appendChild(line);created.push(line.id);line.name='Current-group underline';line.resize(216,3);line.layoutSizingHorizontal='FILL';line.fills=[paint('color/cyan')];line.visible=selected;
 if(variant.name==='State=Focused'){
 const ring=figma.createRectangle();variant.appendChild(ring);created.push(ring.id);ring.name='Keyboard focus only';ring.layoutPositioning='ABSOLUTE';ring.x=4;ring.y=4;ring.resize(208,36);ring.constraints={horizontal:'STRETCH',vertical:'STRETCH'};ring.cornerRadius=4;ring.fills=[];ring.strokes=[paint('color/yellow')];ring.setBoundVariable('strokeWeight',vars.get('stroke/focus'));ring.strokeAlign='INSIDE';
 }
 variant.resize(216,44);variant.layoutSizingHorizontal='FIXED';variant.layoutSizingVertical='FIXED';
}
const review=figma.createAutoLayout('HORIZONTAL');review.name='Desktop/Tab States Review';review.x=3900;review.y=2340;review.fills=[paint('color/surface')];review.setBoundVariable('itemSpacing',vars.get('spacing/md'));created.push(review.id);
for(const variant of variants){const n=variant.createInstance();review.appendChild(n);n.setProperties({[property]:variant.name.slice(6)});created.push(n.id,...n.findAll(()=>true).map(c=>c.id));}
const wrappers=[];
let right=Math.max(...page.children.map(n=>n.x+n.width));
for(const name of ['Camera','Interaction','Personal data','Virtual camera']){
 const existing=page.children.find(n=>n.name==='Desktop/Settings/'+name);
 if(existing){wrappers.push({name,id:existing.id,status:'already_exists'});continue;}
 const n=figma.createAutoLayout('VERTICAL');n.name='Desktop/Settings/'+name;n.resize(1024,700);n.layoutSizingHorizontal='FIXED';n.layoutSizingVertical='FIXED';n.x=right+160;n.y=100;right=n.x+n.width;n.fills=[paint('color/surface')];n.placeholder=true;
 for(const p of ['paddingLeft','paddingRight','paddingTop','paddingBottom'])n.setBoundVariable(p,vars.get('spacing/md'));
 n.setBoundVariable('itemSpacing',vars.get('spacing/sm'));created.push(n.id);wrappers.push({name,id:n.id});
}
return {createdNodeIds:created,tabSetId:componentSet.id,labelProperty:property,variants:variants.map(n=>({id:n.id,state:n.name})),reviewId:review.id,wrappers};

