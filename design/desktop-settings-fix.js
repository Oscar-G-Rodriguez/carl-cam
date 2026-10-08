
const page=await figma.getNodeByIdAsync("0:1"); await figma.setCurrentPageAsync(page);
figma.skipInvisibleInstanceChildren=false;
const variables=new Map((await figma.variables.getLocalVariablesAsync()).map(v=>[v.name,v]));
const byId=new Map([...variables.values()].map(v=>[v.id,v]));
function literal(variable) {
  let value=Object.values(variable.valuesByMode)[0];
  while(value.type==="VARIABLE_ALIAS") value=Object.values(byId.get(value.id).valuesByMode)[0];
  return {r:value.r,g:value.g,b:value.b};
}
function paint(name) {
  const variable=variables.get(name);
  return figma.variables.setBoundVariableForPaint({type:"SOLID",color:literal(variable)},"color",variable);
}
const wrappers=await Promise.all(["53:383","53:384","53:385","53:386"].map(id=>figma.getNodeByIdAsync(id)));
const set=await figma.getNodeByIdAsync("53:358");
for(const root of [...wrappers,set]) for(const text of root.findAllWithCriteria({types:["TEXT"]}))
  for(const segment of text.getStyledTextSegments(["fontName"])) await figma.loadFontAsync(segment.fontName);
const changed=[];
for(const wrapper of wrappers) {
  wrapper.fills=[paint("color/surface")]; changed.push(wrapper.id);
  const outer=wrapper.findAllWithCriteria({types:["FRAME"]}).find(n=>n.name==="Settings content pane");
  outer.paddingLeft=2;outer.paddingRight=2;outer.paddingTop=0;outer.paddingBottom=2;changed.push(outer.id);
}
for(const variant of set.children) {
  variant.setBoundVariable("topLeftRadius",variables.get("radius/control"));
  variant.setBoundVariable("topRightRadius",variables.get("radius/control"));
  changed.push(variant.id);
}
const ring=await figma.getNodeByIdAsync("53:368");
ring.constraints={horizontal:"STRETCH",vertical:"STRETCH"}; changed.push(ring.id);
for(const wrapper of wrappers) for(const instance of wrapper.findAllWithCriteria({types:["INSTANCE"]})) {
  const component=await instance.getMainComponentAsync();
  if(component?.parent?.id===set.id) {
    const focus=instance.findAllWithCriteria({types:["RECTANGLE"]}).find(n=>n.name==="Keyboard focus only");
    if(focus) {focus.resize(instance.width-8,36);changed.push(focus.id);}
    changed.push(instance.id);
  }
}
const review=await figma.getNodeByIdAsync("53:369");
changed.push(...review.findAllWithCriteria({types:["INSTANCE"]}).map(n=>n.id));
return {createdNodeIds:[],mutatedNodeIds:[...new Set(changed)],screens:wrappers.map(w=>({
id:w.id,width:w.width,height:w.height,
focus:w.findAllWithCriteria({types:["RECTANGLE"]}).filter(n=>n.name==="Keyboard focus only").map(n=>({id:n.id,x:n.x,y:n.y,width:n.width,height:n.height})),
viewport:w.findAllWithCriteria({types:["FRAME"]}).filter(n=>["Settings group content","Vertical-only scroll viewport"].includes(n.name)).map(n=>({id:n.id,width:n.width,height:n.height})),
fontsMatch:w.findAllWithCriteria({types:["TEXT"]}).every(n=>n.fontName.family==="Arimo")
}))};
