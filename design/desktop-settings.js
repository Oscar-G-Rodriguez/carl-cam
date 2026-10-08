// Original editable Settings screens composed from Carl-Cam components.
// The 14px personal-data gutter reserves the native Windows scrollbar slot.
const page = await figma.getNodeByIdAsync('0:1');
await figma.setCurrentPageAsync(page);
figma.skipInvisibleInstanceChildren = false;
const wrapperIds = ['53:383', '53:384', '53:385', '53:386'];
const wrappers = await Promise.all(wrapperIds.map(id => figma.getNodeByIdAsync(id)));
if (wrappers.some(n => !n || n.type !== 'FRAME')) throw new Error('Missing Settings wrappers');
if (wrappers.some(n => n.children.length)) return {status: 'inspect_existing_composition', wrapperIds};
const componentIds = ['29:55', '23:50', '29:52', '33:121', '53:355', '53:357'];
const [title, button, selection, checkbox, tab, focusedTab] =
  await Promise.all(componentIds.map(id => figma.getNodeByIdAsync(id)));
for (const component of [title, button, selection, checkbox, tab, focusedTab]) {
  for (const text of component.findAllWithCriteria({types: ['TEXT']}))
    for (const segment of text.getStyledTextSegments(['fontName']))
      await figma.loadFontAsync(segment.fontName);
}
const variables = new Map((await figma.variables.getLocalVariablesAsync()).map(v => [v.name, v]));
const created = [], mutated = [];
function remember(node) { created.push(node.id); return node; }
function paint(name) {
  const byId = new Map([...variables.values()].map(v => [v.id, v]));
  let value = Object.values(variables.get(name).valuesByMode)[0];
  while (value.type === 'VARIABLE_ALIAS')
    value = Object.values(byId.get(value.id).valuesByMode)[0];
  return figma.variables.setBoundVariableForPaint(
    {type: 'SOLID', color: {r: value.r, g: value.g, b: value.b}}, 'color', variables.get(name));
}
function column(parent, name, direction = 'VERTICAL', gap = 'spacing/sm') {
  const node = remember(figma.createAutoLayout(direction)); parent.appendChild(node);
  node.name = name; node.fills = []; node.layoutSizingHorizontal = 'FILL';
  node.setBoundVariable('itemSpacing', variables.get(gap)); return node;
}
function instance(component, parent, name, properties, width) {
  const node = remember(component.createInstance()); parent.appendChild(node);
  node.name = name; node.setProperties(properties); node.resize(width, component.height);
  created.push(...node.findAll(() => true).map(n => n.id)); return node;
}
async function text(parent, value, name, width) {
  const node = remember(figma.createText()); parent.appendChild(node);
  node.name = name; node.fontName = {family: 'Arimo', style: 'Regular'};
  await node.setTextStyleIdAsync('S:b900cacd089357111e011385d9dba3b35f8eabec,');
  node.fills = [paint('color/cream')]; node.resize(width, 16);
  node.textAutoResize = 'HEIGHT'; node.characters = value; return node;
}
function padding(node, token) {
  for (const p of ['paddingTop', 'paddingBottom', 'paddingLeft', 'paddingRight'])
    node.setBoundVariable(p, variables.get(token));
}
const labels = ['Camera', 'Interaction and accessibility', 'Personal data and models', 'Virtual camera'];
const widths = [86, 212, 203, 127];
const cameraText = [
  'Stop the camera, choose an input on the Camera screen, then start it. OBS Virtual Camera is excluded to prevent a feedback loop.',
  'Frames stay live on this computer. Carl-Cam does not save or upload photos or video.',
  'Sustained slow processing can automatically request a smaller camera input. The Camera screen shows actual dimensions and offers Try higher input quality. Training and other private screens do not trigger automatic format changes; the final canvas remains 1280×720.'
];
const dataButtons = ['Use selected control model', 'Collect and train a new personal model',
  'Prepare selected creator-demo candidate…', 'Delete numerical training samples…',
  'Reset personal gesture models…', 'Delete my training data…', 'Reset all Carl-Cam local data…'];
const report = [];
for (let group = 0; group < wrappers.length; group++) {
  const wrapper = wrappers[group]; mutated.push(wrapper.id);
  wrapper.resize(1024, 700); wrapper.layoutSizingHorizontal = 'FIXED';
  wrapper.layoutSizingVertical = 'FIXED'; padding(wrapper, 'spacing/md');
  wrapper.setBoundVariable('itemSpacing', variables.get('spacing/sm'));
  wrapper.fills = [paint('color/surface')];
  const heading = instance(title, wrapper, 'Native window controls',
    {'Title#29:1': 'Carl-Cam — Settings'}, 992);
  heading.layoutSizingHorizontal = 'FILL';
  const workspace = column(wrapper, 'Settings tabs and vertical content');
  workspace.itemSpacing = 0; workspace.layoutSizingVertical = 'FILL';
  const nav = column(workspace, 'Native keyboard tab navigation', 'HORIZONTAL');
  nav.itemSpacing = 0; nav.resize(992, 44); nav.layoutSizingHorizontal = 'FILL';
  for (let i = 0; i < labels.length; i++)
    instance(i === group ? focusedTab : tab, nav, labels[i],
      {'Label#53:0': labels[i]}, widths[i]);
  const outer = column(workspace, 'Settings content pane');
  outer.itemSpacing = 0; outer.layoutSizingVertical = 'FILL';
  outer.paddingLeft = 2; outer.paddingRight = 2; outer.paddingTop = 0; outer.paddingBottom = 2;
  outer.strokes = [paint('color/control/disabled-border')]; outer.strokeWeight = 1;
  outer.strokeAlign = 'INSIDE';
  const viewport = column(outer, 'Vertical-only scroll viewport', 'HORIZONTAL');
  viewport.itemSpacing = 0; viewport.layoutSizingVertical = 'FILL'; viewport.clipsContent = true;
  const content = column(viewport, 'Settings group content');
  padding(content, 'spacing/md'); content.clipsContent = false;
  if (group === 2) {
    content.resize(972, 520); content.layoutSizingHorizontal = 'FILL';
    viewport.overflowDirection = 'VERTICAL';
    const gutter = column(viewport, 'Native Windows scrollbar reserve');
    gutter.resize(14, 502); gutter.layoutSizingHorizontal = 'FIXED';
    gutter.layoutSizingVertical = 'FILL';
  } else content.layoutSizingVertical = 'FILL';
  const bodyWidth = group === 2 ? 940 : 954;
  if (group === 0) {
    for (let i = 0; i < cameraText.length; i++)
      await text(content, cameraText[i], 'Camera guidance ' + (i + 1), bodyWidth);
  } else if (group === 1) {
    for (const label of ['Reduce motion — keep decorative layers still', 'High-contrast desktop controls']) {
      const check = instance(checkbox, content, label, {'Label#33:0': label}, bodyWidth);
      check.layoutSizingHorizontal = 'FILL';
    }
    const form = column(content, 'Dwell and one-to-one assignments');
    for (const [label, value] of [['Gesture dwell', '300 ms gesture hold'],
      ['Thumbs up', 'Red'], ['Open palm', 'Yellow'], ['I love you', 'Blue']]) {
      const row = column(form, label + ' row', 'HORIZONTAL', 'spacing/md');
      row.counterAxisAlignItems = 'CENTER';
      await text(row, label, label + ' label', 88);
      const choice = instance(selection, row, label + ' selector', {'Value#29:0': value}, 850);
      choice.layoutSizingHorizontal = 'FILL';
    }
    await text(content,
      'Peace opens the menu. The two-hand frame shows the effect. Assignment changes apply to future commands without relabeling or retraining.',
      'Assignment guidance', bodyWidth);
  } else if (group === 2) {
    await text(content,
      'Carl-Cam local storage: 0 bytes.\nOnly numerical recordings, model artifacts, settings and local diagnostics are retained.',
      'Local storage accounting', bodyWidth);
    await text(content, 'Mouse/keyboard controls — personal training available',
      'Current model readiness', bodyWidth);
    const models = instance(selection, content, 'Explicit tested model selection',
      {'Value#29:0': 'Mouse/keyboard — learned commands off'}, bodyWidth);
    models.layoutSizingHorizontal = 'FILL';
    for (const value of dataButtons) {
      const action = instance(button, content, value, {'Label#23:0': value}, bodyWidth);
      action.layoutSizingHorizontal = 'FILL';
    }
  } else {
    await text(content,
      "OBS Virtual Camera is optional. Install OBS yourself if needed; Carl-Cam does not install drivers or start OBS. Start your physical camera, then explicitly press Start virtual camera on the Camera screen. In the receiving app choose OBS Virtual Camera. Do not start OBS's own virtual-camera sender at the same time. Output always starts off on each launch, and becomes black if the source is too old.",
      'Optional output and freshness guidance', bodyWidth);
  }
  const footer = column(wrapper, 'Persistent save and camera return', 'HORIZONTAL', 'spacing/md');
  const save = instance(button, footer, 'Save settings', {'Label#23:0': 'Save settings'}, 736);
  save.layoutSizingHorizontal = 'FILL';
  instance(button, footer, 'Back to camera', {'Label#23:0': 'Back to camera'}, 240);
  wrapper.placeholder = false;
  const nodes = wrapper.findAll(() => true), counts = {};
  for (const node of nodes) counts[node.type] = (counts[node.type] || 0) + 1;
  report.push({id: wrapper.id, name: wrapper.name, descendantCount: nodes.length, counts,
    fontsMatch: nodes.filter(n => n.type === 'TEXT').every(n => n.fontName.family === 'Arimo'),
    imageFilledNodes: nodes.filter(n => 'fills' in n && Array.isArray(n.fills) &&
      n.fills.some(p => p.type === 'IMAGE')).map(n => n.id),
    majorBounds: wrapper.children.map(n => ({id:n.id,name:n.name,x:n.x,y:n.y,width:n.width,height:n.height}))});
}
return {createdNodeIds: [...new Set(created)], mutatedNodeIds: mutated, screens: report};
