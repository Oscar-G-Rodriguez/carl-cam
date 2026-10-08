// Editable six-panel desktop diagnostics. Camera-off previews are components,
// never screenshots or invented signals. This composition uses original tokens.
const page = await figma.getNodeByIdAsync('0:1');
await figma.setCurrentPageAsync(page);
figma.skipInvisibleInstanceChildren = false;
const fonts = await figma.listAvailableFontsAsync();
for (const style of ['Regular', 'Bold']) {
  if (!fonts.some(f => f.fontName.family === 'Arimo' && f.fontName.style === style))
    throw new Error('Required bundled Arimo font unavailable');
  await figma.loadFontAsync({family: 'Arimo', style});
}
const [wrapper, title, button, focused, preview] = await Promise.all(
  ['48:316', '29:55', '23:50', '23:54', '29:83'].map(id => figma.getNodeByIdAsync(id)));
if (!wrapper) throw new Error('Inspect wrapper must be created first');
if (wrapper.children.length) return {createdNodeIds: [], wrapperId: wrapper.id, status: 'already_composed'};
for (const component of [title, button, focused, preview]) {
  for (const text of component.findAllWithCriteria({types: ['TEXT']})) {
    for (const segment of text.getStyledTextSegments(['fontName']))
      await figma.loadFontAsync(segment.fontName);
  }
}
const variables = new Map((await figma.variables.getLocalVariablesAsync()).map(v => [v.name, v]));
const created = [], mutated = [wrapper.id];
function remember(node) { created.push(node.id); return node; }
function paint(name) {
  return figma.variables.setBoundVariableForPaint(
    {type: 'SOLID', color: {r: 0, g: 0, b: 0}}, 'color', variables.get(name));
}
// Adapted from the skill's bindVariablesToComponent helper: resolve all IDs
// together and bind only the properties supplied for this original component.
async function bindVariablesToComponent(component, bindings) {
  const entries = Object.entries(bindings);
  const resolved = await Promise.all(entries.map(([, id]) => figma.variables.getVariableByIdAsync(id)));
  for (let i = 0; i < entries.length; i++) {
    const [property] = entries[i], variable = resolved[i];
    if (!variable) throw new Error('Missing token ' + property);
    if (property === 'fills' || property === 'strokes') {
      component[property] = [figma.variables.setBoundVariableForPaint(
        {type: 'SOLID', color: {r: 0, g: 0, b: 0}}, 'color', variable)];
    } else component.setBoundVariable(property, variable);
  }
  mutated.push(component.id);
}
function container(parent, name, direction = 'VERTICAL', gap = 'spacing/sm') {
  const node = remember(figma.createAutoLayout(direction));
  parent.appendChild(node); node.name = name; node.fills = [];
  node.layoutSizingHorizontal = 'FILL';
  node.setBoundVariable('itemSpacing', variables.get(gap));
  return node;
}
function instance(component, parent, name, properties = {}) {
  const node = remember(component.createInstance());
  parent.appendChild(node); node.name = name; node.setProperties(properties);
  created.push(...node.findAll(() => true).map(n => n.id)); return node;
}
async function text(parent, label, name, width) {
  const node = remember(figma.createText()); parent.appendChild(node); node.name = name;
  node.fontName = {family: 'Arimo', style: 'Regular'};
  await node.setTextStyleIdAsync('S:b900cacd089357111e011385d9dba3b35f8eabec,');
  node.fills = [paint('color/cream')]; node.resize(width, 16);
  node.textAutoResize = 'HEIGHT'; node.characters = label; return node;
}
let panel = page.children.find(n => n.type === 'COMPONENT' && n.name === 'Desktop/DiagnosticPanel');
let labelProperty;
if (!panel) {
  panel = remember(figma.createComponent()); panel.name = 'Desktop/DiagnosticPanel';
  panel.description = 'Editable signal label with a centered 16:9 live scene. No retained camera image. Used for the six desktop-only diagnostic stages.';
  panel.layoutMode = 'VERTICAL'; panel.resize(379, 275); panel.x = 3900; panel.y = 1780;
  const spacing = variables.get('spacing/sm').id;
  await bindVariablesToComponent(panel, {fills: variables.get('color/surface').id,
    paddingTop: spacing, paddingBottom: spacing, paddingLeft: spacing,
    paddingRight: spacing, itemSpacing: spacing});
  const label = await text(panel, 'Original (mirrored)', 'Signal label', 363);
  labelProperty = panel.addComponentProperty('Label', 'TEXT', 'Original (mirrored)');
  label.componentPropertyReferences = {characters: labelProperty};
  const viewport = container(panel, 'Live scene viewport');
  viewport.layoutSizingVertical = 'FILL';
  viewport.primaryAxisAlignItems = 'CENTER'; viewport.counterAxisAlignItems = 'CENTER';
  const scene = instance(preview, viewport, 'No camera image retained'); scene.resize(363, 204.1875);
} else labelProperty = Object.keys(panel.componentPropertyDefinitions).find(k => k.startsWith('Label#'));
const header = instance(title, wrapper, 'Native window controls',
  {'Title#29:1': 'Carl-Cam — Inspect live processing'});
header.layoutSizingHorizontal = 'FILL';
const grid = container(wrapper, 'Six diagnostic signals', 'VERTICAL', 'spacing/md');
grid.layoutSizingVertical = 'FILL';
const names = ['Original (mirrored)', 'Linear luminance', 'Sobel edge strength',
  'Person mask', 'Hand landmarks and quadrilateral', 'Final graphic composite'];
const panels = [];
for (let rowIndex = 0; rowIndex < 2; rowIndex++) {
  const row = container(grid, 'Diagnostic row ' + (rowIndex + 1), 'HORIZONTAL', 'spacing/md');
  row.layoutSizingVertical = 'FILL';
  for (let column = 0; column < 3; column++) {
    const name = names[rowIndex * 3 + column];
    const cell = instance(panel, row, name, {[labelProperty]: name});
    cell.layoutSizingHorizontal = 'FILL'; cell.layoutSizingVertical = 'FILL'; panels.push(cell);
  }
}
await text(wrapper, 'Start a camera to inspect its current signals.', 'Current frame measurements', 1168);
await text(wrapper, 'All panels are transient. Hand commands are disabled here, and virtual output receives only the ordinary unmirrored camera—not these diagnostics. A missing exact-frame mask disables background motifs rather than using an old silhouette.', 'Inspection safety explanation', 1168);
const footer = container(wrapper, 'Comparison and persistent camera return', 'HORIZONTAL', 'spacing/md');
const compare = instance(focused, footer, 'Optional comparison',
  {'Label#23:0': 'Compare with MediaPipe (optional)'}); compare.layoutSizingHorizontal = 'FILL';
const back = instance(button, footer, 'Back to camera', {'Label#23:0': 'Back to camera'});
back.resize(240, 46);
for (const cell of panels) {
  const label = cell.findAllWithCriteria({types: ['TEXT']}).find(n => n.name === 'Signal label');
  label.resize(cell.width - 16, 16); label.textAutoResize = 'HEIGHT'; mutated.push(label.id);
  const viewport = cell.findAllWithCriteria({types: ['FRAME']}).find(n => n.name === 'Live scene viewport');
  const scene = viewport.children.find(n => n.type === 'INSTANCE');
  const scale = Math.min(viewport.width / 1280, viewport.height / 720);
  scene.resize(1280 * scale, 720 * scale); mutated.push(scene.id);
}
wrapper.placeholder = false;
const descendants = wrapper.findAll(() => true), counts = {};
for (const node of descendants) counts[node.type] = (counts[node.type] || 0) + 1;
return {createdNodeIds: [...new Set(created)], mutatedNodeIds: [...new Set(mutated)],
  wrapperId: wrapper.id, panelComponentId: panel.id, labelProperty,
  panelInstances: panels.map(n => ({id: n.id, name: n.name, width: n.width, height: n.height})),
  descendantCount: descendants.length, counts,
  fontsMatch: descendants.filter(n => n.type === 'TEXT').every(n => n.fontName.family === 'Arimo'),
  imageFilledNodes: descendants.filter(n => 'fills' in n && Array.isArray(n.fills) && n.fills.some(p => p.type === 'IMAGE')).map(n => n.id)};
