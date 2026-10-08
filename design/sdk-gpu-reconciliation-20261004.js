// Targeted, retry-safe update of existing editable screens; no UI raster import.
const page = await figma.getNodeByIdAsync('0:1');
await figma.setCurrentPageAsync(page);
const roots = await Promise.all(['34:147', '53:383', '59:531'].map(id => figma.getNodeByIdAsync(id)));
const allText = roots.flatMap(root => root.findAllWithCriteria({types: ['TEXT']}));
const fonts = new Map(allText.flatMap(node => node.getStyledTextSegments(['fontName']))
  .map(segment => [JSON.stringify(segment.fontName), segment.fontName]));
fonts.set('body', {family: 'Arimo', style: 'Regular'});
await Promise.all([...fonts.values()].map(font => figma.loadFontAsync(font)));
const createdNodeIds = [], mutatedNodeIds = [];
const updates = {
  '35:167': 'Carl-Cam processes camera images locally and does not save photos or video. MediaPipe may send performance and usage metrics to Google; its camera images and video stay on this computer. Models are bundled. Personal training asks separately before saving numerical hand measurements and labels. Settings can delete samples and personal models. Virtual-camera output starts only when you explicitly request it in this launch.',
  '35:176': 'Vision runtime is unavailable or not verified. Mouse and keyboard still work.',
  '60:603': 'Hand controls and training are on hold because the vision runtime is missing or unsupported. Use mouse/keyboard for now. Saved recordings and models are unchanged.'
};
for (const [id, value] of Object.entries(updates)) {
  const node = await figma.getNodeByIdAsync(id);
  if (!node || node.type !== 'TEXT') throw new Error('Expected editable source text: ' + id);
  node.characters = value;
  if (id !== '35:167') node.visible = false; // The supported runtime has no default hold.
  mutatedNodeIds.push(node.id);
}
const acknowledgment = await figma.getNodeByIdAsync('35:168');
acknowledgment.setProperties({'Label#33:0': 'I understand local processing and MediaPipe usage metrics'});
mutatedNodeIds.push(acknowledgment.id, ...acknowledgment.findAllWithCriteria({types: ['TEXT']}).map(n => n.id));
const privacySection = await figma.getNodeByIdAsync('60:627');
privacySection.setProperties({'Body#60:1': 'Processing and training run locally. MediaPipe may send performance and usage metrics to Google; its input images and video are not sent to Google. Carl-Cam does not save photos/video, upload training samples, or start output automatically. Numerical rows and models stay in your local data folder. Settings can delete samples or models after confirmation. Creator export contains models without recordings and remains a candidate pending release tests.'});
mutatedNodeIds.push(privacySection.id, ...privacySection.findAllWithCriteria({types: ['TEXT']}).map(n => n.id));
const content = await figma.getNodeByIdAsync('55:408');
const vars = new Map((await figma.variables.getLocalVariablesAsync()).map(v => [v.name, v]));
const selection = await figma.getNodeByIdAsync('29:52');
async function body(name, value) {
  let node = content.children.find(n => n.name === name);
  if (!node) {
    node = figma.createText(); content.appendChild(node); node.name = name;
    createdNodeIds.push(node.id);
  } else mutatedNodeIds.push(node.id);
  node.fontName = {family: 'Arimo', style: 'Regular'};
  await node.setTextStyleIdAsync('S:b900cacd089357111e011385d9dba3b35f8eabec,');
  node.fills = [figma.variables.setBoundVariableForPaint(
    {type: 'SOLID', color: {r: 1, g: 1, b: 1}}, 'color', vars.get('color/cream'))];
  node.textAutoResize = 'HEIGHT'; node.resize(954, 16); node.characters = value;
  node.layoutSizingHorizontal = 'FIXED';
  return node;
}
await body('Segmentation processor label', 'Segmentation processor');
let choice = content.children.find(n => n.name === 'Segmentation processor choice');
if (!choice) {
  choice = selection.createInstance(); content.appendChild(choice);
  choice.name = 'Segmentation processor choice';
  createdNodeIds.push(choice.id, ...choice.findAll(() => true).map(n => n.id));
} else mutatedNodeIds.push(choice.id);
choice.setProperties({'Value#29:0': 'CPU — available on every supported computer'});
choice.resize(954, 46); choice.layoutSizingHorizontal = 'FILL';
await body('Segmentation processor guidance', 'This changes hair, skin and clothing masking only. Hand tracking and your gesture model stay separate. Save, then stop and restart the camera to apply. If the verified GPU component is unavailable or fails, Carl-Cam shows the CPU fallback. The current GPU component is available in the source development build; installer integration is still being verified.');
mutatedNodeIds.push(content.id);
const screenEvidence = roots.map(root => {
  const nodes = root.findAll(() => true), types = {};
  for (const node of nodes) types[node.type] = (types[node.type] || 0) + 1;
  return {id: root.id, name: root.name, width: root.width, height: root.height,
    descendantCount: nodes.length, types,
    textFamilies: [...new Set(root.findAllWithCriteria({types: ['TEXT']})
      .flatMap(node => node.getStyledTextSegments(['fontName']).map(s => s.fontName.family)))],
    imageNodes: nodes.filter(n => 'fills' in n && Array.isArray(n.fills)
      && n.fills.some(paint => paint.type === 'IMAGE'))
      .map(n => ({id: n.id, name: n.name, type: n.type, width: n.width, height: n.height}))};
});
if (screenEvidence.some(screen => screen.textFamilies.some(f => f !== 'Arimo')))
  throw new Error('Source typography mismatch');
return {createdNodeIds, mutatedNodeIds: [...new Set(mutatedNodeIds)], screenEvidence,
  graphicsUnchanged: true, sourceFontFamily: 'Arimo',
  selectorOptions: ['CPU — available on every supported computer',
    'NVIDIA GPU — experimental, with CPU fallback'],
  componentDiscovery: 'Python Qt has no Code Connect files; existing local Selection, Checkbox and HelpSection components reused.',
  setupHoldVisible: false, helpHoldVisible: false};
