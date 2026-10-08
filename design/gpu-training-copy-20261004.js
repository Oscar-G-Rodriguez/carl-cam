// Run only in the approved original Carl-Cam Figma file.
const page = await figma.getNodeByIdAsync('0:1');
if (!page || page.type !== 'PAGE') throw new Error('Expected Carl-Cam design page');
await figma.setCurrentPageAsync(page);
const ids = ['65:652', '67:694', '67:706', '67:718', '67:768', '67:813'];
const [review, panel, checkbox, template, fitting, comparison] = await Promise.all(
  ids.map(id => figma.getNodeByIdAsync(id))
);
if (!review || review.type !== 'FRAME' || !panel || panel.type !== 'FRAME'
    || !checkbox || checkbox.type !== 'INSTANCE' || !template || template.type !== 'TEXT'
    || !fitting || fitting.type !== 'TEXT' || !comparison || comparison.type !== 'TEXT') {
  throw new Error('Expected editable Training nodes');
}
const existing = panel.findAllWithCriteria({types: ['TEXT']})
  .filter(n => n.name === 'NVIDIA training availability');
if (existing.length > 1) throw new Error('Ambiguous previous GPU-copy update');
const textNodes = review.findAllWithCriteria({types: ['TEXT']}).concat([fitting, comparison]);
const fonts = [...new Map(textNodes.flatMap(n => n.getStyledTextSegments(['fontName']))
  .map(segment => [JSON.stringify(segment.fontName), segment.fontName])).values()];
await Promise.all(fonts.map(font => figma.loadFontAsync(font)));
const mutatedNodeIds = [panel.id, checkbox.id];
checkbox.setProperties({State: 'Default', Checked: 'False'});
checkbox.name = 'Optional NVIDIA training — unchecked by default';
mutatedNodeIds.push(...checkbox.findAllWithCriteria({types: ['TEXT', 'FRAME', 'INSTANCE']}).map(n => n.id));
const availability = existing[0] || template.clone();
panel.appendChild(availability);
availability.name = 'NVIDIA training availability';
availability.textAutoResize = 'HEIGHT';
availability.layoutSizingHorizontal = 'FIXED';
availability.resize(222, 48);
availability.characters = 'NVIDIA training is available. Leave this unchecked to train on CPU.';
if (availability.width <= 0) throw new Error('GPU availability text has no width');
const createdNodeIds = existing.length ? [] : [availability.id];
if (existing.length) mutatedNodeIds.push(availability.id);
const result = {
  fileKey: 'GOpu9lAhmkGMvEg8aUzi8q', pageId: page.id,
  reviewId: review.id, gpuCheckboxId: checkbox.id, availabilityId: availability.id,
  createdNodeIds, mutatedNodeIds: [...new Set(mutatedNodeIds)],
  availability: availability.characters,
  availabilityBounds: {x: availability.x, y: availability.y, width: availability.width, height: availability.height},
  existingFittingAndComparisonPreserved: true,
  note: 'GPU-present review fixture. Runtime supplies CPU-only, unavailable, or initialization-failed text when appropriate. Fitting progress and comparison summaries are actual dynamic results, not invented candidate scores.'
};
await review.screenshot({scale: 1});
return result;
