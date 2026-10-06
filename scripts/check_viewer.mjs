// Optional browser QA: use an installed Playwright module; never starts the camera.
// DTR_PLAYWRIGHT_MODULE may point at an existing installation's index.mjs.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const { chromium } = await import(process.env.DTR_PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = process.env.DTR_PLAYWRIGHT_OUTPUT || path.join(root, 'output/playwright', `viewer-${Date.now()}`);
fs.mkdirSync(output, { recursive: true });
const source = path.join(root, 'data/roboflow-dtr-v10-grouped/coco/valid');
const annotations = JSON.parse(fs.readFileSync(path.join(source, '_annotations.coco.json')));
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  page.on('console', message => {
    if (message.type() === 'error') errors.push(message.text());
  });
  await page.goto(process.env.DTR_VIEWER_URL || 'http://127.0.0.1:8765');
  if (!await page.evaluate(() => window.isSecureContext)) throw Error('Camera requires a secure context');
  await page.waitForFunction(() => !document.getElementById('task').disabled);
  if (await page.locator('#fps').inputValue() !== '5' || await page.locator('#fps').isDisabled()) {
    throw Error('Default processing FPS cap must be editable and equal to 5');
  }
  const downloads = [];
  async function checkDownloadState(disabled) {
    for (const id of ['download-frame', 'download-mask']) {
      if (await page.locator(`#${id}`).isDisabled() !== disabled) throw Error(`Wrong ${id} state`);
    }
  }
  async function checkDownloads(task) {
    await checkDownloadState(false);
    let pairPrefix;
    for (const [id, suffix, width, height] of [
      ['download-frame', 'annotated', 960, 720], ['download-mask', 'mask', 320, 240],
    ]) {
      const expected = await page.evaluate(kind => kind === 'annotated'
        ? document.getElementById('frame').toDataURL('image/png')
        : document.getElementById('mask').src, suffix);
      const pending = page.waitForEvent('download');
      await page.locator(`#${id}`).click();
      const download = await pending;
      if (await download.failure()) throw Error('Browser download failed');
      const filename = download.suggestedFilename();
      if (!filename.startsWith(`dtr-${task}-image-`) || !filename.endsWith(`-${suffix}.png`)) {
        throw Error(`Invalid PNG filename: ${filename}`);
      }
      const prefix = filename.slice(0, -`-${suffix}.png`.length);
      if (pairPrefix && prefix !== pairPrefix) throw Error('Still-image download pair differs');
      pairPrefix = prefix;
      const bytes = fs.readFileSync(await download.path());
      if (!bytes.equals(Buffer.from(expected.split(',')[1], 'base64')) ||
          bytes.subarray(0, 8).toString('hex') !== '89504e470d0a1a0a' ||
          bytes.readUInt32BE(16) !== width || bytes.readUInt32BE(20) !== height) {
        throw Error(`Downloaded PNG does not match displayed ${suffix}`);
      }
      await download.saveAs(path.join(output, filename));
      downloads.push({filename, width, height, bytes: bytes.length, matchesDisplayedImage: true});
    }
  }
  await checkDownloadState(true);
  const initial = await page.locator('body').innerText();
  fs.writeFileSync(path.join(output, 'initial.txt'), initial);
  if (!initial.includes('DTR Vision') || !initial.includes('V4 teacher')) {
    throw Error('Wrong page or model engine');
  }
  const metadata = await page.evaluate(async () => {
    const { token, ...config } = await (await fetch('/api/config')).json();
    return config; // Do not persist the session token.
  });
  for (const task of ['balloon', 'goal']) {
    const expected = JSON.parse(fs.readFileSync(path.join(root,
      `runs/${task}-proposals-20261005/teacher.json`)));
    if (metadata.tasks[task].model_sha256 !== expected.sha256) {
      throw Error(`Viewer is not using the selected proposal-adapted ${task} checkpoint`);
    }
  }
  if (metadata.duplicate_policy !== 'nested' || metadata.proposal_profile !== (process.env.DTR_PROFILE || 'balloon_components')) {
    throw Error('Viewer has stale or experimental vision settings');
  }
  if (metadata.tasks.goal.proposal_limit !== Number(process.env.DTR_GOAL_LIMIT || 12)) {
    throw Error('Viewer goal budget differs from the expected experiment');
  }
  const observations = {};
  for (const task of ['goal', 'balloon']) {
    await page.getByLabel('Task', { exact: true }).selectOption(task);
    await checkDownloadState(true);
    const resultPromise = page.waitForResponse(response =>
      response.url().includes(`/api/frame?task=${task}`) && response.request().method() === 'POST');
    if (task === 'goal') {
      await page.getByLabel('Test image', { exact: true }).setInputFiles(
        path.join(source, annotations.images[0].file_name));
    } else {
      await page.getByRole('button', { name: 'Analyze image', exact: true }).click();
    }
    const response = await resultPromise;
    const result = await response.json();
    if (!response.ok() || result.flight_commands !== null || result.deployment_approved) {
      throw Error(`Invalid research-only response for ${task}`);
    }
    await page.getByText('Image analyzed once. No camera access.', { exact: false }).waitFor();
    if (await page.locator('#source-size').innerText() !== '640 × 640' ||
        await page.locator('#capped-size').innerText() !== '640 × 640' ||
        await page.locator('#actual-fps').innerText() !== 'N/A — single image') {
      throw Error('Still-image source dimensions or FPS reporting is incorrect');
    }
    if (await page.locator('#error').isVisible()) throw Error(await page.locator('#error').innerText());
    if (!await page.locator('#mask').isVisible()) throw Error('Mask missing');
    observations[task] = result.observations;
    if (task === 'goal') {
      if (!result.observations.some(row => row.suppressed && row.raw_accepted && !row.accepted)) {
        throw Error('Expected nested-goal duplicate suppression in the fixed QA frame');
      }
      if (!(await page.locator('#rows').innerText()).includes('Duplicate suppressed')) {
        throw Error('Suppression reason is not visible');
      }
    }
    await checkDownloads(task);
    await page.screenshot({ path: path.join(output, `${task}-desktop.png`), fullPage: true });
  }
  await page.getByRole('button', { name: 'Stop', exact: true }).click();
  await page.getByText('Stopped. Camera released.', { exact: true }).waitFor();
  await checkDownloadState(true);
  await page.setViewportSize({ width: 390, height: 844 });
  const mobileResponse = page.waitForResponse(response =>
    response.url().includes('/api/frame?task=balloon') && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Analyze image', exact: true }).click();
  await mobileResponse;
  await page.getByText('Image analyzed once. No camera access.', { exact: false }).waitFor();
  await checkDownloads('balloon');
  await page.screenshot({ path: path.join(output, 'mobile.png'), fullPage: true });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
  const cameraActivated = await page.locator('#video').evaluate(video => video.srcObject !== null);
  if (errors.length || overflow || cameraActivated) throw Error(JSON.stringify({ errors, overflow, cameraActivated }));
  const report = { url: page.url(), title: await page.title(), secureContext: true,
    metadata, observations, downloads, errors, overflow, cameraActivated,
    viewports: ['1440x1050', '390x844'], scope: 'Saved validation image only, no live camera or flight qualification' };
  fs.writeFileSync(path.join(output, 'report.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ output, errors, overflow, cameraActivated }));
} finally {
  await browser.close();
}
