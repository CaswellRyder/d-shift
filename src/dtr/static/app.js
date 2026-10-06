"use strict";
const $ = (id) => document.getElementById(id);
const display = $("frame"), view = display.getContext("2d");
const capture = document.createElement("canvas");
capture.width = 320; capture.height = 240;
const input = capture.getContext("2d", { willReadFrequently: true });
const policy = globalThis.DTRFramePolicy;
const sourceCapture = document.createElement("canvas");
const sourceInput = sourceCapture.getContext("2d");
let config, running = false, generation = 0, timer, request, stream, session, sequence = 0;
let fpsCap = 5, delivered = [];
let stillImage;
let downloadable;

function downloadImage(kind) {
  if (!downloadable) return;
  try {
    // Synchronous capture: filenames, overlay and mask refer to this rendered result,
    // even when the next live inference request is in flight.
    const data = kind === "annotated" ? display.toDataURL("image/png") : downloadable.mask;
    if (!data.startsWith("data:image/png;base64,")) throw new Error("PNG image is unavailable.");
    const filename = `${downloadable.basename}-${kind}.png`;
    const link = document.createElement("a");
    link.href = data; link.download = filename;
    document.body.append(link); link.click(); link.remove();
    $("download-status").textContent = `Download requested: ${filename}. Check your browser's downloads.`;
  } catch (exc) {
    $("download-status").textContent = `Could not download image: ${exc.message}`;
  }
}

function message(text) { $("status").textContent = text; }
function error(text) { $("error").textContent = text; $("error").hidden = !text; }
function warning() {
  const meta = config.tasks[$("task").value];
  $("pipeline").textContent = `Camera → color proposals → ${meta.engine === "keras_teacher" ? "V4 teacher" : "INT8 student"} → simple IoU tracks`;
  $("warning").textContent = (meta.synthetic ? "Synthetic test models. Not flight-ready." : "Research model. Not flight-qualified.")
    + ` Acceptance threshold: ${Math.round(meta.threshold * 100)}%.`
    + (meta.proposal_limit > 12 ? ` ${meta.proposal_limit}-candidate desktop experiment; not Pi-qualified.` : "");
}
function emptyRows(text) {
  const row = document.createElement("tr"), cell = document.createElement("td");
  cell.colSpan = 7; cell.className = "no-results"; cell.textContent = text; row.append(cell);
  $("rows").replaceChildren(row);
}
function stop(text = "Stopped. Camera released.") {
  running = false; generation++;
  downloadable = undefined;
  $("download-frame").disabled = true; $("download-mask").disabled = true;
  $("download-status").textContent = "Downloads save the displayed result as PNG files on your device.";
  clearTimeout(timer); request?.abort(); request = undefined;
  stream?.getTracks().forEach((t) => t.stop()); stream = undefined;
  $("video").srcObject = null;
  $("start").disabled = !config; $("demo").disabled = !config; $("stop").disabled = true;
  $("task").disabled = !config; $("camera").disabled = false;
  $("fps").disabled = !config; delivered = [];
  $("image-file").disabled = !config; $("analyze").disabled = !config || !stillImage;
  $("empty").hidden = false; $("mask").hidden = true; $("mask").removeAttribute("src");
  $("mask-empty").hidden = false;
  view.clearRect(0, 0, display.width, display.height);
  for (const id of ["source", "source-size", "capped-size", "camera-fps", "fps-cap", "actual-fps", "processing", "roundtrip", "candidates", "tracks", "accepted", "tentative"]) $(id).textContent = "—";
  emptyRows("No observations yet."); message(text);
}
async function cameras() {
  const selected = stream?.getVideoTracks()[0]?.getSettings().deviceId || $("camera").value;
  const devices = await navigator.mediaDevices.enumerateDevices();
  $("camera").replaceChildren(new Option("Default camera", ""));
  devices.filter((d) => d.kind === "videoinput").forEach((d, i) =>
    $("camera").add(new Option(d.label || `Camera ${i + 1}`, d.deviceId)));
  $("camera").value = selected;
}
function demoFrame(frame) {
  input.fillStyle = "#272d34"; input.fillRect(0, 0, 320, 240);
  const x = 94 + Math.sin(frame / 18) * 45;
  input.fillStyle = "#34c34b"; input.beginPath(); input.ellipse(x, 119, 25, 34, 0, 0, 2 * Math.PI); input.fill();
  input.fillStyle = "#a336c5"; input.beginPath(); input.ellipse(228, 90 + Math.sin(frame / 23) * 28, 23, 30, 0, 0, 2 * Math.PI); input.fill();
  if ($("task").value === "goal") {
    input.fillStyle = "#272d34"; input.fillRect(0, 0, 320, 240);
    input.lineWidth = 9; input.strokeStyle = "#ffc719"; input.strokeRect(x - 28, 85, 60, 65);
    input.strokeStyle = "#ff820f"; input.beginPath(); input.arc(226, 125, 30, 0, 2 * Math.PI); input.stroke();
  }
  input.fillStyle = "#fff"; input.font = "11px sans-serif"; input.fillText("SYNTHETIC DEMO — NOT CAMERA", 10, 228);
}
function prepareSource(image, width, height) {
  const size = policy.sourceSize(width, height);
  if (sourceCapture.width !== size.width || sourceCapture.height !== size.height) {
    sourceCapture.width = size.width; sourceCapture.height = size.height;
  }
  sourceInput.clearRect(0, 0, size.width, size.height);
  sourceInput.drawImage(image, 0, 0, size.width, size.height);
  const fit = policy.letterbox(size.width, size.height);
  // Black bars keep non-4:3 sources from distorting circles into ellipses.
  input.fillStyle = "#000"; input.fillRect(0, 0, 320, 240);
  input.drawImage(sourceCapture, fit.x, fit.y, fit.width, fit.height);
  return { width, height, capped: size };
}
function draw(image, observations) {
  view.drawImage(image, 0, 0, display.width, display.height);
  const scale = display.width / 320;
  view.save(); view.scale(scale, scale); view.lineWidth = 1;
  // Accepted targets draw last so rejected/suppressed IDs cannot cover their labels.
  for (const row of [...observations].sort((a, b) => DTRGoalView.describe(a).priority - DTRGoalView.describe(b).priority)) {
    const info = DTRGoalView.describe(row);
    const [x1, y1, x2, y2] = row.box;
    view.strokeStyle = "#ffa000"; view.setLineDash([3, 2]);
    view.strokeRect(x1, y1, x2 - x1, y2 - y1); view.setLineDash([]);
    view.strokeStyle = info.stroke;
    view.strokeRect(x1 + 2, y1 + 2, Math.max(1, x2 - x1 - 4), Math.max(1, y2 - y1 - 4));
    view.strokeStyle = "#398eff"; view.beginPath();
    row.trail.forEach(([x, y], i) => i ? view.lineTo(x, y) : view.moveTo(x, y)); view.stroke();
    const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
    view.beginPath(); view.moveTo(cx - 3, cy); view.lineTo(cx + 3, cy); view.moveTo(cx, cy - 3); view.lineTo(cx, cy + 3); view.stroke();
    const label = info.overlay;
    view.font = "9px sans-serif";
    const width = view.measureText(label).width + 6;
    const x = Math.min(x1, 320 - width), y = Math.max(12, y1);
    view.fillStyle = "#18202be8"; view.fillRect(x, y - 12, width, 12);
    view.fillStyle = "#fff"; view.fillText(label, x + 3, y - 3);
  }
  view.restore();
}
function showRows(observations) {
  if (!observations.length) { emptyRows("No matching color candidates in this frame."); return; }
  $("rows").replaceChildren(...observations.map((row) => {
    const tr = document.createElement("tr");
    const info = DTRGoalView.describe(row);
    for (const value of [`#${row.track_id}`, row.label, `${(row.score * 100).toFixed(1)}%`, info.shape, info.color, info.status, row.box.join(", ")]) {
      const td = document.createElement("td"); td.textContent = value; tr.append(td);
    }
    return tr;
  }));
}
async function tick(mode, gen, reset = false) {
  if (!running || generation !== gen) return;
  const start = performance.now();
  try {
    let dimensions;
    if (mode === "demo") {
      demoFrame(sequence);
      dimensions = { width: 320, height: 240, capped: { width: 320, height: 240 } };
    }
    else if (mode === "image") dimensions = prepareSource(stillImage, stillImage.width, stillImage.height);
    else {
      if (!$("video").videoWidth) throw new Error("Camera has not produced a frame.");
      dimensions = prepareSource($("video"), $("video").videoWidth, $("video").videoHeight);
    }
    const blob = await new Promise((resolve) => capture.toBlob(resolve, "image/jpeg", 0.88));
    if (!running || generation !== gen) return;
    if (!blob) throw new Error("Could not encode frame.");
    const controller = new AbortController();
    request = controller;
    const timeout = setTimeout(() => controller.abort(), 10000);
    let response;
    try {
      response = await fetch(`/api/frame?task=${$("task").value}&session=${session}&sequence=${sequence++}&reset=${reset ? 1 : 0}`, {
        method: "POST", headers: {"Content-Type": "image/jpeg", "X-DTR-Token": config.token},
        body: blob, signal: controller.signal,
      });
    } finally { clearTimeout(timeout); }
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Inference failed.");
    if (!running || generation !== gen) return;
    const image = await createImageBitmap(blob);
    if (!running || generation !== gen) { image.close(); return; }
    draw(image, result.observations); image.close();
    $("empty").hidden = true;
    $("mask").src = result.mask; $("mask").hidden = false; $("mask-empty").hidden = true;
    downloadable = {
      basename: `dtr-${$("task").value}-${mode}-${new Date().toISOString().replace(/[:.]/g, "-")}-frame${sequence - 1}`,
      mask: result.mask,
    };
    $("download-frame").disabled = false; $("download-mask").disabled = false;
    $("source").textContent = mode === "demo" ? "Synthetic demo" : mode === "image" ? "Local test image" : "Live webcam";
    $("source-size").textContent = `${dimensions.width} × ${dimensions.height}`;
    $("capped-size").textContent = `${dimensions.capped.width} × ${dimensions.capped.height}`;
    const cameraRate = stream?.getVideoTracks()[0]?.getSettings().frameRate;
    $("camera-fps").textContent = mode === "camera" && Number.isFinite(cameraRate) ? cameraRate.toFixed(1) : "N/A";
    $("fps-cap").textContent = mode === "image" ? "N/A — single image" : `${fpsCap} FPS`;
    delivered.push(performance.now());
    if (delivered.length > 20) delivered.shift();
    $("actual-fps").textContent = mode === "image" ? "N/A — single image" : delivered.length < 2 ? "Measuring…" :
      `${(1000 * (delivered.length - 1) / (delivered.at(-1) - delivered[0])).toFixed(2)} FPS`;
    $("processing").textContent = `${result.processing_ms.toFixed(1)} ms`;
    $("roundtrip").textContent = `${(performance.now() - start).toFixed(1)} ms`;
    $("candidates").textContent = result.observations.length;
    $("accepted").textContent = `${result.accepted_count} / ${result.suppressed_count}`;
    $("tentative").textContent = result.observations.filter(r => DTRGoalView.describe(r).tentative).length;
    $("tracks").textContent = new Set(result.observations.map((r) => r.track_id)).size;
    showRows(result.observations);
    const budget = config.tasks[$("task").value].proposal_limit ?? config.proposal_limit;
    message(`320 × 240 processing • up to ${budget} candidates${budget > 12 ? " • higher-compute desktop experiment; not Pi-qualified" : ""} • boxes match the displayed frame. ${mode === "image" ? "Image analyzed once. No camera access." : `At most ${fpsCap} updates/s; one request at a time, no frame queue. Actual speed depends on processing.`}`);
    if (mode === "image") {
      running = false;
      $("start").disabled = false; $("demo").disabled = false; $("task").disabled = false;
      $("camera").disabled = false; $("image-file").disabled = false; $("analyze").disabled = false;
      $("fps").disabled = false;
      return;
    }
    timer = setTimeout(() => tick(mode, gen), policy.delay(fpsCap, performance.now() - start));
  } catch (exc) {
    if (generation !== gen) return;
    stop("Stopped after an error.");
    error(exc.name === "AbortError" ? "Inference timed out. Check the Python server and restart." : exc.message);
  }
}
async function start(mode) {
  stop(); error("");
  const gen = generation;
  $("start").disabled = true; $("demo").disabled = true; $("stop").disabled = false;
  $("task").disabled = true; $("camera").disabled = true;
  $("fps").disabled = true;
  $("image-file").disabled = true; $("analyze").disabled = true;
  session ??= crypto.randomUUID();
  try {
    fpsCap = policy.fps($("fps").value);
    if (mode === "camera") {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Use http://127.0.0.1:8765 in a browser on this MacBook.");
      message("Waiting for camera permission…");
      const device = $("camera").value;
      const acquired = await navigator.mediaDevices.getUserMedia({audio: false, video: {
        width: {ideal: policy.SOURCE_WIDTH}, height: {ideal: policy.SOURCE_HEIGHT},
        frameRate: {ideal: fpsCap}, resizeMode: "none",
        ...(device ? {deviceId: {exact: device}} : {}),
      }});
      if (generation !== gen) { acquired.getTracks().forEach((t) => t.stop()); return; }
      stream = acquired; $("video").srcObject = stream;
      stream.getVideoTracks()[0].addEventListener("ended", () => { if (generation === gen) stop("Camera disconnected."); });
      await $("video").play();
      if (generation !== gen) return;
      await cameras();
    }
    if (generation !== gen) return;
    running = true; tick(mode, gen, true);
  } catch (exc) {
    if (generation !== gen) return;
    stop("Could not start.");
    error(`${exc.name}: ${exc.message}` + (mode === "camera" && exc.name === "NotAllowedError" ?
      " Allow camera access for this browser in macOS Privacy & Security → Camera, then retry. Demo works without a camera." : ""));
  }
}
$("start").addEventListener("click", () => start("camera"));
$("demo").addEventListener("click", () => start("demo"));
$("stop").addEventListener("click", () => stop());
$("analyze").addEventListener("click", () => start("image"));
$("download-frame").addEventListener("click", () => downloadImage("annotated"));
$("download-mask").addEventListener("click", () => downloadImage("mask"));
$("image-file").addEventListener("change", async () => {
  const file = $("image-file").files[0];
  if (!file) return;
  stop(); error("");
  const gen = generation;
  $("image-file").value = "";
  try {
    if (file.size > 15 * 1024 * 1024) throw new Error("Choose an image smaller than 15 MB.");
    const decoded = await createImageBitmap(file);
    if (gen !== generation) { decoded.close(); return; }
    stillImage?.close(); stillImage = decoded;
    start("image");
  } catch (exc) { error(`Could not load image: ${exc.message}`); }
});
$("task").addEventListener("change", () => { stop("Task changed. Start camera or demo again."); warning(); });
window.addEventListener("pagehide", () => stop());
document.addEventListener("visibilitychange", () => { if (document.hidden) stop("Paused while tab is hidden. Start again to resume."); });
fetch("/api/config").then(async (response) => {
  if (!response.ok) throw new Error("Could not load model metadata.");
  config = await response.json();
  for (const task of Object.keys(config.tasks)) $("task").add(new Option(task === "balloon" ? "Balloon" : "Goal", task));
  warning(); stop("Camera is off.");
}).catch((exc) => error(exc.message));
