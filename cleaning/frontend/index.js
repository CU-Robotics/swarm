import * as util from "/frontend/util.js"

const canvas = document.getElementById('canvas')
const ctx = canvas.getContext('2d')

let image = new Image();
let plates = [];
let plateStart = null;
let plateCurrent = null;
let settings = {
    scale: 0.5,
    minScale: 0.5,
    offset: [0, 0],
    prev: [0, 0],
    dragging: false,
    drawingBox: false,
};
let selectedBox = null;
let boxes = [];
const keys = {};



document.addEventListener('keydown', async e => {
    keys[e.key] = true
    switch (e.key) {
        case 'Backspace':
        case ' ':
            const jump = e.key == ' ' ? +1 : -1;
            const body = JSON.stringify({
                plates: boxes.map(box => box.map(coord => Math.round(coord))),
            });
            await fetch("/image/labels", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body,
            });
            await fetch(`/image/skip?jump=${jump}`, {
                method: "POST",
            });
            boxes.length = 0;
            await getCurrentImageAndLabels();
            break;
        case 'Escape':
            if (settings.drawingBox) {
                settings.drawingBox = false;
                plateStart = plateCurrent = null;
                requestAnimationFrame(draw);
            }
            break;
        case 't':
            boxes = boxes.filter(b => b !== selectedBox);
            selectedBox = null;
            requestAnimationFrame(draw);
    }
});
document.addEventListener('keyup', e => {
    keys[e.key] = false
    switch (e.key) {
        case 'p':
            settings.dragging = false;
            break;
    }
});

function clampOffset() {
    const width = image.width * settings.scale;
    const height = image.height * settings.scale;

    if (width <= canvas.width) {
        settings.offset[0] = (canvas.width - width) / 2;
    } else {
        settings.offset[0] = Math.min(0, Math.max(settings.offset[0], canvas.width - width));
    }

    if (height <= canvas.height) {
        settings.offset[1] = (canvas.height - height) / 2;
    } else {
        settings.offset[1] = Math.min(0, Math.max(settings.offset[1], canvas.height - height));
    }
}

image.onload = () => {
    canvas.width = image.width * 0.5;
    canvas.height = image.height * 0.5;     
    settings.offset = [0, 0];
    settings.scale = settings.minScale;

    requestAnimationFrame(draw);
};

canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const x = e.offsetX;
    const y = e.offsetY;
    const oldScale = settings.scale;
    const zoomFactor = 1.1;

    settings.scale *= e.deltaY < 0 ? zoomFactor : 1/zoomFactor;
    settings.scale = Math.max(settings.minScale, settings.scale);

    const [offX, offY] = settings.offset;
    settings.offset[0] -= (x - offX) / oldScale * (settings.scale - oldScale);
    settings.offset[1] -= (y - offY) / oldScale * (settings.scale - oldScale);

    clampOffset();
    requestAnimationFrame(draw);
});
canvas.addEventListener('mousedown', e => {
    settings.dragging = keys?.p; 
    settings.prev[0] = e.clientX;
    settings.prev[1] = e.clientY;

    if (!keys?.p) {
        settings.drawingBox = true;
        const {x, y} = imageCoords(e.offsetX, e.offsetY);
        plateStart = [x, y];
    }

    const {x, y} = imageCoords(e.offsetX, e.offsetY);
    for (let i = boxes.length - 1; i >= 0; i--) {
        const [bx, by, w, h] = boxes[i];
        if (w <= 0 || h <= 0) console.log(boxes[i]);
        if (x >= bx && x <= bx + w && y >= by && y <= by + h) {
            selectedBox = boxes[i];
            requestAnimationFrame(draw);
            break;
        }
    }
});
canvas.addEventListener('mouseup', () => { 
    settings.dragging = false 
    if (settings.drawingBox) {
        if (plateCurrent) {
            const [x, y, w, h] = plateCurrent;
            console.log(w*h);
            if (w*h > 100) {
                boxes.push(plateCurrent);
            }
        }
        settings.drawingBox = false;
        plateCurrent = plateStart = null;
        requestAnimationFrame(draw);
    }
});
canvas.addEventListener('mouseleave', () => { 
    settings.dragging = false;
    if (settings.drawingBox) {
        settings.drawingBox = false;
        plateCurrent = plateStart = null;
        requestAnimationFrame(draw);
    } 
});
canvas.addEventListener('mousemove', e => {
    if (settings.dragging) {
        const dx = e.clientX - settings.prev[0];
        const dy = e.clientY - settings.prev[1];

        settings.offset[0] += dx;
        settings.offset[1] += dy;
        settings.prev[0] = e.clientX;
        settings.prev[1] = e.clientY;

        clampOffset();
        requestAnimationFrame(draw);
    } else if (settings.drawingBox) {
        let {x, y} = imageCoords(e.offsetX, e.offsetY);
        let [ox, oy] = plateStart;
        [ox, x] = [ox, x].sort((a,b)=>a-b);
        [oy, y] = [oy, y].sort((a,b)=>a-b);

        plateCurrent = [ox, oy, x-ox, y-oy];
        requestAnimationFrame(draw);
    }
});

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(settings.offset[0], settings.offset[1]);
    ctx.scale(settings.scale, settings.scale);
    ctx.drawImage(image, 0, 0);

    let lazy = null;
    boxes.forEach((box, i) => {
        if (box == selectedBox) {
            lazy = () => {
                ctx.lineWidth = 2 / settings.scale;
                ctx.fillStyle = "rgba(255, 140, 0, 0.2)";
                ctx.strokeStyle = "darkorange";
                const [x, y, w, h] = box;
                ctx.fillRect(x, y, w, h);
                ctx.strokeRect(x, y, w, h);
            }
            return;
        } else {
            ctx.lineWidth = 2 / settings.scale;
            ctx.fillStyle = "rgba(0, 0, 255, 0.2)";
            ctx.strokeStyle = "blue";
        }
        const [x, y, w, h] = box;
        ctx.fillRect(x, y, w, h);
        ctx.strokeRect(x, y, w, h);
    });
    lazy?.();
    
    if (plateCurrent) {
        ctx.strokeStyle = 'green';
        ctx.lineWidth = 2 / settings.scale;
        const [x, y, w, h] = plateCurrent;
        ctx.strokeRect(x, y, w, h);
    }

    ctx.restore();
}

function imageCoords(x, y) {
    return {
        x: (x - settings.offset[0]) / settings.scale,
        y: (y - settings.offset[1]) / settings.scale,
    };
}

async function getCurrentImageAndLabels() {
    let [labels, blob] = await Promise.all([
        fetch("/image/labels").then(r => r.json()),
        fetch("/image/current").then(r => r.blob()),
    ]);

    // populate plates
    boxes.length = 0;
    labels.plates?.forEach(plate => {
        boxes.push(plate);
    });

    // load the image
    await new Promise(resolve => {
        image.onload = resolve;
        image.src = URL.createObjectURL(blob);
    });

    canvas.width = image.width * 0.5;
    canvas.height = image.height * 0.5;     
    settings.offset = [0, 0];
    settings.scale = settings.minScale;

    requestAnimationFrame(draw);
}

async function main() {
    await getCurrentImageAndLabels();
}

main();