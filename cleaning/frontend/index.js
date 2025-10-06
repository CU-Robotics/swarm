import { Labeler, clamp, zip } from '/frontend/util.js'

const MIN_SCALE = 0.45;
const MIN_PLATE_AREA = 100;

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const labeler = new Labeler();

// saving
['save', 'kill'].forEach(endpoint => {
    document.getElementById(endpoint).addEventListener('click', async ev => {
        ev.preventDefault();
        await labeler.commitChanges(0);
        await fetch(`/${endpoint}`, { method: "POST" });

        if (endpoint == 'kill') {
            location.reload();
        }
    });
});

// setup
const keys = {};
const view = {
    offset: {x: 0, y: 0},
    prev: {x: 0, y: 0},
    scale: MIN_SCALE,
    panning: false,
};

// load initial image
await updateImage();
requestAnimationFrame(globalRedraw);

// input
document.addEventListener('keydown', async ev => {
    keys[ev.key] = true;
    switch (ev.key) {
        // seek image
        case ' ':
        case 'Backspace':
            const seek = ev.key == ' ' ? +1 : -1;
            await labeler.commitChanges(seek);
            await updateImage();
            break;
        case 'Escape':
            labeler.discardPlate();
            break;
        case 't':
            labeler.plates.discardFocusedPlate();
            break;
    }

    requestAnimationFrame(globalRedraw);
});

document.addEventListener('keyup', ev => {
    keys[ev.key] = false
    switch (ev.key) {
        case 'd':    
            if (labeler.pendingPlate && 
                labeler.pendingPlate.area() > MIN_PLATE_AREA && 
                labeler.pendingPlate.w > Math.sqrt(MIN_PLATE_AREA) && 
                labeler.pendingPlate.h > Math.sqrt(MIN_PLATE_AREA)) {
                labeler.commitPendingPlate();
            } else {
                labeler.discardPlate();
            }
            requestAnimationFrame(globalRedraw);
            break;
    }
});

document.addEventListener('wheel', ev => {
    ev.preventDefault();

    const x = ev.offsetX;
    const y = ev.offsetY;
    const oldScale = view.scale;
    const zoomFactor = 1.1;

    view.scale *= ev.deltaY < 0 ? zoomFactor : zoomFactor**-1;
    view.scale = Math.max(MIN_SCALE, view.scale);

    const {x: vx, y: vy} = view.offset;
    view.offset.x -= (x - vx) / oldScale * (view.scale - oldScale);
    view.offset.y -= (y - vy) / oldScale * (view.scale - oldScale);

    clampOffsets();
    requestAnimationFrame(globalRedraw);
});

canvas.addEventListener('mousedown', ev => {
    view.panning = true;

    view.prev.x = ev.offsetX;
    view.prev.y = ev.offsetY;

    const {x, y} = rawToImage(ev.offsetX, ev.offsetY);
    labeler.plates.pick(x, y);

    if (keys.d) {
        labeler.enqueuePlate(x, y);
    }

    requestAnimationFrame(globalRedraw);
});

document.addEventListener('mouseup', ev => {
    view.panning = false;

    if (ev.target == canvas && labeler.pendingPlate) {
        if (labeler.pendingPlate.area() > MIN_PLATE_AREA && 
            labeler.pendingPlate.w > Math.sqrt(MIN_PLATE_AREA) && 
            labeler.pendingPlate.h > Math.sqrt(MIN_PLATE_AREA)) {
            labeler.commitPendingPlate();
        } else {
            labeler.discardPlate();
        }
    }

    requestAnimationFrame(globalRedraw);
});

canvas.addEventListener('mousemove', ev => {
    if (view.panning) {
        if (!keys.d) {
            const dx = ev.offsetX - view.prev.x;
            const dy = ev.offsetY - view.prev.y;

            view.offset.x += dx;
            view.offset.y += dy;
            view.prev.x = ev.offsetX;
            view.prev.y = ev.offsetY;

            clampOffsets();
        } else {
            const {x, y} = rawToImage(ev.offsetX, ev.offsetY);
            labeler.pendingPlate?.update(x, y);
        }

        requestAnimationFrame(globalRedraw);
    }
});

function rawToImage(x, y) {
    return {
        x: (x - view.offset.x) / view.scale,
        y: (y - view.offset.y) / view.scale,
    };
}

function clampOffsets() {
    const [iw, ih] = [labeler.image.width * view.scale, labeler.image.height * view.scale];

    // apply clamps to offset values
    for (const [i, im_size, canvas_size] of zip("xy", [iw, ih], [canvas.width, canvas.height])) {
        if (im_size <= canvas_size) {
            view.offset[i] = (canvas_size - im_size) / 2;
        } else {
            view.offset[i] = clamp(view.offset[i], canvas_size - im_size, 0);
        }
    }
}

async function updateImage() {
    await labeler.update();

    canvas.width = labeler.image.width * MIN_SCALE;
    canvas.height = labeler.image.height * MIN_SCALE;     
    view.offset.x = view.offset.y = 0;
    view.scale = MIN_SCALE;
}

function globalRedraw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(view.offset.x, view.offset.y);
    ctx.scale(view.scale, view.scale);
    ctx.drawImage(labeler.image, 0, 0);

    for (const plate of labeler) {
        let border = null, fill = null, text = true;
        if (plate == labeler.pendingPlate) {
            border = "orange";
            text = false;
        } else if (plate == labeler.plates.focusedPlate) {
            border = "yellow";
            fill = "rgba(255, 255, 0,  0.2)";
        } else {
            border = "cyan";
            fill = "rgba(0, 255, 255, 0.2)";
        }

        plate.draw(ctx, view, fill, border, 2 * view.scale**-1, text);
    }

    ctx.restore();
}