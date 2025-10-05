export class ArmorPlate {
    static ICON = null;
    static COLOR = null;

    constructor(x = 0, y = 0, w = 0, h = 0, icon = ArmorPlate.ICON, color = ArmorPlate.COLOR) {
        this.x = this.anchorX = x;
        this.y = this.anchorY = y;
        this.w = w;
        this.h = h;
        this.icon = icon;
        this.color = color;
    }

    update(freeX, freeY) {
        const [x1, x2] = [this.anchorX, freeX].sort((a, b) => a - b);
        const [y1, y2] = [this.anchorY, freeY].sort((a, b) => a - b);

        this.x = x1;
        this.y = y1;
        this.w = x2 - x1;
        this.h = y2 - y1;
    }

    area() {
        return this.w * this.h;
    }

    contains(x, y) {
        return (x >= this.x && x <= this.x + this.w && 
                y >= this.y && y <= this.y + this.h);
    }

    draw(ctx, view, fillStyle, strokeStyle, lineWidth, drawText = true, textMargin = 5) {
        ctx.lineWidth = lineWidth;
        ctx.strokeStyle = strokeStyle;

        if (fillStyle) {
            ctx.fillStyle = fillStyle;
            ctx.fillRect(this.x, this.y, this.w, this.h);
        }
        
        ctx.strokeRect(this.x, this.y, this.w, this.h);

        if (drawText) {
            ctx.save();
            ctx.setTransform(1, 0, 0, 1, 0, 0);

            const x = this.x * view.scale + view.offset.x;
            const y = this.y * view.scale + view.offset.y;
            const label = `${this.color} ${this.icon}`;
            const fontSize = 12;

            ctx.fillStyle = "white";
            ctx.font = `${fontSize}px monospace`;
            ctx.fillText(label, textMargin + x, textMargin + fontSize + y);
            ctx.restore();
        }
    }

    *[Symbol.iterator]() {
        yield this.x;
        yield this.y;
        yield this.w;
        yield this.h;
    }
}

export class ArmorPlates {
    constructor() {
        this.list = [];
        this.focusedPlate = null;
    }

    discardFocusedPlate() {
        const idx = this.list.indexOf(this.focusedPlate);
        if (idx >= 0) {
            this.list.splice(idx, 1);
        }
    }

    pick(x, y) {
        for (let i = this.list.length - 1; i >= 0; i--) {
            const plate = this.list[i];

            if (plate.contains(x, y)) {
                this.focusedPlate = plate;
                return;
            }
        }
    }

    // bubble focused plate to the top
    *[Symbol.iterator]() {
        let lazyPlate = null;
        for (const plate of this.list) {
            if (plate !== this.focusedPlate || lazyPlate) {
                yield plate;
            } else if (!lazyPlate) {
                lazyPlate = plate;
            }
        }

        if (lazyPlate) {
            yield lazyPlate;
        }
    }
}

export class Labeler {
    constructor(canvas) {
        this.canvas = canvas;
        this.image = new Image();
        this.plates = new ArmorPlates();
        this.commitQueue = Promise.resolve();
        this.pendingPlate = null;
    }

    enqueuePlate(x, y) {
        this.pendingPlate = new ArmorPlate(x, y);
    }

    discardPlate() {
        this.pendingPlate = null;
    }

    commitPendingPlate() {
        if (this.pendingPlate) {
            this.plates.list.push(this.pendingPlate);
            this.plates.focusedPlate = this.pendingPlate;
            this.pendingPlate = null;
        }
    }

    async commitChanges(seek = 1) {
        this.commitQueue = this.commitQueue.then(async () => {
            // save current labels
            const body = JSON.stringify({
                // serialize plates
                plates: this.plates.list.map(plate => [...plate]),
            });

            await fetch("/image/labels", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body,
            });

            // get next image
            await fetch(`/image/skip?jump=${seek}`, {
                method: "POST",
            });

            await this.update();
        });

        return this.commitQueue;
    }

    async update() {
        const [labels, blob] = await Promise.all([
            fetch("/image/labels").then(r => r.json()),
            fetch("/image/current").then(r => r.blob()),
        ]);

        ArmorPlate.ICON = labels.icon ?? "none";
        ArmorPlate.COLOR = labels.color ?? "none";

        // populate all plates
        this.plates.list.length = 0;
        labels.plates?.forEach(plate => {
            this.plates.list.push(new ArmorPlate(...plate)); 
        });

        // load image
        await new Promise(resolve => {
            this.image.onload = resolve;
            this.image.src = URL.createObjectURL(blob);
        });
    }

    *[Symbol.iterator]() {
        yield* this.plates;
        
        if (this.pendingPlate) {
            yield this.pendingPlate;
        }
    }
}

export function clamp(v, lo, hi) {
    return Math.min(hi, Math.max(v, lo));
}

export function zip(...arrs) {
    const len = Math.min(...arrs.map(arr => arr.length));
    return Array.from({ length: len }, (_, i) => arrs.map(arr => arr[i]));
}