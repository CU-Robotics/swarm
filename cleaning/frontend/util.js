export class ArmorPlate {
    constructor(x = 0, y = 0, w = 0, h = 0) {
        self.x = x;
        self.y = y;
        self.w = w;
        self.h = h;
    }

    update(x, y) {
        let ox = this.x;
        let oy = this.y;

        [ox, x] = [ox, x].sort((a, b) => a - b);
        [oy, y] = [oy, y].sort((a, b) => a - b);

        this.x = ox;
        this.y = oy;
        this.w = x - ox;
        this.y = y - oy;
    }

    area() {
        return self.w * self.h;
    }

    contains(x, y) {
        return (x >= this.x && x <= this.x + this.w && 
                y >= this.y && y <= this.y + this.h);
    }

    draw(ctx, fillStyle, strokeStyle, lineWidth) {
        ctx.lineWidth = lineWidth;
        ctx.strokeStyle = strokeStyle;
        ctx.fillStyle = fillStyle;

        ctx.fillRect(this.x, this.y, this.w, this.h);
        ctx.strokeRect(this.x, this.y, this.w, this.h);
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

    pick(x, y) {
        for (const plate of this.list) {
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

    commitPlate() {
        this.plates.push(this.pendingPlate);
        this.pendingPlate = null;
    }

    async commitAndUpdate(seek = 1) {
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
    }

    async update() {
        const [labels, blob] = await Promise.all([
            fetch("/image/labels").then(r => r.json()),
            fetch("/image/current").then(r => r.blob()),
        ]);

        // populate all plates
        this.plates.list.length = 0;
        labels.plates?.forEach(plate => {
            this.plates.list.push(plate); 
        });

        // load image
        await new Promise(resolve => {
            this.image.onload = resolve;
            this.image.src = URL.createObjectURL(blob);
        });
    }
}