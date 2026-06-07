type PhysMotionAttribute = 'position' | 'rotation' | 'scale';

type InitMessage = {
    type: 'init';
    binaryUrl: string;
    frameCount: number;
    availableFrames: number;
    numSplats: number;
    attributes: PhysMotionAttribute[];
    frameStrideBytes: number;
    chunkFrameCount: number;
    textureWidth: number;
    textureHeight: number;
    indices?: ArrayBuffer;
    baseTransformA: ArrayBuffer;
    baseTransformB: ArrayBuffer;
};

type PrepareMessage = {
    type: 'prepare';
    id: number;
    frame: number;
};

type ExtendMessage = {
    type: 'extendAvailableFrames';
    availableFrames: number;
};

type WorkerMessage = InitMessage | PrepareMessage | ExtendMessage | { type: 'destroy' };

type RowRange = {
    y: number;
    height: number;
};

type PreparedFrameStats = {
    frame: number;
    trackedCount: number;
    touchedRowCount: number;
    textureHeight: number;
    rowRangeCount: number;
    transformABytes: number;
    transformBBytes: number;
    centersBytes: number;
    totalBytes: number;
    decodeMs: number;
    packMs: number;
    touchedRowRatio: number;
};

type WorkerState = {
    binaryUrl: string;
    frameCount: number;
    availableFrames: number;
    numSplats: number;
    attributes: PhysMotionAttribute[];
    frameStrideBytes: number;
    chunkFrameCount: number;
    textureWidth: number;
    textureHeight: number;
    indices?: Uint32Array;
    baseTransformA: Uint32Array;
    baseTransformB: Uint16Array;
    trackedCount: number;
    positionOffset: number;
    rotationOffset: number;
    scaleOffset: number;
    floatsPerFrame: number;
    rowRanges: RowRange[];
    splatRangeIndex: Int32Array;
    splatLocalTexel: Int32Array;
    touchedRowCount: number;
};

let state: WorkerState | null = null;
const workerScope = self as unknown as DedicatedWorkerGlobalScope;
const rawChunkCache = new Map<number, Promise<ArrayBuffer>>();
const halfFloatView = new Float32Array(1);
const halfIntView = new Uint32Array(halfFloatView.buffer);

const readUrlRange = async (url: string, start: number, endExclusive: number) => {
    const expected = endExclusive - start;
    const response = await fetch(url, {
        headers: {
            Range: `bytes=${start}-${endExclusive - 1}`
        }
    });
    if (!response.ok && response.status !== 206) {
        throw new Error(`failed to read motion range ${start}-${endExclusive - 1}: ${response.status} ${response.statusText}`);
    }
    const buffer = await response.arrayBuffer();
    if (response.status === 206) {
        if (buffer.byteLength !== expected) {
            throw new Error(`range size mismatch: expected ${expected} bytes, got ${buffer.byteLength}`);
        }
        return buffer;
    }
    if (response.status === 200 && buffer.byteLength === expected) {
        return buffer;
    }
    throw new Error(`server ignored Range request: expected ${expected} bytes, got ${buffer.byteLength}; refusing to load full motion file`);
};

const floatToHalf = (value: number) => {
    if (!Number.isFinite(value)) {
        return value < 0 ? 0xfc00 : 0x7c00;
    }
    halfFloatView[0] = value;
    const x = halfIntView[0];
    const sign = (x >>> 16) & 0x8000;
    let mantissa = x & 0x7fffff;
    let exponent = (x >>> 23) & 0xff;
    if (exponent === 0xff) {
        return sign | (mantissa ? 0x7e00 : 0x7c00);
    }
    exponent = exponent - 127 + 15;
    if (exponent >= 0x1f) {
        return sign | 0x7c00;
    }
    if (exponent <= 0) {
        if (exponent < -10) {
            return sign;
        }
        mantissa = (mantissa | 0x800000) >> (1 - exponent);
        return sign | ((mantissa + 0x1000) >> 13);
    }
    return sign | (exponent << 10) | ((mantissa + 0x1000) >> 13);
};

const rowRangesFromTouched = (touchedRows: Uint8Array): RowRange[] => {
    const ranges: RowRange[] = [];
    let row = 0;
    while (row < touchedRows.length) {
        while (row < touchedRows.length && !touchedRows[row]) {
            row++;
        }
        const start = row;
        while (row < touchedRows.length && touchedRows[row]) {
            row++;
        }
        if (row > start) {
            ranges.push({ y: start, height: row - start });
        }
    }
    return ranges;
};

const buildRowMapping = (
    indices: Uint32Array | undefined,
    trackedCount: number,
    numSplats: number,
    textureWidth: number,
    textureHeight: number
) => {
    const touchedRows = new Uint8Array(textureHeight);
    for (let i = 0; i < trackedCount; ++i) {
        const id = indices ? indices[i] : i;
        if (id < numSplats) {
            const row = Math.floor(id / textureWidth);
            if (row >= 0 && row < textureHeight) {
                touchedRows[row] = 1;
            }
        }
    }

    const rowRanges = rowRangesFromTouched(touchedRows);
    const rowToRange = new Int32Array(textureHeight);
    rowToRange.fill(-1);
    for (let rangeIndex = 0; rangeIndex < rowRanges.length; ++rangeIndex) {
        const range = rowRanges[rangeIndex];
        for (let row = range.y; row < range.y + range.height; ++row) {
            rowToRange[row] = rangeIndex;
        }
    }

    const splatRangeIndex = new Int32Array(trackedCount);
    const splatLocalTexel = new Int32Array(trackedCount);
    splatRangeIndex.fill(-1);
    splatLocalTexel.fill(-1);

    for (let i = 0; i < trackedCount; ++i) {
        const id = indices ? indices[i] : i;
        if (id >= numSplats) {
            continue;
        }
        const row = Math.floor(id / textureWidth);
        if (row < 0 || row >= textureHeight) {
            continue;
        }
        const rangeIndex = rowToRange[row];
        if (rangeIndex < 0) {
            continue;
        }
        const range = rowRanges[rangeIndex];
        splatRangeIndex[i] = rangeIndex;
        splatLocalTexel[i] = (row - range.y) * textureWidth * 4 + (id % textureWidth) * 4;
    }

    const touchedRowCount = touchedRows.reduce((sum, value) => sum + value, 0);
    return {
        rowRanges,
        splatRangeIndex,
        splatLocalTexel,
        touchedRowCount
    };
};

const ensureChunk = (frame: number) => {
    if (!state) {
        throw new Error('motion worker is not initialized');
    }
    const chunkIndex = Math.floor(frame / state.chunkFrameCount);
    const cached = rawChunkCache.get(chunkIndex);
    if (cached) {
        return cached;
    }

    const startFrame = chunkIndex * state.chunkFrameCount;
    const frameCount = Math.min(state.chunkFrameCount, state.frameCount - startFrame);
    const startByte = startFrame * state.frameStrideBytes;
    const endByte = (startFrame + frameCount) * state.frameStrideBytes;
    const promise = readUrlRange(state.binaryUrl, startByte, endByte).catch((error): never => {
        rawChunkCache.delete(chunkIndex);
        throw error;
    });
    rawChunkCache.set(chunkIndex, promise);
    while (rawChunkCache.size > 2) {
        const firstKey = rawChunkCache.keys().next().value;
        if (firstKey === chunkIndex) {
            break;
        }
        rawChunkCache.delete(firstKey);
    }
    return promise;
};

const frameFloatsFor = async (frame: number) => {
    if (!state) {
        throw new Error('motion worker is not initialized');
    }
    const chunk = await ensureChunk(frame);
    const chunkStartFrame = Math.floor(frame / state.chunkFrameCount) * state.chunkFrameCount;
    const byteOffset = (frame - chunkStartFrame) * state.frameStrideBytes;
    return new Float32Array(
        chunk,
        byteOffset,
        state.floatsPerFrame
    );
};

const prepareFrame = async (frame: number) => {
    if (!state) {
        throw new Error('motion worker is not initialized');
    }
    if (frame < 0 || frame >= state.availableFrames || frame >= state.frameCount) {
        return null;
    }

    const decodeStart = performance.now();
    const frameFloats = await frameFloatsFor(frame);
    const decodeEnd = performance.now();
    const { trackedCount, textureWidth, textureHeight, indices, rowRanges, splatRangeIndex, splatLocalTexel } = state;
    const transformARows = rowRanges.map((range) => {
        const start = range.y * textureWidth * 4;
        const end = (range.y + range.height) * textureWidth * 4;
        return {
            y: range.y,
            height: range.height,
            data: new Uint32Array(state!.baseTransformA.subarray(start, end))
        };
    });
    const transformBRows = rowRanges.map((range) => {
        const start = range.y * textureWidth * 4;
        const end = (range.y + range.height) * textureWidth * 4;
        return {
            y: range.y,
            height: range.height,
            data: new Uint16Array(state!.baseTransformB.subarray(start, end))
        };
    });
    const transformAFloats = transformARows.map(row => new Float32Array(row.data.buffer));
    const centers = state.positionOffset >= 0 ? new Float32Array(trackedCount * 3) : undefined;
    const bounds = centers && !indices ? {
        min: [Infinity, Infinity, Infinity] as [number, number, number],
        max: [-Infinity, -Infinity, -Infinity] as [number, number, number]
    } : undefined;

    const packStart = performance.now();
    for (let i = 0; i < trackedCount; ++i) {
        const rangeIndex = splatRangeIndex[i];
        const localTexel = splatLocalTexel[i];
        if (rangeIndex < 0 || localTexel < 0) {
            continue;
        }
        const transformAData = transformARows[rangeIndex].data;
        const transformAFloat = transformAFloats[rangeIndex];
        const transformBData = transformBRows[rangeIndex].data;

        if (state.positionOffset >= 0) {
            const src = state.positionOffset + i * 3;
            const px = frameFloats[src];
            const py = frameFloats[src + 1];
            const pz = frameFloats[src + 2];
            transformAFloat[localTexel] = px;
            transformAFloat[localTexel + 1] = py;
            transformAFloat[localTexel + 2] = pz;
            if (centers) {
                const dst = i * 3;
                centers[dst] = px;
                centers[dst + 1] = py;
                centers[dst + 2] = pz;
            }
            if (bounds) {
                bounds.min[0] = Math.min(bounds.min[0], px);
                bounds.min[1] = Math.min(bounds.min[1], py);
                bounds.min[2] = Math.min(bounds.min[2], pz);
                bounds.max[0] = Math.max(bounds.max[0], px);
                bounds.max[1] = Math.max(bounds.max[1], py);
                bounds.max[2] = Math.max(bounds.max[2], pz);
            }
        }

        if (state.rotationOffset >= 0) {
            const src = state.rotationOffset + i * 4;
            let w = frameFloats[src];
            let x = frameFloats[src + 1];
            let y = frameFloats[src + 2];
            let z = frameFloats[src + 3];
            const len = Math.hypot(x, y, z, w);
            if (len > 0) {
                x /= len;
                y /= len;
                z /= len;
                w /= len;
            }
            if (w < 0) {
                x = -x;
                y = -y;
                z = -z;
            }
            transformAData[localTexel + 3] = floatToHalf(x) | floatToHalf(y) << 16;
            transformBData[localTexel + 3] = floatToHalf(z);
        }

        if (state.scaleOffset >= 0) {
            const src = state.scaleOffset + i * 3;
            transformBData[localTexel] = floatToHalf(Math.exp(frameFloats[src]));
            transformBData[localTexel + 1] = floatToHalf(Math.exp(frameFloats[src + 1]));
            transformBData[localTexel + 2] = floatToHalf(Math.exp(frameFloats[src + 2]));
        }
    }
    const packEnd = performance.now();

    const transfers: Transferable[] = [];
    let transformABytes = 0;
    let transformBBytes = 0;
    const centersBytes = centers?.byteLength ?? 0;
    let bytes = centersBytes;
    for (const row of transformARows) {
        transformABytes += row.data.byteLength;
        bytes += row.data.byteLength;
        transfers.push(row.data.buffer);
    }
    for (const row of transformBRows) {
        transformBBytes += row.data.byteLength;
        bytes += row.data.byteLength;
        transfers.push(row.data.buffer);
    }
    if (centers) {
        transfers.push(centers.buffer);
    }
    const decodeMs = +(decodeEnd - decodeStart).toFixed(2);
    const packMs = +(packEnd - packStart).toFixed(2);
    const stats: PreparedFrameStats = {
        frame,
        trackedCount,
        touchedRowCount: state.touchedRowCount,
        textureHeight,
        rowRangeCount: rowRanges.length,
        transformABytes,
        transformBBytes,
        centersBytes,
        totalBytes: bytes,
        decodeMs,
        packMs,
        touchedRowRatio: state.touchedRowCount / Math.max(textureHeight, 1)
    };

    return {
        frame,
        transformARows,
        transformBRows,
        centers,
        bounds,
        bytes,
        stats,
        timings: {
            decode: decodeMs,
            pack: packMs
        },
        transfers
    };
};

self.onmessage = (event: MessageEvent<WorkerMessage>) => {
    const message = event.data;
    if (message.type === 'init') {
        const indices = message.indices ? new Uint32Array(message.indices) : undefined;
        let cursor = 0;
        let positionOffset = -1;
        let rotationOffset = -1;
        let scaleOffset = -1;
        const trackedCount = indices ? indices.length : message.numSplats;
        for (const attribute of message.attributes) {
            if (attribute === 'position') {
                positionOffset = cursor;
                cursor += trackedCount * 3;
            } else if (attribute === 'rotation') {
                rotationOffset = cursor;
                cursor += trackedCount * 4;
            } else if (attribute === 'scale') {
                scaleOffset = cursor;
                cursor += trackedCount * 3;
            }
        }
        rawChunkCache.clear();
        const rowMapping = buildRowMapping(
            indices,
            trackedCount,
            message.numSplats,
            message.textureWidth,
            message.textureHeight
        );
        state = {
            binaryUrl: message.binaryUrl,
            frameCount: message.frameCount,
            availableFrames: Math.min(message.availableFrames, message.frameCount),
            numSplats: message.numSplats,
            attributes: message.attributes,
            frameStrideBytes: message.frameStrideBytes,
            chunkFrameCount: message.chunkFrameCount,
            textureWidth: message.textureWidth,
            textureHeight: message.textureHeight,
            indices,
            baseTransformA: new Uint32Array(message.baseTransformA),
            baseTransformB: new Uint16Array(message.baseTransformB),
            trackedCount,
            positionOffset,
            rotationOffset,
            scaleOffset,
            floatsPerFrame: cursor,
            rowRanges: rowMapping.rowRanges,
            splatRangeIndex: rowMapping.splatRangeIndex,
            splatLocalTexel: rowMapping.splatLocalTexel,
            touchedRowCount: rowMapping.touchedRowCount
        };
        workerScope.postMessage({ type: 'ready' });
        return;
    }

    if (message.type === 'extendAvailableFrames') {
        if (state) {
            state.availableFrames = Math.max(state.availableFrames, Math.min(message.availableFrames, state.frameCount));
        }
        return;
    }

    if (message.type === 'destroy') {
        rawChunkCache.clear();
        state = null;
        return;
    }

    if (message.type === 'prepare') {
        void prepareFrame(message.frame)
            .then((prepared) => {
                if (!prepared) {
                    workerScope.postMessage({
                        type: 'missing',
                        id: message.id,
                        frame: message.frame
                    });
                    return;
                }
                const { transfers, ...payload } = prepared;
                workerScope.postMessage({
                    type: 'prepared',
                    id: message.id,
                    prepared: payload
                }, transfers);
            })
            .catch((error): void => {
                workerScope.postMessage({
                    type: 'error',
                    id: message.id,
                    message: error instanceof Error ? error.message : String(error)
                });
            });
    }
};

export {};
