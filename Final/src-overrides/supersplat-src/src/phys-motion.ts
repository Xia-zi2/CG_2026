import { Quat, Vec3 } from 'playcanvas';

import { Splat } from './splat';
import type { MotionPreparedFrame, MotionTextureSnapshot } from './splat';
import { Events } from './events';
import { Scene } from './scene';
import { State } from './splat-state';

type ImportFile = {
    filename: string;
    url?: string;
    contents?: File;
    skipReorder?: boolean;
};

type PhysMotionAttribute = 'position' | 'rotation' | 'scale';

type PhysMotionManifest = {
    format?: 'phys-motion-v1' | string;
    base?: string;
    binary: string;
    frameCount: number;
    availableFrames?: number;
    frameRate?: number;
    numSplats: number;
    attributes?: PhysMotionAttribute[];
    frameStrideBytes?: number;
    indices?: string;
    updateBounds?: boolean;
    proxy?: {
        motion: string;
        skinning: string;
        groupCount: number;
        blendCount: number;
        motionStrideBytes?: number;
        skinningStrideBytes?: number;
    };
};

type PhysMotionTrack = {
    manifest: PhysMotionManifest;
    disposed: boolean;
    binary?: ArrayBuffer;
    binaryUrl?: string;
    indices?: Uint32Array;
    proxyMotion?: ArrayBuffer;
    proxyMotionUrl?: string;
    proxySkinning?: ArrayBuffer;
    proxyRestCenters?: Float32Array;
    proxyRestPositions?: Float32Array;
    proxyRestRotations?: Float32Array;
    proxyRestScales?: Float32Array;
    proxyPalette?: {
        paletteStart: number;
        groupCount: number;
        previousTransformIndices: Uint16Array;
        mode: 'exact' | 'hard';
        averageWeight: number;
        exactRatio: number;
    };
    target?: Splat;
    floatsPerSplat: number;
    frameStrideBytes: number;
    chunkFrameCount: number;
    frameCache: Map<number, Promise<ArrayBuffer>>;
    cacheBudgetBytes: number;
    requestedFrame: number;
    appliedFrame: number;
    droppedFrames: number;
    availableFrames: number;
    preparedClient?: MotionWorkerClient;
    preparedFrameCache: Map<number, Promise<MotionPreparedFrame | null> | MotionPreparedFrame | null>;
    preparedCacheBudgetBytes: number;
    preparedBytesEstimate: number;
    preparedRequestMsEstimate: number;
    preparedApplyMsEstimate: number;
    preparedWarmupCompleted: boolean;
    preparedWarmupStatusFrameCount: number;
    streamPreviewFrameRate?: number;
};

const manifestNames = ['.physmotion.json', 'phys_motion.json'];
const bytesPerMiB = 1024 * 1024;
const proxyScratchQuat = new Quat();
const proxyBaseQuat = new Quat();
const proxyBlendQuat = new Quat();
const proxyPoint = new Vec3();
const proxyLocal = new Vec3();
const proxyRotated = new Vec3();
const proxyTmpVec = new Vec3();
const motionPrewarmParam = new URLSearchParams(window.location.search).get('physMotionPrewarm')?.toLowerCase();
const motionPrewarmEnabled = motionPrewarmParam !== 'off' && motionPrewarmParam !== '0' && motionPrewarmParam !== 'false';
const motionProfileEnabled = new URLSearchParams(window.location.search).get('physMotionProfile')?.toLowerCase() === 'on';
const motionPrewarmMaxFrames = Math.max(
    1,
    Math.min(
        240,
        Math.floor(Number(new URLSearchParams(window.location.search).get('physMotionPrewarmMaxFrames')) || 60)
    )
);

const isPhysMotionManifest = (filename: string) => {
    const lower = filename.toLowerCase();
    return manifestNames.some(name => lower.endsWith(name));
};

const normalizeName = (name: string) => name.replace(/\\/g, '/').split('/').pop().toLowerCase();

const findFile = (files: ImportFile[], filename: string) => {
    const target = normalizeName(filename);
    return files.find(file => normalizeName(file.filename) === target);
};

const readArrayBuffer = async (file: ImportFile) => {
    if (file.contents) {
        return file.contents.arrayBuffer();
    }
    if (file.url) {
        return (await fetch(file.url)).arrayBuffer();
    }
    throw new Error(`motion package file '${file.filename}' has no contents or url`);
};

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

const readText = async (file: ImportFile) => {
    if (file.contents) {
        return file.contents.text();
    }
    if (file.url) {
        return (await fetch(file.url)).text();
    }
    throw new Error(`motion package file '${file.filename}' has no contents or url`);
};

const floatsPerSplatFor = (attributes: PhysMotionAttribute[]) => {
    return attributes.reduce((sum, attribute) => {
        switch (attribute) {
            case 'position': return sum + 3;
            case 'rotation': return sum + 4;
            case 'scale': return sum + 3;
            default: return sum;
        }
    }, 0);
};

const normalizeQuatArray = (values: number[]) => {
    const length = Math.hypot(values[0], values[1], values[2], values[3]);
    if (length <= 1e-12) {
        values[0] = 1;
        values[1] = 0;
        values[2] = 0;
        values[3] = 0;
        return values;
    }
    values[0] /= length;
    values[1] /= length;
    values[2] /= length;
    values[3] /= length;
    return values;
};

const getSceneSplats = (events: Events) => (events.invoke('scene.splats') as Splat[]) ?? [];

const proxyPaletteMode = () => {
    const mode = new URLSearchParams(window.location.search).get('physProxyPalette')?.toLowerCase() ?? 'auto';
    if (mode === '0' || mode === 'off' || mode === 'false') return 'off';
    if (mode === 'exact') return 'exact';
    if (mode === 'hard' || mode === '1' || mode === 'true') return 'hard';
    return 'auto';
};

class MotionWorkerClient {
    private worker: Worker;
    private nextId = 1;
    private pending = new Map<number, {
        resolve: (value: MotionPreparedFrame | null) => void;
        reject: (error: Error) => void;
    }>();

    constructor() {
        this.worker = new Worker(new URL('./phys-motion-worker.js', import.meta.url), { type: 'module' });
        this.worker.onmessage = (event: MessageEvent<any>) => {
            const message = event.data;
            if (message.type === 'prepared') {
                const pending = this.pending.get(message.id);
                if (pending) {
                    this.pending.delete(message.id);
                    pending.resolve(message.prepared as MotionPreparedFrame);
                }
            } else if (message.type === 'missing') {
                const pending = this.pending.get(message.id);
                if (pending) {
                    this.pending.delete(message.id);
                    pending.resolve(null);
                }
            } else if (message.type === 'error') {
                const pending = this.pending.get(message.id);
                if (pending) {
                    this.pending.delete(message.id);
                    pending.reject(new Error(message.message));
                }
            }
        };
        this.worker.onerror = (event) => {
            const message = event.message || 'motion worker failed';
            for (const pending of this.pending.values()) {
                pending.reject(new Error(message));
            }
            this.pending.clear();
        };
    }

    init(payload: {
        binaryUrl: string;
        frameCount: number;
        availableFrames: number;
        numSplats: number;
        attributes: PhysMotionAttribute[];
        frameStrideBytes: number;
        chunkFrameCount: number;
        textureSnapshot: MotionTextureSnapshot;
        indices?: Uint32Array;
    }) {
        const transfers: Transferable[] = [
            payload.textureSnapshot.baseTransformA,
            payload.textureSnapshot.baseTransformB
        ];
        let indices: ArrayBuffer | undefined;
        if (payload.indices) {
            indices = new Uint32Array(payload.indices).buffer;
            transfers.push(indices);
        }
        this.worker.postMessage({
            type: 'init',
            binaryUrl: payload.binaryUrl,
            frameCount: payload.frameCount,
            availableFrames: payload.availableFrames,
            numSplats: payload.numSplats,
            attributes: payload.attributes,
            frameStrideBytes: payload.frameStrideBytes,
            chunkFrameCount: payload.chunkFrameCount,
            textureWidth: payload.textureSnapshot.textureWidth,
            textureHeight: payload.textureSnapshot.textureHeight,
            baseTransformA: payload.textureSnapshot.baseTransformA,
            baseTransformB: payload.textureSnapshot.baseTransformB,
            indices
        }, transfers);
    }

    setAvailableFrames(availableFrames: number) {
        this.worker.postMessage({
            type: 'extendAvailableFrames',
            availableFrames
        });
    }

    prepare(frame: number) {
        const id = this.nextId++;
        const promise = new Promise<MotionPreparedFrame | null>((resolve, reject) => {
            this.pending.set(id, { resolve, reject });
        });
        this.worker.postMessage({ type: 'prepare', id, frame });
        return promise;
    }

    destroy() {
        for (const pending of this.pending.values()) {
            pending.resolve(null);
        }
        this.pending.clear();
        this.worker.postMessage({ type: 'destroy' });
        this.worker.terminate();
    }
}

const chooseHardProxyGroups = (proxySkinning: ArrayBuffer, count: number, blendCount: number) => {
    const groupIds = new Uint32Array(proxySkinning, 0, count * blendCount);
    const weights = new Float32Array(proxySkinning, groupIds.byteLength, count * blendCount);
    const hardGroups = new Uint32Array(count);
    let exactCount = 0;
    let weightSum = 0;

    for (let i = 0; i < count; ++i) {
        let bestGroup = groupIds[i * blendCount];
        let bestWeight = weights[i * blendCount] ?? 0;
        for (let j = 1; j < blendCount; ++j) {
            const offset = i * blendCount + j;
            const weight = weights[offset] ?? 0;
            if (weight > bestWeight) {
                bestWeight = weight;
                bestGroup = groupIds[offset];
            }
        }
        hardGroups[i] = bestGroup;
        weightSum += bestWeight;
        if (bestWeight >= 0.999) {
            exactCount++;
        }
    }

    return {
        hardGroups,
        averageWeight: count > 0 ? weightSum / count : 0,
        exactRatio: count > 0 ? exactCount / count : 0
    };
};

const registerPhysMotionEvents = (events: Events, _scene: Scene) => {
    let activeTrack: PhysMotionTrack | null = null;
    let applying = false;
    let setFrameAgain = false;

    const restoreProxyPalette = (track: PhysMotionTrack | null) => {
        if (!track?.proxyPalette || !track.target || !track.indices) {
            return;
        }
        track.target.restoreMotionProxyPalette(track.indices, track.proxyPalette.previousTransformIndices, track.proxyPalette.groupCount);
        track.proxyPalette = undefined;
    };

    const disposeTrack = (track: PhysMotionTrack | null) => {
        if (!track || track.disposed) {
            return;
        }
        track.disposed = true;
        restoreProxyPalette(track);
        track.preparedClient?.destroy();
        track.preparedClient = undefined;
        track.preparedFrameCache.clear();
        track.frameCache.clear();
        track.target = undefined;
        track.binary = undefined;
        track.indices = undefined;
        track.proxyMotion = undefined;
        track.proxySkinning = undefined;
        track.proxyRestCenters = undefined;
        track.proxyRestPositions = undefined;
        track.proxyRestRotations = undefined;
        track.proxyRestScales = undefined;
        track.proxyPalette = undefined;
    };

    const resolveTarget = () => {
        const selected = events.invoke('selection') as Splat;
        if (selected?.visible) {
            return selected;
        }
        return getSceneSplats(events)[0] ?? null;
    };

    const importBaseIfNeeded = async (files: ImportFile[], manifest: PhysMotionManifest) => {
        const existing = getSceneSplats(events);
        if (existing.length > 0 || !manifest.base) {
            return existing[0] ?? null;
        }

        const baseFile = findFile(files, manifest.base);
        if (!baseFile) {
            throw new Error(`motion manifest references base '${manifest.base}', but it was not provided`);
        }

        const imported = await events.invoke('import', [{ ...baseFile, skipReorder: true }]) as Splat[];
        return imported?.[0] ?? null;
    };

    const ensureFrameChunk = (track: PhysMotionTrack, frame: number) => {
        if (track.disposed) {
            return Promise.reject(new Error('Motion track has been disposed'));
        }
        if (track.binary) {
            return Promise.resolve(track.binary);
        }
        if (!track.binaryUrl) {
            throw new Error('Motion binary is not available');
        }

        const chunkIndex = Math.floor(frame / track.chunkFrameCount);
        const cached = track.frameCache.get(chunkIndex);
        if (cached) {
            return cached;
        }

        const startFrame = chunkIndex * track.chunkFrameCount;
        const frameCount = Math.min(track.chunkFrameCount, track.manifest.frameCount - startFrame);
        const startByte = startFrame * track.frameStrideBytes;
        const endByte = (startFrame + frameCount) * track.frameStrideBytes;
        const promise = readUrlRange(track.binaryUrl, startByte, endByte).catch((error): never => {
            track.frameCache.delete(chunkIndex);
            throw error;
        });
        track.frameCache.set(chunkIndex, promise);

        const maxCachedChunks = Math.max(2, Math.floor(track.cacheBudgetBytes / Math.max(track.frameStrideBytes * track.chunkFrameCount, 1)));
        while (track.frameCache.size > maxCachedChunks) {
            const firstKey = track.frameCache.keys().next().value;
            if (firstKey === chunkIndex) {
                break;
            }
            track.frameCache.delete(firstKey);
        }

        void promise.then(() => {
            if (track.disposed) {
                return;
            }
            const nextChunk = chunkIndex + 1;
            if (nextChunk * track.chunkFrameCount < track.manifest.frameCount && !track.frameCache.has(nextChunk)) {
                void ensureFrameChunk(track, nextChunk * track.chunkFrameCount);
            }
        }).catch((): null => null);

        return promise;
    };

    const decodeFrame = async (track: PhysMotionTrack, frame: number) => {
        if (track.disposed) {
            return null;
        }
        const proxyFrame = decodeProxyFrame(track, frame);
        if (proxyFrame) {
            return proxyFrame;
        }

        const { manifest, indices, floatsPerSplat, frameStrideBytes } = track;
        const attributes = manifest.attributes ?? ['position'];
        const count = indices ? indices.length : manifest.numSplats;
        const binary = await ensureFrameChunk(track, frame);
        if (track.disposed) {
            return null;
        }
        const chunkStartFrame = track.binary ? 0 : Math.floor(frame / track.chunkFrameCount) * track.chunkFrameCount;
        const byteOffset = (frame - chunkStartFrame) * frameStrideBytes;
        const floatOffset = byteOffset / Float32Array.BYTES_PER_ELEMENT;
        const floatsThisFrame = count * floatsPerSplat;
        const frameFloats = new Float32Array(binary, floatOffset * Float32Array.BYTES_PER_ELEMENT, floatsThisFrame);

        let cursor = 0;
        let positions: Float32Array = null;
        let rotations: Float32Array = null;
        let scales: Float32Array = null;

        for (const attribute of attributes) {
            const width = attribute === 'rotation' ? 4 : 3;
            const data = frameFloats.subarray(cursor, cursor + count * width);
            if (attribute === 'position') positions = data;
            if (attribute === 'rotation') rotations = data;
            if (attribute === 'scale') scales = data;
            cursor += count * width;
        }

        return {
            positions,
            rotations,
            scales,
            indices,
            updateBounds: manifest.updateBounds
        };
    };

    const decodeProxyFrame = (track: PhysMotionTrack, frame: number) => {
        if (track.disposed) {
            return null;
        }
        const { manifest, indices, proxyMotion, proxySkinning } = track;
        const proxy = manifest.proxy;
        if (!proxy || !indices || (!proxyMotion && !track.proxyMotionUrl) || !proxySkinning || !track.target) {
            return null;
        }

        const groupCount = proxy.groupCount;
        const blendCount = proxy.blendCount;
        const count = indices.length;
        if (!proxyMotion) {
            return null;
        }
        const motionFloatsPerFrame = groupCount * 7;
        const proxyFloats = new Float32Array(proxyMotion, frame * motionFloatsPerFrame * Float32Array.BYTES_PER_ELEMENT, motionFloatsPerFrame);
        if (track.proxyPalette && track.proxyRestCenters) {
            return {
                proxyPalette: {
                    paletteStart: track.proxyPalette.paletteStart,
                    groupCount,
                    restCenters: track.proxyRestCenters,
                    frame: proxyFloats
                },
                updateBounds: false
            };
        }

        const groupIds = new Uint32Array(proxySkinning, 0, count * blendCount);
        const weights = new Float32Array(proxySkinning, count * blendCount * Uint32Array.BYTES_PER_ELEMENT, count * blendCount);
        const restCenters = track.proxyRestCenters ?? new Float32Array(proxyMotion, 0, groupCount * 3);
        const restPositions = track.proxyRestPositions;
        const restRotations = track.proxyRestRotations;
        const restScales = track.proxyRestScales;
        if (!restPositions || !restRotations || !restScales) {
            return null;
        }
        const positions = new Float32Array(count * 3);
        const rotations = new Float32Array(count * 4);
        const scales = new Float32Array(count * 3);

        for (let i = 0; i < count; ++i) {
            const id = indices[i];
            const src3 = i * 3;
            const src4 = i * 4;
            proxyPoint.set(restPositions[src3], restPositions[src3 + 1], restPositions[src3 + 2]);
            const quatAccum = [0, 0, 0, 0];
            proxyTmpVec.set(0, 0, 0);
            let totalWeight = 0;
            let referenceQuat: number[] = null;

            for (let j = 0; j < blendCount; ++j) {
                const skinIndex = i * blendCount + j;
                const weight = weights[skinIndex];
                if (weight <= 0) {
                    continue;
                }
                const groupId = groupIds[skinIndex];
                if (groupId >= groupCount) {
                    continue;
                }
                const centerOffset = groupId * 3;
                const rotationOffset = groupCount * 3 + groupId * 4;
                const cx = proxyFloats[centerOffset];
                const cy = proxyFloats[centerOffset + 1];
                const cz = proxyFloats[centerOffset + 2];
                const rx = restCenters[centerOffset];
                const ry = restCenters[centerOffset + 1];
                const rz = restCenters[centerOffset + 2];
                const qw = proxyFloats[rotationOffset];
                const qx = proxyFloats[rotationOffset + 1];
                const qy = proxyFloats[rotationOffset + 2];
                const qz = proxyFloats[rotationOffset + 3];
                proxyScratchQuat.set(qx, qy, qz, qw).normalize();
                proxyLocal.set(proxyPoint.x - rx, proxyPoint.y - ry, proxyPoint.z - rz);
                proxyScratchQuat.transformVector(proxyLocal, proxyRotated);
                proxyTmpVec.x += weight * (cx + proxyRotated.x);
                proxyTmpVec.y += weight * (cy + proxyRotated.y);
                proxyTmpVec.z += weight * (cz + proxyRotated.z);

                const quat = [qw, qx, qy, qz];
                if (!referenceQuat) {
                    referenceQuat = quat;
                }
                const sign = referenceQuat[0] * quat[0] + referenceQuat[1] * quat[1] + referenceQuat[2] * quat[2] + referenceQuat[3] * quat[3] < 0 ? -1 : 1;
                quatAccum[0] += sign * weight * quat[0];
                quatAccum[1] += sign * weight * quat[1];
                quatAccum[2] += sign * weight * quat[2];
                quatAccum[3] += sign * weight * quat[3];
                totalWeight += weight;
            }

            if (totalWeight <= 1e-8) {
                proxyTmpVec.copy(proxyPoint);
                quatAccum[0] = 1;
            } else if (Math.abs(totalWeight - 1) > 1e-5) {
                proxyTmpVec.mulScalar(1 / totalWeight);
            }
            normalizeQuatArray(quatAccum);

            positions[src3] = proxyTmpVec.x;
            positions[src3 + 1] = proxyTmpVec.y;
            positions[src3 + 2] = proxyTmpVec.z;
            proxyBlendQuat.set(quatAccum[1], quatAccum[2], quatAccum[3], quatAccum[0]).normalize();
            proxyBaseQuat.set(restRotations[src4 + 1], restRotations[src4 + 2], restRotations[src4 + 3], restRotations[src4]).normalize();
            proxyBlendQuat.mul(proxyBaseQuat);
            rotations[src4] = proxyBlendQuat.w;
            rotations[src4 + 1] = proxyBlendQuat.x;
            rotations[src4 + 2] = proxyBlendQuat.y;
            rotations[src4 + 3] = proxyBlendQuat.z;
            scales[src3] = restScales[src3];
            scales[src3 + 1] = restScales[src3 + 1];
            scales[src3 + 2] = restScales[src3 + 2];
        }

        return {
            positions,
            rotations,
            scales,
            indices,
            updateBounds: false
        };
    };

    const isPromiseLike = (value: unknown): value is Promise<MotionPreparedFrame | null> => {
        return !!value && typeof (value as any).then === 'function';
    };

    const updateEma = (current: number, sample: number, alpha = 0.25) => (
        current > 0 ? current * (1 - alpha) + sample * alpha : sample
    );

    const circularDistance = (a: number, b: number, n: number) => {
        if (n <= 0) {
            return Math.abs(a - b);
        }
        const d = Math.abs(a - b);
        return Math.min(d, n - d);
    };

    const canLoopPreparedFrames = (track: PhysMotionTrack) => (
        track.availableFrames >= track.manifest.frameCount && track.manifest.frameCount > 0
    );

    const preparedFrameDistance = (frame: number, keepFrame: number, track: PhysMotionTrack) => (
        canLoopPreparedFrames(track)
            ? circularDistance(frame, keepFrame, track.manifest.frameCount)
            : Math.abs(frame - keepFrame)
    );

    const preparedFrameStats = (track: PhysMotionTrack) => {
        let resolvedCount = 0;
        let pendingCount = 0;
        let resolvedBytes = 0;
        for (const value of track.preparedFrameCache.values()) {
            if (!value) {
                continue;
            }
            if (isPromiseLike(value)) {
                pendingCount++;
            } else {
                resolvedCount++;
                resolvedBytes += value.bytes;
            }
        }
        return { resolvedCount, pendingCount, resolvedBytes };
    };

    const canWarmFullPreparedLoop = (track: PhysMotionTrack) => {
        if (!motionPrewarmEnabled || !track.preparedClient || track.availableFrames < track.manifest.frameCount) {
            return false;
        }
        if (track.manifest.frameCount > motionPrewarmMaxFrames) {
            return false;
        }
        const preparedBytes = Math.max(track.preparedBytesEstimate || track.frameStrideBytes, 1);
        return track.manifest.frameCount * preparedBytes <= track.preparedCacheBudgetBytes;
    };

    const logPreparedFrameStats = (track: PhysMotionTrack, prepared: MotionPreparedFrame) => {
        if (!motionProfileEnabled || !prepared.stats) {
            return;
        }

        const stats = prepared.stats;
        const cacheStats = preparedFrameStats(track);
        console.table({
            frame: stats.frame,
            preparedMB: (stats.totalBytes / bytesPerMiB).toFixed(1),
            touchedRows: `${stats.touchedRowCount}/${stats.textureHeight}`,
            touchedRatio: stats.touchedRowRatio.toFixed(3),
            rowRanges: stats.rowRangeCount,
            decodeMs: stats.decodeMs.toFixed(2),
            packMs: stats.packMs.toFixed(2),
            cacheMB: (cacheStats.resolvedBytes / bytesPerMiB).toFixed(1),
            budgetMB: (track.preparedCacheBudgetBytes / bytesPerMiB).toFixed(1),
            pending: cacheStats.pendingCount
        });

        if (stats.touchedRowRatio > 0.5) {
            console.warn(
                `[physmotion] high touched row ratio ${stats.touchedRowRatio.toFixed(2)}; consider static/dynamic splat split or dynamic index compaction`
            );
        }
    };

    const setPlaybackFrameRate = (track: PhysMotionTrack) => {
        const frameRate = track.manifest.frameRate;
        if (!frameRate) {
            return;
        }
        const streaming = !!track.binaryUrl && track.availableFrames < track.manifest.frameCount;
        const warmingPreparedLoop = canWarmFullPreparedLoop(track) && !track.preparedWarmupCompleted;
        const playbackFrameRate = streaming
            ? Math.min(frameRate, track.streamPreviewFrameRate ?? 12)
            : warmingPreparedLoop
                ? Math.min(frameRate, track.streamPreviewFrameRate ?? 12)
            : frameRate;
        events.fire('timeline.setFrameRate', playbackFrameRate);
    };

    const evictPreparedFrames = (track: PhysMotionTrack, keepFrame: number) => {
        if (track.disposed) {
            return;
        }
        let totalBytes = 0;
        const resolved: Array<[number, MotionPreparedFrame]> = [];
        for (const [frame, value] of track.preparedFrameCache.entries()) {
            if (!value || isPromiseLike(value)) {
                continue;
            }
            resolved.push([frame, value]);
            totalBytes += value.bytes;
        }
        if (totalBytes <= track.preparedCacheBudgetBytes) {
            return;
        }

        resolved.sort((a, b) => (
            preparedFrameDistance(b[0], keepFrame, track) - preparedFrameDistance(a[0], keepFrame, track)
        ));
        for (const [frame, value] of resolved) {
            if (frame === keepFrame || value.pinned) {
                continue;
            }
            track.preparedFrameCache.delete(frame);
            totalBytes -= value.bytes;
            if (totalBytes <= track.preparedCacheBudgetBytes) {
                break;
            }
        }
    };

    const maxPendingPreparedFrames = (track: PhysMotionTrack) => {
        const preparedBytes = Math.max(track.preparedBytesEstimate || track.frameStrideBytes, 1);
        const budgetFrames = Math.floor(track.preparedCacheBudgetBytes / preparedBytes);
        if (budgetFrames <= 1) {
            return 1;
        }
        return Math.max(1, Math.min(8, Math.floor(budgetFrames / 2)));
    };

    const requestPreparedFrame = (track: PhysMotionTrack, frame: number) => {
        if (track.disposed || !track.preparedClient || frame < 0 || frame >= track.availableFrames || frame >= track.manifest.frameCount) {
            return Promise.resolve(null);
        }
        const cached = track.preparedFrameCache.get(frame);
        if (cached !== undefined) {
            return isPromiseLike(cached) ? cached : Promise.resolve(cached);
        }

        const requestStart = performance.now();
        const promise = track.preparedClient.prepare(frame).then((prepared) => {
            if (track.disposed) {
                track.preparedFrameCache.delete(frame);
                return null;
            }
            if (!prepared) {
                track.preparedFrameCache.delete(frame);
                return null;
            }
            const requestMs = performance.now() - requestStart;
            prepared.timings = {
                ...(prepared.timings ?? {}),
                request: +requestMs.toFixed(2)
            };
            track.preparedFrameCache.set(frame, prepared);
            track.preparedBytesEstimate = Math.max(track.preparedBytesEstimate, prepared.bytes);
            track.preparedRequestMsEstimate = updateEma(track.preparedRequestMsEstimate, requestMs);
            evictPreparedFrames(track, track.requestedFrame);
            logPreparedFrameStats(track, prepared);
            return prepared;
        }).catch((error): MotionPreparedFrame | null => {
            track.preparedFrameCache.delete(frame);
            if (track.disposed) {
                return null;
            }
            throw error;
        });
        track.preparedFrameCache.set(frame, promise);
        return promise;
    };

    const markPreparedWarmupIfReady = (track: PhysMotionTrack, orderedFrames: number[]) => {
        if (track.preparedWarmupCompleted || !canWarmFullPreparedLoop(track)) {
            return;
        }
        let resolvedBytes = 0;
        for (const frame of orderedFrames) {
            const cached = track.preparedFrameCache.get(frame);
            if (!cached || isPromiseLike(cached)) {
                return;
            }
            resolvedBytes += cached.bytes;
        }
        if (resolvedBytes > track.preparedCacheBudgetBytes) {
            return;
        }
        for (const frame of orderedFrames) {
            const cached = track.preparedFrameCache.get(frame);
            if (cached && !isPromiseLike(cached)) {
                cached.pinned = true;
            }
        }
        track.preparedWarmupCompleted = true;
        track.preparedWarmupStatusFrameCount = orderedFrames.length;
        setPlaybackFrameRate(track);
        events.fire(
            'physics.status',
            `动画帧缓存完成：${orderedFrames.length}/${track.manifest.frameCount} 帧，单帧 prepare 约 ${track.preparedRequestMsEstimate.toFixed(1)}ms，上传约 ${track.preparedApplyMsEstimate.toFixed(1)}ms`
        );
    };

    const plannedPreparedFrames = (track: PhysMotionTrack, frame: number) => {
        if (canWarmFullPreparedLoop(track)) {
            return Array.from({ length: track.manifest.frameCount }, (_value, index) => index);
        }

        const available = Math.max(0, Math.min(track.availableFrames, track.manifest.frameCount));
        if (available <= 0) {
            return [];
        }

        const clampedFrame = Math.max(0, Math.min(frame, available - 1));
        const preparedBytes = Math.max(track.preparedBytesEstimate || track.frameStrideBytes, 1);
        const budgetFrames = Math.max(1, Math.floor(track.preparedCacheBudgetBytes / preparedBytes));
        const frameRate = track.manifest.frameRate || 30;
        const frameIntervalMs = 1000 / Math.max(frameRate, 1);
        const measuredFrames = Math.ceil(
            (track.preparedRequestMsEstimate + track.preparedApplyMsEstimate) / Math.max(frameIntervalMs, 1)
        );
        const lookaheadFrames = Math.max(
            0,
            Math.min(8, budgetFrames - 1, measuredFrames + 6)
        );
        const ordered: number[] = [clampedFrame];
        const canWrap = canLoopPreparedFrames(track);
        for (let offset = 1; offset <= lookaheadFrames; ++offset) {
            let nextFrame = clampedFrame + offset;
            if (nextFrame >= available) {
                if (!canWrap) {
                    break;
                }
                nextFrame = nextFrame % track.manifest.frameCount;
            }
            ordered.push(nextFrame);
        }
        return Array.from(new Set(ordered));
    };

    const schedulePreparedLookahead = (track: PhysMotionTrack, frame: number) => {
        if (track.disposed || !track.preparedClient || track.availableFrames <= 0) {
            return;
        }
        const orderedFrames = plannedPreparedFrames(track, frame);
        if (orderedFrames.length === 0) {
            return;
        }
        const pendingBudget = maxPendingPreparedFrames(track);
        let pendingCount = preparedFrameStats(track).pendingCount;
        let started = 0;

        for (const nextFrame of orderedFrames) {
            const cached = track.preparedFrameCache.get(nextFrame);
            if (cached !== undefined) {
                continue;
            }

            if (pendingCount >= pendingBudget) {
                break;
            }

            pendingCount++;
            started++;

            void requestPreparedFrame(track, nextFrame)
                .then((prepared) => {
                    if (prepared && !track.disposed && activeTrack === track) {
                        markPreparedWarmupIfReady(track, orderedFrames);
                        schedulePreparedLookahead(track, track.requestedFrame);
                    }
                })
                .catch((error) => {
                    if (!track.disposed) {
                        console.warn('[physmotion] prepared lookahead failed', nextFrame, error);
                    }
                });
        }

        markPreparedWarmupIfReady(track, orderedFrames);
        if (started > 0 && canWarmFullPreparedLoop(track) && track.preparedWarmupStatusFrameCount !== orderedFrames.length) {
            track.preparedWarmupStatusFrameCount = orderedFrames.length;
            const stats = preparedFrameStats(track);
            events.fire(
                'physics.status',
                `warming full motion cache: ${stats.resolvedCount + stats.pendingCount}/${orderedFrames.length} frames; playback frame rate will restore after warmup`
            );
        }
    };

    const setFrame = async (frame: number) => {
        if (!activeTrack || activeTrack.disposed) {
            return;
        }

        const { manifest } = activeTrack;
        if (frame < 0 || frame >= manifest.frameCount) {
            return;
        }
        activeTrack.requestedFrame = frame;
        if (frame >= activeTrack.availableFrames) {
            events.fire('physics.status', `buffering motion frame ${frame + 1}/${manifest.frameCount}; available ${activeTrack.availableFrames}`);
            schedulePreparedLookahead(activeTrack, Math.max(0, activeTrack.availableFrames - 1));
            return;
        }

        if (applying) {
            setFrameAgain = true;
            return;
        }

        applying = true;
        try {
            while (activeTrack && !activeTrack.disposed) {
                const track = activeTrack;
                setFrameAgain = false;
                const frameToApply = track.requestedFrame;
                if (frameToApply >= track.availableFrames) {
                    events.fire('physics.status', `buffering motion frame ${frameToApply + 1}/${manifest.frameCount}; available ${track.availableFrames}`);
                    schedulePreparedLookahead(track, Math.max(0, track.availableFrames - 1));
                    break;
                }
                const target = track.target?.visible ? track.target : resolveTarget();
                if (!target) {
                    return;
                }
                track.target = target;

                if (target.splatData.numSplats !== manifest.numSplats && !track.indices) {
                    throw new Error(`motion numSplats=${manifest.numSplats} does not match loaded splat numSplats=${target.splatData.numSplats}`);
                }

                if (track.preparedClient) {
                    const prepared = await requestPreparedFrame(track, frameToApply);
                    if (!prepared || track.disposed || activeTrack !== track) {
                        events.fire('physics.status', `buffering motion frame ${frameToApply + 1}/${manifest.frameCount}`);
                        break;
                    }
                    if (track.requestedFrame !== frameToApply && setFrameAgain) {
                        continue;
                    }
                    const applyStart = performance.now();
                    const ok = await target.applyPreparedMotionFrame(prepared, track.indices);
                    if (track.disposed || activeTrack !== track) {
                        break;
                    }
                    if (!ok) {
                        events.fire('physics.status', `failed to upload prepared motion frame ${frameToApply + 1}/${manifest.frameCount}`);
                        break;
                    }
                    track.preparedApplyMsEstimate = updateEma(track.preparedApplyMsEstimate, performance.now() - applyStart);
                    schedulePreparedLookahead(track, frameToApply);
                } else {
                    const decoded = await decodeFrame(track, frameToApply);
                    if (!decoded || track.disposed || activeTrack !== track) {
                        break;
                    }
                    await target.applyMotionFrame(decoded);
                    if (track.disposed || activeTrack !== track) {
                        break;
                    }
                }
                if (frameToApply !== track.appliedFrame && track.appliedFrame !== -1) {
                    const skipped = Math.max(0, Math.abs(frameToApply - track.appliedFrame) - 1);
                    track.droppedFrames += skipped;
                }
                track.appliedFrame = frameToApply;

                if (!setFrameAgain || track.requestedFrame === frameToApply) {
                    break;
                }
            }
        } finally {
            const rerunRequested = setFrameAgain && !!activeTrack && !activeTrack.disposed;
            applying = false;
            if (rerunRequested) {
                const requestedFrame = activeTrack.requestedFrame;
                setFrameAgain = false;
                void setFrame(requestedFrame);
            }
        }
    };

    events.function('physmotion.loadPackage', async (files: ImportFile[]) => {
        const manifestFile = files.find(file => isPhysMotionManifest(file.filename));
        if (!manifestFile) {
            throw new Error('No .physmotion.json manifest found in motion package');
        }

        events.fire('startSpinner');
        try {
            const manifest = JSON.parse(await readText(manifestFile)) as PhysMotionManifest;
            const attributes = manifest.attributes ?? ['position'];

            if (!manifest.binary || !manifest.frameCount || !manifest.numSplats) {
                throw new Error('Motion manifest must define binary, frameCount, and numSplats');
            }
            disposeTrack(activeTrack);
            activeTrack = null;

            const binaryFile = findFile(files, manifest.binary);
            if (!binaryFile) {
                throw new Error(`motion manifest references binary '${manifest.binary}', but it was not provided`);
            }

            const target = await importBaseIfNeeded(files, manifest);
            let streamBinary = !!binaryFile.url && !binaryFile.contents;
            let binary = streamBinary ? undefined : await readArrayBuffer(binaryFile);
            let indices: Uint32Array = null;
            if (manifest.indices) {
                const indexFile = findFile(files, manifest.indices);
                if (!indexFile) {
                    throw new Error(`motion manifest references indices '${manifest.indices}', but it was not provided`);
                }
                indices = new Uint32Array(await readArrayBuffer(indexFile));
            }
            let proxyMotion: ArrayBuffer = undefined;
            let proxyMotionUrl: string = undefined;
            let proxySkinning: ArrayBuffer = undefined;
            let proxyRestCenters: Float32Array = undefined;
            let proxyRestPositions: Float32Array = undefined;
            let proxyRestRotations: Float32Array = undefined;
            let proxyRestScales: Float32Array = undefined;
            let proxyPalette: PhysMotionTrack['proxyPalette'] = undefined;
            if (manifest.proxy) {
                const proxyMotionFile = findFile(files, manifest.proxy.motion);
                const proxySkinningFile = findFile(files, manifest.proxy.skinning);
                if (proxyMotionFile && proxySkinningFile && indices && target) {
                    proxySkinning = await readArrayBuffer(proxySkinningFile);
                    if (proxyMotionFile.url && !proxyMotionFile.contents) {
                        proxyMotionUrl = proxyMotionFile.url;
                        proxyMotion = await readArrayBuffer(proxyMotionFile);
                    } else {
                        proxyMotion = await readArrayBuffer(proxyMotionFile);
                    }
                    proxyRestCenters = new Float32Array(proxyMotion, 0, manifest.proxy.groupCount * 3);
                    const splatData = target.splatData;
                    const restX = splatData.getProp('x') as Float32Array;
                    const restY = splatData.getProp('y') as Float32Array;
                    const restZ = splatData.getProp('z') as Float32Array;
                    const restR0 = splatData.getProp('rot_0') as Float32Array;
                    const restR1 = splatData.getProp('rot_1') as Float32Array;
                    const restR2 = splatData.getProp('rot_2') as Float32Array;
                    const restR3 = splatData.getProp('rot_3') as Float32Array;
                    const restS0 = splatData.getProp('scale_0') as Float32Array;
                    const restS1 = splatData.getProp('scale_1') as Float32Array;
                    const restS2 = splatData.getProp('scale_2') as Float32Array;
                    proxyRestPositions = new Float32Array(indices.length * 3);
                    proxyRestRotations = new Float32Array(indices.length * 4);
                    proxyRestScales = new Float32Array(indices.length * 3);
                    for (let i = 0; i < indices.length; ++i) {
                        const id = indices[i];
                        const dst3 = i * 3;
                        const dst4 = i * 4;
                        proxyRestPositions[dst3] = restX[id];
                        proxyRestPositions[dst3 + 1] = restY[id];
                        proxyRestPositions[dst3 + 2] = restZ[id];
                        proxyRestRotations[dst4] = restR0[id];
                        proxyRestRotations[dst4 + 1] = restR1[id];
                        proxyRestRotations[dst4 + 2] = restR2[id];
                        proxyRestRotations[dst4 + 3] = restR3[id];
                        proxyRestScales[dst3] = restS0[id];
                        proxyRestScales[dst3 + 1] = restS1[id];
                        proxyRestScales[dst3 + 2] = restS2[id];
                    }
                    const mode = proxyPaletteMode();
                    const { hardGroups, averageWeight, exactRatio } = chooseHardProxyGroups(proxySkinning, indices.length, manifest.proxy.blendCount);
                    const useExact = exactRatio >= 0.999;
                    const useHard = mode === 'hard' || (mode === 'auto' && averageWeight >= 0.55);
                    if (mode !== 'off' && manifest.proxy.groupCount > 0 && (useExact || useHard)) {
                        const binding = target.setupMotionProxyPalette(indices, hardGroups, manifest.proxy.groupCount);
                        proxyPalette = {
                            ...binding,
                            mode: useExact ? 'exact' : 'hard',
                            averageWeight,
                            exactRatio
                        };
                    }
                    streamBinary = false;
                    binary = undefined;
                }
            }

            const floatsPerSplat = floatsPerSplatFor(attributes);
            const trackedSplats = indices ? indices.length : manifest.numSplats;
            const frameStrideBytes = manifest.frameStrideBytes ?? trackedSplats * floatsPerSplat * Float32Array.BYTES_PER_ELEMENT;
            if (frameStrideBytes % Float32Array.BYTES_PER_ELEMENT !== 0) {
                throw new Error('Motion frameStrideBytes must be divisible by 4');
            }
            if (binary && binary.byteLength < frameStrideBytes * manifest.frameCount) {
                throw new Error(`Motion binary is too small for ${manifest.frameCount} frames`);
            }

            const deviceMemoryGB = Math.max(1, Math.min(8, Number((navigator as any).deviceMemory) || 4));
            const streamCacheBudgetBytes = Math.round(Math.max(32, Math.min(64, deviceMemoryGB * 12))) * bytesPerMiB;
            const preparedCacheBudgetBytes = Math.round(Math.max(96, Math.min(256, deviceMemoryGB * 48))) * bytesPerMiB;
            const chunkTargetBytes = 12 * bytesPerMiB;
            const chunkFrameCount = Math.max(1, Math.min(8, Math.floor(chunkTargetBytes / frameStrideBytes) || 1));
            const manifestAvailableFrames = manifest.availableFrames === undefined
                ? manifest.frameCount
                : Number(manifest.availableFrames);
            const availableFrames = Math.max(
                0,
                Math.min(Math.floor(Number.isFinite(manifestAvailableFrames) ? manifestAvailableFrames : manifest.frameCount), manifest.frameCount)
            );
            let preparedClient: MotionWorkerClient | undefined;
            if (streamBinary && target && !proxyMotion && !proxySkinning) {
                const textureSnapshot = target.getMotionTextureSnapshot();
                if (textureSnapshot) {
                    preparedClient = new MotionWorkerClient();
                    preparedClient.init({
                        binaryUrl: binaryFile.url!,
                        frameCount: manifest.frameCount,
                        availableFrames,
                        numSplats: manifest.numSplats,
                        attributes,
                        frameStrideBytes,
                        chunkFrameCount,
                        textureSnapshot,
                        indices
                    });
                }
            }
            activeTrack = {
                manifest: {
                    ...manifest,
                    attributes
                },
                disposed: false,
                binary,
                binaryUrl: streamBinary ? binaryFile.url : undefined,
                indices,
                proxyMotion,
                proxyMotionUrl,
                proxySkinning,
                proxyRestCenters,
                proxyRestPositions,
                proxyRestRotations,
                proxyRestScales,
                proxyPalette,
                target,
                floatsPerSplat,
                frameStrideBytes,
                chunkFrameCount,
                frameCache: new Map(),
                cacheBudgetBytes: streamCacheBudgetBytes,
                requestedFrame: 0,
                appliedFrame: -1,
                droppedFrames: 0,
                availableFrames,
                preparedClient,
                preparedFrameCache: new Map(),
                preparedCacheBudgetBytes,
                preparedBytesEstimate: frameStrideBytes,
                preparedRequestMsEstimate: 0,
                preparedApplyMsEstimate: 0,
                preparedWarmupCompleted: false,
                preparedWarmupStatusFrameCount: 0,
                streamPreviewFrameRate: 12
            };

            events.fire('timeline.setFrames', manifest.frameCount);
            setPlaybackFrameRate(activeTrack);
            if (streamBinary) {
                const streamingPreview = availableFrames < manifest.frameCount;
                const suffix = streamingPreview && manifest.frameRate
                    ? `；边下边播预览 ${Math.min(manifest.frameRate, activeTrack.streamPreviewFrameRate ?? 12)} fps，完整缓存后恢复 ${manifest.frameRate} fps`
                    : '';
                const preparedSuffix = preparedClient ? '；full motion worker 已启用' : '；worker 不可用，回退主线程应用';
                events.fire('physics.status', `动画包已启用按需加载：每块 ${chunkFrameCount} 帧，raw 缓存约 ${Math.round(streamCacheBudgetBytes / 1024 / 1024)}MB，prepared 缓存约 ${Math.round(preparedCacheBudgetBytes / 1024 / 1024)}MB${suffix}${preparedSuffix}`);
            } else if (manifest.proxy && proxySkinning) {
                const proxyBytes = (proxyMotion?.byteLength ?? 0) + proxySkinning.byteLength;
                const suffix = proxyMotionUrl ? '；proxy motion 按需读取' : '';
                events.fire('physics.status', `Spark-style proxy 预览已启用：${manifest.proxy.groupCount} 组，轻量数据约 ${Math.max(1, Math.round(proxyBytes / 1024 / 1024))}MB${suffix}`);
            }
            await setFrame(0);
            if (activeTrack?.preparedClient) {
                schedulePreparedLookahead(activeTrack, 0);
            }
            events.fire('physmotion.loaded', activeTrack);

            return target;
        } finally {
            events.fire('stopSpinner');
        }
    });

    events.function('physmotion.clear', () => {
        disposeTrack(activeTrack);
        activeTrack = null;
        setFrameAgain = false;
    });

    events.function('phys.selectionSummary', (includeIndices = false) => {
        const splat = resolveTarget();
        if (!splat) {
            return null;
        }

        const state = splat.splatData.getProp('state') as Uint8Array;
        const x = splat.splatData.getProp('x') as Float32Array;
        const y = splat.splatData.getProp('y') as Float32Array;
        const z = splat.splatData.getProp('z') as Float32Array;
        const worldTransform = splat.worldTransform;
        const point = new Vec3();
        const world = new Vec3();
        const localMin = new Vec3(Infinity, Infinity, Infinity);
        const localMax = new Vec3(-Infinity, -Infinity, -Infinity);
        const worldMin = new Vec3(Infinity, Infinity, Infinity);
        const worldMax = new Vec3(-Infinity, -Infinity, -Infinity);
        const indices: number[] = [];
        let count = 0;

        for (let i = 0; i < state.length; ++i) {
            if (state[i] !== State.selected) {
                continue;
            }

            point.set(x[i], y[i], z[i]);
            worldTransform.transformPoint(point, world);

            localMin.x = Math.min(localMin.x, point.x);
            localMin.y = Math.min(localMin.y, point.y);
            localMin.z = Math.min(localMin.z, point.z);
            localMax.x = Math.max(localMax.x, point.x);
            localMax.y = Math.max(localMax.y, point.y);
            localMax.z = Math.max(localMax.z, point.z);
            worldMin.x = Math.min(worldMin.x, world.x);
            worldMin.y = Math.min(worldMin.y, world.y);
            worldMin.z = Math.min(worldMin.z, world.z);
            worldMax.x = Math.max(worldMax.x, world.x);
            worldMax.y = Math.max(worldMax.y, world.y);
            worldMax.z = Math.max(worldMax.z, world.z);

            if (includeIndices) {
                indices.push(i);
            }
            count++;
        }

        if (count === 0) {
            return {
                splat: splat.name,
                count: 0,
                indices: includeIndices ? [] : undefined
            };
        }

        const pack = (v: Vec3) => [v.x, v.y, v.z];
        return {
            splat: splat.name,
            count,
            indices: includeIndices ? indices : undefined,
            aabbLocal: {
                min: pack(localMin),
                max: pack(localMax)
            },
            aabbWorld: {
                min: pack(worldMin),
                max: pack(worldMax)
            },
            physGaussianPreprocessDraft: {
                sim_area: [worldMin.x, worldMax.x, worldMin.y, worldMax.y, worldMin.z, worldMax.z],
                scale: 1.0,
                rotation_degree: [0],
                rotation_axis: [0]
            }
        };
    });

    events.function('physmotion.extendAvailableFrames', (availableFrames: number) => {
        if (!activeTrack) {
            return 0;
        }
        const nextAvailable = Math.max(
            activeTrack.availableFrames,
            Math.min(Math.max(0, Math.floor(Number(availableFrames) || 0)), activeTrack.manifest.frameCount)
        );
        activeTrack.availableFrames = nextAvailable;
        activeTrack.preparedClient?.setAvailableFrames(nextAvailable);
        setPlaybackFrameRate(activeTrack);
        schedulePreparedLookahead(activeTrack, Math.max(0, Math.min(activeTrack.requestedFrame, nextAvailable - 1)));
        if (activeTrack.requestedFrame < nextAvailable) {
            void setFrame(activeTrack.requestedFrame);
        }
        return activeTrack.availableFrames;
    });

    events.function('physmotion.setFrameAsync', async (frame: number) => {
        await setFrame(frame);
        return activeTrack?.target ?? null;
    });

    events.on('timeline.frame', (frame: number) => {
        void setFrame(frame);
    });

    events.on('scene.clear', () => {
        disposeTrack(activeTrack);
        activeTrack = null;
        setFrameAgain = false;
    });
};

export { registerPhysMotionEvents, isPhysMotionManifest, PhysMotionManifest };
