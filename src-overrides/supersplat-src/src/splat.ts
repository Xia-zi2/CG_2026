import {
    ADDRESS_CLAMP_TO_EDGE,
    FILTER_NEAREST,
    PIXELFORMAT_R8,
    PIXELFORMAT_R16U,
    Asset,
    BoundingBox,
    Color,
    Entity,
    FloatPacking,
    GSplatData,
    GSplatResource,
    Mat4,
    Quat,
    Texture,
    Vec3
} from 'playcanvas';

import { Element, ElementType } from './element';
import { Serializer } from './serializer';
import { vertexShader, fragmentShader, gsplatCenter } from './shaders/splat-shader';
import { State, SplatState } from './splat-state';
import { Transform } from './transform';
import { TransformPalette } from './transform-palette';

const vec = new Vec3();
const veca = new Vec3();
const vecb = new Vec3();
const motionMin = new Vec3();
const motionMax = new Vec3();
const motionGroupCenter = new Vec3();
const motionGroupScale = new Vec3(1, 1, 1);
const motionGroupQuat = new Quat();
const motionRestInv = new Mat4();
const motionGroupTRS = new Mat4();
const motionGroupTransform = new Mat4();
const motionPatchTexturesParam = new URLSearchParams(window.location.search).get('physMotionPatchTextures')?.toLowerCase();
const motionPatchTexturesEnabled = motionPatchTexturesParam !== 'off';
const motionProfileEnabled = new URLSearchParams(window.location.search).get('physMotionProfile')?.toLowerCase() === 'on';

type MotionProxyPaletteFrame = {
    paletteStart: number;
    groupCount: number;
    restCenters: Float32Array;
    frame: Float32Array;
};

type MotionProxyPaletteBinding = {
    paletteStart: number;
    groupCount: number;
    previousTransformIndices: Uint16Array;
};

type MotionFrameData = {
    positions?: Float32Array;
    rotations?: Float32Array;
    scales?: Float32Array;
    indices?: Uint32Array;
    proxyPalette?: MotionProxyPaletteFrame;
    updateBounds?: boolean;
    refreshSorterMapping?: boolean;
};

type MotionPreparedRow = {
    y: number;
    height: number;
    data: Uint16Array | Uint32Array;
};

type MotionPreparedFrame = {
    frame: number;
    transformARows: MotionPreparedRow[];
    transformBRows: MotionPreparedRow[];
    centers?: Float32Array;
    bounds?: { min: [number, number, number]; max: [number, number, number] };
    bytes: number;
    pinned?: boolean;
    timings?: Record<string, number>;
    stats?: {
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
};

type MotionTextureSnapshot = {
    textureWidth: number;
    textureHeight: number;
    baseTransformA: ArrayBuffer;
    baseTransformB: ArrayBuffer;
};

const uploadMotionRows = async (
    texture: Texture,
    data: Uint16Array | Uint32Array,
    touchedRows: Uint8Array,
    width: number,
    components: number
) => {
    if (!(texture as any).impl?.write) {
        return false;
    }

    const writes: Promise<unknown>[] = [];
    let row = 0;
    while (row < touchedRows.length) {
        while (row < touchedRows.length && !touchedRows[row]) {
            row++;
        }
        const start = row;
        while (row < touchedRows.length && touchedRows[row]) {
            row++;
        }
        const count = row - start;
        if (count > 0) {
            const begin = start * width * components;
            const end = (start + count) * width * components;
            writes.push(Promise.resolve(texture.write(0, start, width, count, data.subarray(begin, end))));
        }
    }
    if (writes.length > 0) {
        await Promise.all(writes);
        return true;
    }
    return false;
};

const uploadPreparedRows = async (
    texture: Texture,
    rows: MotionPreparedRow[]
) => {
    if (!(texture as any).impl?.write) {
        return false;
    }

    let uploaded = false;
    for (const row of rows) {
        const height = Math.max(row.height, 1);
        const width = Math.floor(row.data.length / height / 4);
        if (width <= 0) {
            continue;
        }
        try {
            await Promise.resolve(texture.write(0, row.y, width, row.height, row.data));
            uploaded = true;
        } catch (error) {
            console.warn('[physmotion] prepared texture row upload failed', { y: row.y, height: row.height, width }, error);
            return false;
        }
    }
    return uploaded;
};

const boundingPoints =
    [-1, 1].map((x) => {
        return [-1, 1].map((y) => {
            return [-1, 1].map((z) => {
                return [
                    new Vec3(x, y, z), new Vec3(x * 0.75, y, z),
                    new Vec3(x, y, z), new Vec3(x, y * 0.75, z),
                    new Vec3(x, y, z), new Vec3(x, y, z * 0.75)
                ];
            });
        });
    }).flat(3);

class Splat extends Element {
    asset: Asset;
    splatData: GSplatData;
    numSplats = 0;
    numDeleted = 0;
    numLocked = 0;
    numSelected = 0;
    entity: Entity;
    changedCounter = 0;
    stateTexture: Texture;
    // encapsulates per-splat state mirror (cpu Uint8Array + gpu Texture).
    // all writes go through state.setBits/clearBits/toggleBits, then flush().
    state: SplatState;
    transformTexture: Texture;
    selectionBoundStorage: BoundingBox;
    localBoundStorage: BoundingBox;
    worldBoundStorage: BoundingBox;

    _visible = true;
    transformPalette: TransformPalette;

    selectionAlpha = 1;

    _name = '';
    _tintClr = new Color(1, 1, 1);
    _temperature = 0;
    _saturation = 1;
    _brightness = 0;
    _blackPoint = 0;
    _whitePoint = 1;
    _transparency = 1;

    measurePoints: Vec3[] = [];
    measureSelection = -1;

    rebuildMaterial: (bands: number) => void;

    constructor(asset: Asset, rotation: Quat) {
        super(ElementType.splat);

        const splatResource = asset.resource as GSplatResource;
        const splatData = splatResource.gsplatData;
        const { device } = splatResource;

        this._name = (asset.file as any).filename;
        this.asset = asset;
        this.splatData = splatData as GSplatData;
        this.numSplats = splatData.numSplats;

        this.entity = new Entity('splatEntitiy');
        this.entity.setLocalRotation(rotation);
        this.entity.addComponent('gsplat', { asset });

        const instance = this.entity.gsplat.instance;

        // added per-splat state channel
        // bit 1: selected
        // bit 2: deleted
        // bit 3: locked
        if (!this.splatData.getProp('state')) {
            this.splatData.getElement('vertex').properties.push({
                type: 'uchar',
                name: 'state',
                storage: new Uint8Array(this.splatData.numSplats),
                byteSize: 1
            });
        }

        // per-splat transform matrix
        this.splatData.getElement('vertex').properties.push({
            type: 'ushort',
            name: 'transform',
            storage: new Uint16Array(this.splatData.numSplats),
            byteSize: 2
        });

        const { x: width, y: height } = (splatResource as any).textureDimensions;

        // pack spherical harmonic data
        const createTexture = (name: string, format: number) => {
            return new Texture(device, {
                name: name,
                width: width,
                height: height,
                format: format,
                mipmaps: false,
                minFilter: FILTER_NEAREST,
                magFilter: FILTER_NEAREST,
                addressU: ADDRESS_CLAMP_TO_EDGE,
                addressV: ADDRESS_CLAMP_TO_EDGE
            });
        };

        // create the state texture and the SplatState mirror that owns it.
        // splatData.getProp('state') aliases state.data so existing read-only
        // consumers (serialize, status-bar, etc) keep working unchanged.
        this.stateTexture = createTexture('splatState', PIXELFORMAT_R8);
        this.state = new SplatState(this.splatData.getProp('state') as Uint8Array, this.stateTexture);
        this.transformTexture = createTexture('splatTransform', PIXELFORMAT_R16U);

        // create the transform palette
        this.transformPalette = new TransformPalette(device);

        this.rebuildMaterial = (bands: number) => {
            const { material } = instance;
            const { glsl } = material.shaderChunks;
            glsl.set('gsplatVS', vertexShader);
            glsl.set('gsplatPS', fragmentShader);
            glsl.set('gsplatCenterVS', gsplatCenter);

            material.setDefine('SH_BANDS', `${Math.min(bands, (instance.resource as GSplatResource).shBands)}`);
            material.setParameter('splatState', this.stateTexture);
            material.setParameter('splatTransform', this.transformTexture);
            material.update();
        };

        this.selectionBoundStorage = new BoundingBox();
        this.localBoundStorage = instance.resource.aabb;
        // @ts-ignore
        this.worldBoundStorage = instance.meshInstance._aabb;

        // @ts-ignore
        instance.meshInstance._updateAabb = false;

        // when sort changes, re-render the scene
        instance.sorter.on('updated', () => {
            this.changedCounter++;
        });
    }

    destroy() {
        super.destroy();
        this.entity.destroy();
        this.asset.registry.remove(this.asset);
        this.asset.unload();
    }

    async updateState(changedState = State.selected) {
        // uploads dirty range + refreshes counts in one pass.
        this.state.flush();
        this.numSplats = this.state.data.length - this.state.numDeleted;
        this.numLocked = this.state.numLocked;
        this.numSelected = this.state.numSelected;
        this.numDeleted = this.state.numDeleted;

        // handle splats being added or removed
        if (changedState & State.deleted) {
            await this.updateSorting();
        } else {
            await this.updateLocalBounds();
        }

        this.scene.forceRender = true;
        this.scene.events.fire('splat.stateChanged', this);
    }

    async updatePositions() {
        const data = await this.scene.dataProcessor.calcPositions(this);

        // update the splat centers which are used for render-time sorting
        const state = this.splatData.getProp('state') as Uint8Array;
        const { sorter } = this.entity.gsplat.instance;
        const { centers } = sorter;
        for (let i = 0; i < this.splatData.numSplats; ++i) {
            if (state[i] === State.selected) {
                centers[i * 3 + 0] = data[i * 4];
                centers[i * 3 + 1] = data[i * 4 + 1];
                centers[i * 3 + 2] = data[i * 4 + 2];
            }
        }

        await this.updateSorting();

        this.scene.forceRender = true;
        this.scene.events.fire('splat.positionsChanged', this);
    }

    async updateSorting() {
        const state = this.splatData.getProp('state') as Uint8Array;

        let mapping;

        // create a sorter mapping to remove deleted splats
        if (this.numSplats !== state.length) {
            mapping = new Uint32Array(this.numSplats);
            let idx = 0;
            for (let i = 0; i < state.length; ++i) {
                if ((state[i] & State.deleted) === 0) {
                    mapping[idx++] = i;
                }
            }
        }

        // update sorting instance
        this.entity.gsplat.instance.sorter.setMapping(mapping);

        // recalculate bounds after sorting changes
        await this.updateLocalBounds();
    }

    getMotionTextureSnapshot(): MotionTextureSnapshot | null {
        const resource = this.asset.resource as GSplatResource;
        const transformA = resource.getTexture('transformA');
        const transformB = resource.getTexture('transformB');
        if (!motionPatchTexturesEnabled || !transformA || !transformB || !(transformA as any).impl?.write || !(transformB as any).impl?.write) {
            return null;
        }

        const transformAData = transformA.getSource() as unknown as Uint32Array;
        const transformBData = transformB.getSource() as unknown as Uint16Array;
        if (!transformAData || !transformBData) {
            return null;
        }

        return {
            textureWidth: resource.textureDimensions.x,
            textureHeight: resource.textureDimensions.y,
            // TODO: avoid full transform texture snapshots; send only touched rows for dynamic splats.
            baseTransformA: new Uint32Array(transformAData).buffer,
            baseTransformB: new Uint16Array(transformBData).buffer
        };
    }

    async applyPreparedMotionFrame(frame: MotionPreparedFrame, indices?: Uint32Array) {
        const resource = this.asset.resource as GSplatResource;
        const sorter = this.entity.gsplat.instance.sorter;
        const transformA = resource.getTexture('transformA');
        const transformB = resource.getTexture('transformB');
        const start = motionProfileEnabled ? performance.now() : 0;

        const uploadedA = transformA && frame.transformARows.length > 0
            ? await uploadPreparedRows(transformA, frame.transformARows)
            : true;
        const uploadedB = transformB && frame.transformBRows.length > 0
            ? await uploadPreparedRows(transformB, frame.transformBRows)
            : true;

        if (!uploadedA || !uploadedB) {
            return false;
        }

        if (frame.centers) {
            const centers = sorter.centers as Float32Array;
            const count = indices ? indices.length : Math.min(this.splatData.numSplats, frame.centers.length / 3);
            for (let i = 0; i < count; ++i) {
                const id = indices ? indices[i] : i;
                const src = i * 3;
                const dst = id * 3;
                centers[dst] = frame.centers[src];
                centers[dst + 1] = frame.centers[src + 1];
                centers[dst + 2] = frame.centers[src + 2];
                resource.centers[dst] = frame.centers[src];
                resource.centers[dst + 1] = frame.centers[src + 1];
                resource.centers[dst + 2] = frame.centers[src + 2];
            }
        }
        if (frame.bounds && !indices) {
            motionMin.set(frame.bounds.min[0], frame.bounds.min[1], frame.bounds.min[2]);
            motionMax.set(frame.bounds.max[0], frame.bounds.max[1], frame.bounds.max[2]);
            this.localBoundStorage.setMinMax(motionMin, motionMax);
            this.updateWorldBound();
        }

        this.changedCounter++;
        this.scene.forceRender = true;
        this.scene.events.fire('splat.motionFrameApplied', this);

        if (motionProfileEnabled) {
            console.table({
                frame: frame.frame,
                preparedBytesMB: +(frame.bytes / 1024 / 1024).toFixed(2),
                uploadAndCentersMs: +(performance.now() - start).toFixed(2),
                workerDecodeMs: frame.timings?.decode ?? 0,
                workerPackMs: frame.timings?.pack ?? 0,
                touchedRows: frame.stats ? `${frame.stats.touchedRowCount}/${frame.stats.textureHeight}` : '',
                touchedRatio: frame.stats ? +frame.stats.touchedRowRatio.toFixed(3) : 0,
                rowRanges: frame.stats?.rowRangeCount ?? 0
            });
        }

        return true;
    }

    async applyMotionFrame(frame: MotionFrameData) {
        if (frame.proxyPalette) {
            this.applyMotionProxyPaletteFrame(frame.proxyPalette);
            return;
        }

        const profile = motionProfileEnabled ? {
            frame: -1,
            start: performance.now(),
            validate: 0,
            positionLoop: 0,
            centerSync: 0,
            rotationLoop: 0,
            scaleLoop: 0,
            textureUpload: 0,
            fallbackUpload: 0
        } : null;
        const mark = (key: string, start: number) => {
            if (profile) {
                (profile as any)[key] += performance.now() - start;
            }
        };

        const { positions, rotations, scales, indices } = frame;
        const splatData = this.splatData;
        const count = indices ? indices.length : splatData.numSplats;

        if (positions && positions.length < count * 3) {
            throw new Error(`motion positions buffer is too small: expected ${count * 3}, got ${positions.length}`);
        }
        if (rotations && rotations.length < count * 4) {
            throw new Error(`motion rotations buffer is too small: expected ${count * 4}, got ${rotations.length}`);
        }
        if (scales && scales.length < count * 3) {
            throw new Error(`motion scales buffer is too small: expected ${count * 3}, got ${scales.length}`);
        }
        if (profile) {
            profile.validate = performance.now() - profile.start;
        }

        const resource = this.asset.resource as GSplatResource;
        const sorter = this.entity.gsplat.instance.sorter;
        const centers = sorter.centers as Float32Array;
        const transformA = resource.getTexture('transformA');
        const transformB = resource.getTexture('transformB');
        const canPatchTransformTextures = motionPatchTexturesEnabled && !!indices && !!transformA && !!transformB && !!(transformA as any).impl?.write && !!(transformB as any).impl?.write;
        const textureWidth = resource.textureDimensions.x;
        const textureHeight = resource.textureDimensions.y;
        const transformAData = canPatchTransformTextures ? transformA.getSource() as unknown as Uint32Array : null;
        const transformAFloat = transformAData ? new Float32Array(transformAData.buffer) : null;
        const transformBData = canPatchTransformTextures ? transformB.getSource() as unknown as Uint16Array : null;
        const touchedRows = canPatchTransformTextures ? new Uint8Array(textureHeight) : null;

        if (positions) {
            const positionStart = performance.now();
            const x = splatData.getProp('x') as Float32Array;
            const y = splatData.getProp('y') as Float32Array;
            const z = splatData.getProp('z') as Float32Array;

            motionMin.set(Infinity, Infinity, Infinity);
            motionMax.set(-Infinity, -Infinity, -Infinity);

            for (let i = 0; i < count; ++i) {
                const id = indices ? indices[i] : i;
                const src = i * 3;
                const dst = id * 3;
                const px = positions[src];
                const py = positions[src + 1];
                const pz = positions[src + 2];

                x[id] = px;
                y[id] = py;
                z[id] = pz;
                centers[dst] = px;
                centers[dst + 1] = py;
                centers[dst + 2] = pz;
                if (transformAFloat && touchedRows) {
                    const texel = id * 4;
                    transformAFloat[texel] = px;
                    transformAFloat[texel + 1] = py;
                    transformAFloat[texel + 2] = pz;
                    touchedRows[Math.floor(id / textureWidth)] = 1;
                }

                motionMin.x = Math.min(motionMin.x, px);
                motionMin.y = Math.min(motionMin.y, py);
                motionMin.z = Math.min(motionMin.z, pz);
                motionMax.x = Math.max(motionMax.x, px);
                motionMax.y = Math.max(motionMax.y, py);
                motionMax.z = Math.max(motionMax.z, pz);
            }
            mark('positionLoop', positionStart);

            // Keep the resource-side cache in sync for consumers that do not read the sorter copy.
            // Indexed motion packages only touch a subset, so avoid copying the whole scene every frame.
            const centerStart = performance.now();
            if (indices) {
                for (let i = 0; i < count; ++i) {
                    const id = indices[i] * 3;
                    const src = i * 3;
                    resource.centers[id] = positions[src];
                    resource.centers[id + 1] = positions[src + 1];
                    resource.centers[id + 2] = positions[src + 2];
                }
            } else {
                resource.centers.set(centers);
            }
            mark('centerSync', centerStart);
        }

        if (rotations) {
            const rotationStart = performance.now();
            const r0 = splatData.getProp('rot_0') as Float32Array;
            const r1 = splatData.getProp('rot_1') as Float32Array;
            const r2 = splatData.getProp('rot_2') as Float32Array;
            const r3 = splatData.getProp('rot_3') as Float32Array;
            for (let i = 0; i < count; ++i) {
                const id = indices ? indices[i] : i;
                const src = i * 4;
                r0[id] = rotations[src];
                r1[id] = rotations[src + 1];
                r2[id] = rotations[src + 2];
                r3[id] = rotations[src + 3];
                if (transformAData && transformBData && touchedRows) {
                    let w = rotations[src];
                    let x = rotations[src + 1];
                    let y = rotations[src + 2];
                    let z = rotations[src + 3];
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
                    const texel = id * 4;
                    transformAData[texel + 3] = FloatPacking.float2Half(x) | FloatPacking.float2Half(y) << 16;
                    transformBData[texel + 3] = FloatPacking.float2Half(z);
                    touchedRows[Math.floor(id / textureWidth)] = 1;
                }
            }
            mark('rotationLoop', rotationStart);
        }

        if (scales) {
            const scaleStart = performance.now();
            const s0 = splatData.getProp('scale_0') as Float32Array;
            const s1 = splatData.getProp('scale_1') as Float32Array;
            const s2 = splatData.getProp('scale_2') as Float32Array;
            for (let i = 0; i < count; ++i) {
                const id = indices ? indices[i] : i;
                const src = i * 3;
                s0[id] = scales[src];
                s1[id] = scales[src + 1];
                s2[id] = scales[src + 2];
                if (transformBData && touchedRows) {
                    const texel = id * 4;
                    transformBData[texel] = FloatPacking.float2Half(Math.exp(scales[src]));
                    transformBData[texel + 1] = FloatPacking.float2Half(Math.exp(scales[src + 1]));
                    transformBData[texel + 2] = FloatPacking.float2Half(Math.exp(scales[src + 2]));
                    touchedRows[Math.floor(id / textureWidth)] = 1;
                }
            }
            mark('scaleLoop', scaleStart);
        }

        const uploadStart = performance.now();
        const patchedTransformTextures = transformAData && transformBData && touchedRows
            ? (await uploadMotionRows(transformA, transformAData, touchedRows, textureWidth, 4))
                && (await uploadMotionRows(transformB, transformBData, touchedRows, textureWidth, 4))
            : false;
        mark('textureUpload', uploadStart);
        if (!patchedTransformTextures) {
            const fallbackStart = performance.now();
            resource.updateTransformData(splatData);
            mark('fallbackUpload', fallbackStart);
        }
        if (frame.refreshSorterMapping) {
            sorter.setMapping(null);
        }

        if (positions && frame.updateBounds !== false && !indices) {
            this.localBoundStorage.setMinMax(motionMin, motionMax);
            this.updateWorldBound();
        }

        this.changedCounter++;
        this.scene.forceRender = true;
        this.scene.events.fire('splat.motionFrameApplied', this);

        if (profile) {
            console.table({
                validateMs: +profile.validate.toFixed(2),
                positionLoopMs: +profile.positionLoop.toFixed(2),
                centerSyncMs: +profile.centerSync.toFixed(2),
                rotationLoopMs: +profile.rotationLoop.toFixed(2),
                scaleLoopMs: +profile.scaleLoop.toFixed(2),
                textureUploadMs: +profile.textureUpload.toFixed(2),
                fallbackUploadMs: +profile.fallbackUpload.toFixed(2),
                totalMs: +(performance.now() - profile.start).toFixed(2),
                count,
                patchTexture: !!patchedTransformTextures
            });
        }
    }

    setupMotionProxyPalette(indices: Uint32Array, groupIds: Uint32Array, groupCount: number): MotionProxyPaletteBinding {
        if (indices.length !== groupIds.length) {
            throw new Error(`proxy palette setup expected ${indices.length} group ids, got ${groupIds.length}`);
        }

        const paletteStart = this.transformPalette.alloc(groupCount);
        if (paletteStart + groupCount >= 65536) {
            throw new Error(`proxy palette needs ${groupCount} transform slots, but R16 transform indices are exhausted`);
        }

        const previousTransformIndices = new Uint16Array(indices.length);
        const transformIndices = this.transformTexture.lock() as Uint16Array;
        for (let i = 0; i < indices.length; ++i) {
            const id = indices[i];
            if (id >= transformIndices.length) {
                continue;
            }
            previousTransformIndices[i] = transformIndices[id];
            transformIndices[id] = paletteStart + Math.min(groupIds[i], groupCount - 1);
        }
        this.transformTexture.unlock();

        for (let group = 0; group < groupCount; ++group) {
            this.transformPalette.setTransform(paletteStart + group, Mat4.IDENTITY, false);
        }
        this.transformPalette.upload();

        this.scene.forceRender = true;
        return { paletteStart, groupCount, previousTransformIndices };
    }

    restoreMotionProxyPalette(indices: Uint32Array, previousTransformIndices: Uint16Array, groupCount = 0) {
        if (!indices || !previousTransformIndices) {
            return;
        }

        const transformIndices = this.transformTexture.lock() as Uint16Array;
        const count = Math.min(indices.length, previousTransformIndices.length);
        for (let i = 0; i < count; ++i) {
            const id = indices[i];
            if (id < transformIndices.length) {
                transformIndices[id] = previousTransformIndices[i];
            }
        }
        this.transformTexture.unlock();
        if (groupCount > 0) {
            this.transformPalette.free(groupCount);
        }
        this.scene.forceRender = true;
    }

    applyMotionProxyPaletteFrame(proxy: MotionProxyPaletteFrame) {
        const { paletteStart, groupCount, restCenters, frame } = proxy;
        const rotationOffset = groupCount * 3;

        for (let group = 0; group < groupCount; ++group) {
            const centerOffset = group * 3;
            const quatOffset = rotationOffset + group * 4;
            const cx = frame[centerOffset];
            const cy = frame[centerOffset + 1];
            const cz = frame[centerOffset + 2];
            const rx = restCenters[centerOffset];
            const ry = restCenters[centerOffset + 1];
            const rz = restCenters[centerOffset + 2];
            const qw = frame[quatOffset];
            const qx = frame[quatOffset + 1];
            const qy = frame[quatOffset + 2];
            const qz = frame[quatOffset + 3];

            motionGroupCenter.set(cx, cy, cz);
            motionGroupQuat.set(qx, qy, qz, qw).normalize();
            motionGroupTRS.setTRS(motionGroupCenter, motionGroupQuat, motionGroupScale);
            motionRestInv.setTranslate(-rx, -ry, -rz);
            motionGroupTransform.mul2(motionGroupTRS, motionRestInv);
            this.transformPalette.setTransform(paletteStart + group, motionGroupTransform, false);
        }

        this.transformPalette.upload();
        this.changedCounter++;
        this.scene.forceRender = true;
        this.scene.events.fire('splat.motionFrameApplied', this);
    }

    get worldTransform() {
        return this.entity.getWorldTransform();
    }

    set name(newName: string) {
        if (newName !== this.name) {
            this._name = newName;
            this.scene.events.fire('splat.name', this);
        }
    }

    get name() {
        return this._name;
    }

    get filename() {
        return (this.asset.file as any).filename;
    }

    calcSplatWorldPosition(splatId: number, result: Vec3) {
        if (splatId >= this.splatData.numSplats) {
            return false;
        }

        // use centers data, which are updated when edits occur
        const { sorter } = this.entity.gsplat.instance;
        const { centers } = sorter;

        result.set(
            centers[splatId * 3 + 0],
            centers[splatId * 3 + 1],
            centers[splatId * 3 + 2]
        );

        this.worldTransform.transformPoint(result, result);

        return true;
    }

    async add() {
        // add the entity to the scene
        this.scene.contentRoot.addChild(this.entity);

        // assign splat to the dedicated splat layer (rendered by splat camera with MRT)
        this.entity.gsplat.layers = [this.scene.splatLayer.id];

        this.scene.events.on('view.bands', this.rebuildMaterial, this);
        this.rebuildMaterial(this.scene.events.invoke('view.bands'));

        // we must update state in case the state data was loaded from ply
        await this.updateState();
    }

    remove() {
        this.scene.events.off('view.bands', this.rebuildMaterial, this);

        this.scene.contentRoot.removeChild(this.entity);
        this.scene.boundDirty = true;
    }

    serialize(serializer: Serializer) {
        serializer.packa(this.entity.getWorldTransform().data);
        serializer.pack(this.changedCounter);
        serializer.pack(this.visible);
        serializer.pack(this.tintClr.r, this.tintClr.g, this.tintClr.b);
        serializer.pack(this.temperature, this.saturation, this.brightness, this.blackPoint, this.whitePoint, this.transparency);
    }

    onPreRender() {
        const events = this.scene.events;
        const selected = this.scene.camera.renderOverlays && events.invoke('selection') === this;
        const cameraMode = events.invoke('camera.mode');
        const cameraOverlay = events.invoke('camera.overlay');

        // configure rings rendering
        const material = this.entity.gsplat.instance.material;
        material.setParameter('outlineMode', events.invoke('view.outlineSelection') ? 1 : 0);
        material.setParameter('ringSize', (selected && cameraOverlay && cameraMode === 'rings') ? 0.04 : 0);

        // configure colors
        const selectedClr = events.invoke('selectedClr');
        const unselectedClr = events.invoke('unselectedClr');
        const lockedClr = events.invoke('lockedClr');

        if (!selected) {
            material.setParameter('selectedClr', [0, 0, 0, 0]);
        } else if (events.invoke('view.outlineSelection')) {
            material.setParameter('selectedClr', [0, 0, 0, 0]);
        } else {
            material.setParameter('selectedClr', [selectedClr.r, selectedClr.g, selectedClr.b, selectedClr.a * this.selectionAlpha]);
        }
        material.setParameter('unselectedClr', [unselectedClr.r, unselectedClr.g, unselectedClr.b, unselectedClr.a]);
        material.setParameter('lockedClr', [lockedClr.r, lockedClr.g, lockedClr.b, lockedClr.a]);

        // combine black pointer, white point and brightness
        const offset = -this.blackPoint + this.brightness;
        const scale = 1 / (this.whitePoint - this.blackPoint);

        material.setParameter('clrOffset', [offset, offset, offset]);
        material.setParameter('clrScale', [
            scale * this.tintClr.r * (1 + this.temperature),
            scale * this.tintClr.g,
            scale * this.tintClr.b * (1 - this.temperature),
            this.transparency
        ]);

        material.setParameter('saturation', this.saturation);
        material.setParameter('transformPalette', this.transformPalette.texture);

        if (this.visible && selected) {
            // render bounding box
            if (events.invoke('camera.bound')) {
                const bound = this.localBound;
                const scale = new Mat4().setTRS(bound.center, Quat.IDENTITY, bound.halfExtents);
                scale.mul2(this.entity.getWorldTransform(), scale);

                for (let i = 0; i < boundingPoints.length / 2; i++) {
                    const a = boundingPoints[i * 2];
                    const b = boundingPoints[i * 2 + 1];
                    scale.transformPoint(a, veca);
                    scale.transformPoint(b, vecb);

                    this.scene.app.drawLine(veca, vecb, Color.WHITE, true, this.scene.worldLayer);
                }
            }
        }

        this.entity.enabled = this.visible;
    }

    focalPoint() {
        // GSplatData has a function for calculating an weighted average of the splat positions
        // to get a focal point for the camera, but we use bound center instead
        return this.worldBound.center;
    }

    move(position?: Vec3, rotation?: Quat, scale?: Vec3) {
        const entity = this.entity;
        if (position) {
            entity.setLocalPosition(position);
        }
        if (rotation) {
            entity.setLocalRotation(rotation);
        }
        if (scale) {
            entity.setLocalScale(scale);
        }

        this.updateWorldBound();

        this.scene.events.fire('splat.moved', this);
    }

    // calculate both selection and local bounds (async, callers must await)
    async updateLocalBounds(): Promise<void> {
        await this.scene.dataProcessor.calcBound(this, this.selectionBoundStorage, this.localBoundStorage);
        this.updateWorldBound();
    }

    // update world bound from local bound (synchronous)
    private updateWorldBound() {
        this.worldBoundStorage.setFromTransformedAabb(this.localBoundStorage, this.entity.getWorldTransform());
        this.scene.boundDirty = true;
    }

    // get the selection bound
    get selectionBound() {
        return this.selectionBoundStorage;
    }

    // get local space bound
    get localBound() {
        return this.localBoundStorage;
    }

    // get world space bound
    get worldBound() {
        return this.worldBoundStorage;
    }

    set visible(value: boolean) {
        if (value !== this.visible) {
            this._visible = value;
            this.scene?.events.fire('splat.visibility', this);
        }
    }

    get visible() {
        return this._visible;
    }

    set tintClr(value: Color) {
        if (!this._tintClr.equals(value)) {
            this._tintClr.set(value.r, value.g, value.b);
            this.scene.events.fire('splat.tintClr', this);
        }
    }

    get tintClr() {
        return this._tintClr;
    }

    set temperature(value: number) {
        if (value !== this._temperature) {
            this._temperature = value;
            this.scene.events.fire('splat.temperature', this);
        }
    }

    get temperature() {
        return this._temperature;
    }

    set saturation(value: number) {
        if (value !== this._saturation) {
            this._saturation = value;
            this.scene.events.fire('splat.saturation', this);
        }
    }

    get saturation() {
        return this._saturation;
    }

    set brightness(value: number) {
        if (value !== this._brightness) {
            this._brightness = value;
            this.scene.events.fire('splat.brightness', this);
        }
    }

    get brightness() {
        return this._brightness;
    }

    set blackPoint(value: number) {
        if (value !== this._blackPoint) {
            this._blackPoint = value;
            this.scene.events.fire('splat.blackPoint', this);
        }
    }

    get blackPoint() {
        return this._blackPoint;
    }

    set whitePoint(value: number) {
        if (value !== this._whitePoint) {
            this._whitePoint = value;
            this.scene.events.fire('splat.whitePoint', this);
        }
    }

    get whitePoint() {
        return this._whitePoint;
    }

    set transparency(value: number) {
        if (value !== this._transparency) {
            this._transparency = value;
            this.scene.events.fire('splat.transparency', this);
        }
    }

    get transparency() {
        return this._transparency;
    }

    // get pivot position/rotation/scale (caller should have awaited operation that changed data)
    getPivot(mode: 'center' | 'boundCenter', selection: boolean, result: Transform) {
        const { entity } = this;
        switch (mode) {
            case 'center':
                result.set(entity.getLocalPosition(), entity.getLocalRotation(), entity.getLocalScale());
                break;
            case 'boundCenter': {
                const bound = selection ? this.selectionBound : this.localBound;
                entity.getLocalTransform().transformPoint(bound.center, vec);
                result.set(vec, entity.getLocalRotation(), entity.getLocalScale());
                break;
            }
        }
    }

    docSerialize() {
        const pack3 = (v: Vec3) => [v.x, v.y, v.z];
        const pack4 = (q: Quat) => [q.x, q.y, q.z, q.w];
        const packC = (c: Color) => [c.r, c.g, c.b, c.a];
        return {
            name: this.name,
            position: pack3(this.entity.getLocalPosition()),
            rotation: pack4(this.entity.getLocalRotation()),
            scale: pack3(this.entity.getLocalScale()),
            visible: this.visible,
            tintClr: packC(this.tintClr),
            temperature: this.temperature,
            saturation: this.saturation,
            brightness: this.brightness,
            blackPoint: this.blackPoint,
            whitePoint: this.whitePoint,
            transparency: this.transparency
        };
    }

    docDeserialize(doc: any) {
        const { name, position, rotation, scale, visible, tintClr, temperature, saturation, brightness, blackPoint, whitePoint, transparency } = doc;

        this.name = name;
        this.move(new Vec3(position), new Quat(rotation), new Vec3(scale));
        this.visible = visible;
        this.tintClr = new Color(tintClr[0], tintClr[1], tintClr[2], tintClr[3]);
        this.temperature = temperature ?? 0;
        this.saturation = saturation ?? 1;
        this.brightness = brightness;
        this.blackPoint = blackPoint;
        this.whitePoint = whitePoint;
        this.transparency = transparency;
    }
}

export { Splat };
export type { MotionPreparedFrame, MotionTextureSnapshot };
