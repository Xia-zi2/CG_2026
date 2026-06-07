import { Events } from './events';
import { Splat } from './splat';
import { State } from './splat-state';
import { Quat, Vec3 } from 'playcanvas';

type MaterialPreset = 'jelly' | 'metal' | 'sand' | 'foam' | 'snow' | 'plasticine' | 'rigid' | 'obstacle';

type ObjectDrive = {
    linearEnabled: boolean;
    linearForce: number[];
    linearNumDt: number;
    linearStart: number;
    spinEnabled: boolean;
    spinAxis: number[];
    spinAngular: number;
    spinTranslation: number;
    spinStart: number;
    spinEnd: number;
};

type Bounds3 = {
    min: number[];
    max: number[];
};

type BoundaryConstraintPreview = {
    type: 'velocity_cuboid';
    constraint_type: 'set_velocity_on_cuboid';
    source: 'obstacle_part';
    objectId: number;
    bodyId: number;
    coordinateSpace: 'mpm_shifted_normalized';
    sceneBounds: Bounds3;
    simArea: number[];
    mpmBounds: Bounds3;
    paddedMpmBounds: Bounds3;
    point: number[];
    size: number[];
    scale: number[];
    velocity: number[];
    start_time: number;
    end_time: number;
    reset: number;
    gridDx: number;
    padding: number;
    paddingGridNodes: number;
    estimatedGridNodeCounts: number[];
    estimatedGridNodeCount: number;
    gridIndexBounds: Bounds3;
};

type PhysicsObject = {
    objectId: number;
    bodyId: number;
    name: string;
    material: MaterialPreset;
    density: number;
    E: number;
    nu: number;
    mode: 'mpm' | 'rigid-soft' | 'rigid-body' | 'obstacle';
    constraintMode?: 'grid-velocity-cuboid';
    cuboidPaddingGridNodes?: number;
    fill: boolean;
    drive: ObjectDrive;
    count: number;
    indices: number[];
    aabbWorld?: {
        min: number[];
        max: number[];
    };
    aabbLocal?: {
        min: number[];
        max: number[];
    };
    boundaryDebug?: BoundaryConstraintPreview | null;
};

type PhysicsPayload = {
    version: 'phys-ui-v1';
    solver: 'explicit-mpm' | 'implicit-mpm' | 'pbmpm' | 'finite-rigid-body' | 'local-global-g2p' | 'implicit-object-energy';
    source: {
        splat: string;
        numSplats: number;
    };
    modelId?: string;
    officialConfig: {
        enabled: boolean;
        available?: boolean;
        name?: string;
        values?: any;
    };
    preprocessing: {
        rotation_degree: number[];
        rotation_axis: number[];
        sim_area: number[] | null;
        scale: number;
        n_grid: number;
        opacity_threshold: number;
    };
    simulation: {
        gravityEnabled: boolean;
        gravity: number[];
        groundEnabled: boolean;
        groundHeight: number;
        boundingBoxEnabled: boolean;
        frame_dt: number;
        frame_num: number;
        substep_dt: number;
        damping?: number;
        implicitBeta?: number;
        implicitGamma?: number;
        newtonTol?: number;
        newtonAbsTol?: number;
        newtonMaxIter?: number;
        gmresTol?: number;
        gmresMaxIter?: number;
        jvpEps?: number;
        lineSearchMaxIter?: number;
        armijoC1?: number;
        ewEtaMin?: number;
        ewEtaMax?: number;
        ewGamma?: number;
        ewAlpha?: number;
        stiffnessPreconditionerScale?: number;
        stagnationTol?: number;
        pbmpm?: {
            strength_scale?: number;
            stiffness_scale?: number;
            n_min?: number;
            n_max?: number;
            plastic_mode?: number;
            yield_min?: number;
            yield_max?: number;
        };
        preview?: {
            dragHitIndex?: number;
            dragVelocity?: number[];
            dragObjectId?: number;
            dragBodyId?: number;
            targetVoxelGroups?: number;
            voxelSize?: number;
            gridResolution?: number;
            adaptiveGrid?: boolean;
            gridCellScale?: number;
            proxyBlendCount?: number;
            proxyBlendPower?: number;
            velocitySpreadIterations?: number;
            velocitySpreadStrength?: number;
            shapeCoupling?: number;
            shapeStiffness?: number;
        };
    };
    boundaryConstraints: BoundaryConstraintPreview[];
    objects: PhysicsObject[];
    selectedCount: number;
};

const eulerScratch = new Vec3();
const pointScratch = new Vec3();
const rotatedScratch = new Vec3();
const physicsRotationScratch = new Quat();
const mpmGridLim = 2.0;

const materialDefaults: Record<MaterialPreset, { E: number, nu: number, density: number, mode: PhysicsObject['mode'] }> = {
    jelly: { E: 1e5, nu: 0.3, density: 200, mode: 'mpm' },
    metal: { E: 2e6, nu: 0.3, density: 2700, mode: 'mpm' },
    sand: { E: 1e5, nu: 0.2, density: 1500, mode: 'mpm' },
    foam: { E: 1e4, nu: 0.1, density: 80, mode: 'mpm' },
    snow: { E: 1.4e5, nu: 0.2, density: 400, mode: 'mpm' },
    plasticine: { E: 4e4, nu: 0.35, density: 1300, mode: 'mpm' },
    rigid: { E: 1e7, nu: 0.25, density: 1000, mode: 'rigid-soft' },
    obstacle: { E: 2e6, nu: 0.4, density: 200, mode: 'obstacle' }
};

const defaultDrive = (): ObjectDrive => ({
    linearEnabled: false,
    linearForce: [0, 0, 0],
    linearNumDt: 1,
    linearStart: 0,
    spinEnabled: false,
    spinAxis: [0, 0, 1],
    spinAngular: 0,
    spinTranslation: 0,
    spinStart: 0,
    spinEnd: 0.2
});

const selectedIndicesFor = (splat: Splat) => {
    const state = splat.splatData.getProp('state') as Uint8Array;
    const result: number[] = [];
    for (let i = 0; i < state.length; ++i) {
        if (state[i] === State.selected) {
            result.push(i);
        }
    }
    return result;
};

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

type ProxyGroup = {
    indices: number[];
    center: number[];
};

type ProxyBounds = {
    mins: number[];
    maxs: number[];
};

type DragPreviewRequest = {
    hitIndex?: number;
    velocity: number[];
    objectId?: number;
    bodyId?: number;
    screenStart?: number[];
    screenEnd?: number[];
    duration?: number;
    canvasSize?: number[];
    dragGain?: number;
};

type DragProjection = {
    displacement: number[];
    displacementWorld: number[];
    velocity: number[];
    velocityWorld: number[];
    startWorld: number[] | null;
    endWorld: number[] | null;
    mode: string;
    depth: number | null;
    worldPerPixel: number | null;
};

type ProxyMaterial = {
    E: number;
    nu: number;
    density: number;
    rigidWeight: number;
    stressScale: number;
};

type LocalProxyFrame = {
    positions: Float32Array;
    rotations: Float32Array;
    scales: Float32Array;
    indices: Uint32Array;
    updateBounds: false;
};

type LocalProxyState = {
    splat: Splat;
    indices: number[];
    indexArray: Uint32Array;
    groups: ProxyGroup[];
    groupByIndex: Map<number, number>;
    restCenters: number[][];
    centers: number[][];
    velocities: number[][];
    affines: number[][];
    deformations: number[][];
    rotations: number[][];
    masses: number[];
    materials: ProxyMaterial[];
    bounds: ProxyBounds;
    voxelSize: number;
    gridRes: number;
    blendCount: number;
    skinning: Map<number, [number, number][]>;
    restPositions: Float32Array;
    restRotations: Float32Array;
    restScales: Float32Array;
    hitGroup: number | null;
    kinematicGroups: Set<number>;
    gravity: number[];
    groundZ: number | null;
    shapeStiffness: number;
    maxDriveSpeed: number;
    dragDisplacement: number[];
    lastTime: number;
    liveBusy: boolean;
};

type LocalProxyTemplate = {
    bodyId: number;
    indices: number[];
    groups: ProxyGroup[];
    groupByIndex: Map<number, number>;
    materials: ProxyMaterial[];
    masses: number[];
    kinematicGroups: Set<number>;
    bounds: ProxyBounds;
    voxelSize: number;
    gridRes: number;
    blendCount: number;
    skinning: Map<number, [number, number][]>;
    shapeStiffness: number;
};

const vec3Length = (value: number[]) => Math.hypot(value[0] ?? 0, value[1] ?? 0, value[2] ?? 0);

const normalizeQuatValues = (values: number[]) => {
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

const clampVectorLength = (value: number[], maxLength: number) => {
    const length = vec3Length(value);
    if (length <= maxLength || length <= 1e-12) {
        return value.slice(0, 3);
    }
    const scale = maxLength / length;
    return [value[0] * scale, value[1] * scale, value[2] * scale];
};

const normalizedPoseVector = (value: any, fallback: number[]) => {
    const result = [
        Number(value?.x),
        Number(value?.y),
        Number(value?.z)
    ];
    if (!result.every(Number.isFinite) || vec3Length(result) <= 1e-8) {
        return fallback.slice(0, 3);
    }
    const inv = 1 / Math.max(vec3Length(result), 1e-12);
    return [result[0] * inv, result[1] * inv, result[2] * inv];
};

const clampPointToProxyBounds = (point: number[], bounds: ProxyBounds, marginRatio = 0.08) => {
    const span = Math.max(
        bounds.maxs[0] - bounds.mins[0],
        bounds.maxs[1] - bounds.mins[1],
        bounds.maxs[2] - bounds.mins[2],
        1e-5
    );
    const pad = span * Math.max(0, marginRatio);
    return [
        Math.max(bounds.mins[0] - pad, Math.min(bounds.maxs[0] + pad, point[0])),
        Math.max(bounds.mins[1] - pad, Math.min(bounds.maxs[1] + pad, point[1])),
        Math.max(bounds.mins[2] - pad, Math.min(bounds.maxs[2] + pad, point[2]))
    ];
};

const proxyBoundsForIndices = (
    x: Float32Array,
    y: Float32Array,
    z: Float32Array,
    indices: number[]
): ProxyBounds => {
    const mins = [Infinity, Infinity, Infinity];
    const maxs = [-Infinity, -Infinity, -Infinity];
    for (const index of indices) {
        mins[0] = Math.min(mins[0], x[index]);
        mins[1] = Math.min(mins[1], y[index]);
        mins[2] = Math.min(mins[2], z[index]);
        maxs[0] = Math.max(maxs[0], x[index]);
        maxs[1] = Math.max(maxs[1], y[index]);
        maxs[2] = Math.max(maxs[2], z[index]);
    }
    return { mins, maxs };
};

const buildLocalProxyGroups = (
    x: Float32Array,
    y: Float32Array,
    z: Float32Array,
    indices: number[],
    targetGroups = 192
) => {
    const bounds = proxyBoundsForIndices(x, y, z, indices);
    const extent = [
        Math.max(bounds.maxs[0] - bounds.mins[0], 1e-8),
        Math.max(bounds.maxs[1] - bounds.mins[1], 1e-8),
        Math.max(bounds.maxs[2] - bounds.mins[2], 1e-8)
    ];
    const maxExtent = Math.max(extent[0], extent[1], extent[2], 1e-8);
    const divisions = Math.max(1, Math.round(Math.max(8, Math.min(targetGroups, 4096)) ** (1 / 3)));
    const voxelSize = Math.max(maxExtent / divisions, maxExtent / 256, 1e-8);
    const buckets = new Map<string, number[]>();

    for (const index of indices) {
        const key = [
            Math.floor((x[index] - bounds.mins[0]) / voxelSize),
            Math.floor((y[index] - bounds.mins[1]) / voxelSize),
            Math.floor((z[index] - bounds.mins[2]) / voxelSize)
        ].join(',');
        const bucket = buckets.get(key);
        if (bucket) {
            bucket.push(index);
        } else {
            buckets.set(key, [index]);
        }
    }

    const groups: ProxyGroup[] = [];
    const groupByIndex = new Map<number, number>();
    for (const groupIndices of buckets.values()) {
        const center = [0, 0, 0];
        for (const index of groupIndices) {
            center[0] += x[index];
            center[1] += y[index];
            center[2] += z[index];
        }
        center[0] /= groupIndices.length;
        center[1] /= groupIndices.length;
        center[2] /= groupIndices.length;
        const groupId = groups.length;
        for (const index of groupIndices) {
            groupByIndex.set(index, groupId);
        }
        groups.push({ indices: groupIndices, center });
    }

    return { groups, groupByIndex, bounds, voxelSize };
};

const buildLocalProxyGrouping = (
    x: Float32Array,
    y: Float32Array,
    z: Float32Array,
    indices: number[],
    fixedIndexSet: Set<number>,
    targetGroups = 192
) => {
    const movableIndices = indices.filter(index => !fixedIndexSet.has(index));
    const fixedIndices = indices.filter(index => fixedIndexSet.has(index));
    const groups: ProxyGroup[] = [];
    const groupByIndex = new Map<number, number>();
    const kinematicGroups = new Set<number>();
    const voxelSizes: number[] = [];

    const appendGroups = (grouping: ReturnType<typeof buildLocalProxyGroups>, kinematic: boolean) => {
        const offset = groups.length;
        for (const group of grouping.groups) {
            const groupId = groups.length;
            groups.push(group);
            if (kinematic) {
                kinematicGroups.add(groupId);
            }
        }
        for (const [index, groupId] of grouping.groupByIndex) {
            groupByIndex.set(index, offset + groupId);
        }
        voxelSizes.push(grouping.voxelSize);
    };

    if (movableIndices.length > 0) {
        appendGroups(buildLocalProxyGroups(x, y, z, movableIndices, targetGroups), false);
    }
    if (fixedIndices.length > 0) {
        appendGroups(buildLocalProxyGroups(x, y, z, fixedIndices, targetGroups), true);
    }

    const bounds = proxyBoundsForIndices(x, y, z, indices);
    const voxelSize = Math.min(...voxelSizes.filter(value => Number.isFinite(value) && value > 0));
    return {
        groups,
        groupByIndex,
        bounds,
        voxelSize: Number.isFinite(voxelSize) ? voxelSize : 1e-4,
        kinematicGroups
    };
};

const localProxyGridResolution = (groupCount: number, voxelSize: number, bounds: ProxyBounds) => {
    const maxExtent = Math.max(
        bounds.maxs[0] - bounds.mins[0],
        bounds.maxs[1] - bounds.mins[1],
        bounds.maxs[2] - bounds.mins[2],
        1e-6
    );
    const bySpacing = Math.round(maxExtent * 1.24 / Math.max(voxelSize * 1.15, 1e-6)) + 1;
    const byCount = Math.ceil(Math.max(groupCount, 1) ** (1 / 3)) + 3;
    return Math.max(5, Math.min(32, Math.max(bySpacing, byCount)));
};

const localProxySkinning = (
    x: Float32Array,
    y: Float32Array,
    z: Float32Array,
    indices: number[],
    centers: number[][],
    groupByIndex: Map<number, number>,
    blendCount: number,
    power: number
) => {
    blendCount = Math.max(1, Math.min(blendCount, Math.min(8, centers.length)));
    power = Math.max(0.5, Math.min(power, 6));
    const eps = 1e-12;
    const result = new Map<number, [number, number][]>();
    for (const index of indices) {
        const distances: [number, number][] = [];
        for (let groupId = 0; groupId < centers.length; ++groupId) {
            const center = centers[groupId];
            const dx = x[index] - center[0];
            const dy = y[index] - center[1];
            const dz = z[index] - center[2];
            distances.push([dx * dx + dy * dy + dz * dz, groupId]);
        }
        distances.sort((a, b) => a[0] - b[0]);
        const selected = distances.slice(0, blendCount);
        const ownGroup = groupByIndex.get(index);
        if (ownGroup !== undefined && !selected.some((entry) => entry[1] === ownGroup)) {
            const center = centers[ownGroup];
            const dx = x[index] - center[0];
            const dy = y[index] - center[1];
            const dz = z[index] - center[2];
            selected[selected.length - 1] = [dx * dx + dy * dy + dz * dz, ownGroup];
        }
        const raw = selected.map(([distance]) => (distance + eps) ** (-0.5 * power));
        const sum = Math.max(raw.reduce((accum, value) => accum + value, 0), eps);
        result.set(index, selected.map((entry, i) => [entry[1], raw[i] / sum]));
    }
    return result;
};

const quatFromAngularVelocity = (omega: number[], dt: number): number[] => {
    const angle = vec3Length(omega) * dt;
    if (angle <= 1e-8) {
        return [1, 0, 0, 0];
    }
    const invLength = 1 / Math.max(vec3Length(omega), 1e-12);
    const half = angle * 0.5;
    const s = Math.sin(half);
    return [
        Math.cos(half),
        omega[0] * invLength * s,
        omega[1] * invLength * s,
        omega[2] * invLength * s
    ];
};

const quatMulArray = (a: number[], b: number[]) => {
    const aw = a[0], ax = a[1], ay = a[2], az = a[3];
    const bw = b[0], bx = b[1], by = b[2], bz = b[3];
    const result = [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw
    ];
    return normalizeQuatValues(result);
};

const rotateVectorByQuatValues = (quat: number[], value: number[]) => {
    const qw = quat[0], qx = quat[1], qy = quat[2], qz = quat[3];
    const vx = value[0], vy = value[1], vz = value[2];
    const tx = 2 * (qy * vz - qz * vy);
    const ty = 2 * (qz * vx - qx * vz);
    const tz = 2 * (qx * vy - qy * vx);
    return [
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx)
    ];
};

const vectorToSplatLocal = (splat: Splat, worldVector: number[]) => {
    const rotation = splat.entity.getLocalRotation();
    return rotateVectorByQuatValues(
        [rotation.w, -rotation.x, -rotation.y, -rotation.z],
        worldVector
    );
};

const proxyMaterialFromObject = (object?: PhysicsObject | null): ProxyMaterial => {
    const preset = object ? materialDefaults[object.material] ?? materialDefaults.jelly : materialDefaults.jelly;
    const E = Math.max(Number(object?.E ?? preset.E) || preset.E, 1);
    const nu = Math.max(0, Math.min(Number(object?.nu ?? preset.nu) || preset.nu, 0.49));
    const density = Math.max(Number(object?.density ?? preset.density) || preset.density, 1);
    const rigidWeight = object?.mode === 'rigid-body' || object?.mode === 'rigid-soft' ? 1 : 0;
    const logE = Math.log10(Math.max(E, 1));
    return {
        E,
        nu,
        density,
        rigidWeight,
        stressScale: Math.max(0.04, Math.min(0.42, 0.08 + logE * 0.035 + rigidWeight * 0.08))
    };
};

const averageProxyMaterial = (materials: ProxyMaterial[]) => {
    if (!materials.length) {
        return proxyMaterialFromObject(null);
    }
    const result = materials.reduce((accum, material) => {
        accum.E += material.E;
        accum.nu += material.nu;
        accum.density += material.density;
        accum.rigidWeight += material.rigidWeight;
        accum.stressScale += material.stressScale;
        return accum;
    }, { E: 0, nu: 0, density: 0, rigidWeight: 0, stressScale: 0 });
    const inv = 1 / materials.length;
    return {
        E: result.E * inv,
        nu: Math.max(0, Math.min(result.nu * inv, 0.49)),
        density: Math.max(result.density * inv, 1),
        rigidWeight: Math.max(0, Math.min(result.rigidWeight * inv, 1)),
        stressScale: Math.max(0.04, Math.min(result.stressScale * inv, 0.5))
    };
};

const objectAabbCenter = (object?: PhysicsObject | null) => {
    const aabb = object?.aabbLocal ?? object?.aabbWorld;
    const min = aabb?.min;
    const max = aabb?.max;
    if (!Array.isArray(min) || !Array.isArray(max) || min.length < 3 || max.length < 3) {
        return null;
    }
    return [
        (Number(min[0]) + Number(max[0])) * 0.5,
        (Number(min[1]) + Number(max[1])) * 0.5,
        (Number(min[2]) + Number(max[2])) * 0.5
    ];
};

const proxyMaterialForFixedObject = (object: PhysicsObject, bodyParts: PhysicsObject[]) => {
    const movableParts = bodyParts.filter(part => part.mode !== 'obstacle');
    if (!movableParts.length) {
        return proxyMaterialFromObject(null);
    }
    const center = objectAabbCenter(object);
    if (!center) {
        return proxyMaterialFromObject(movableParts[0]);
    }
    let nearest = movableParts[0];
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const part of movableParts) {
        const partCenter = objectAabbCenter(part);
        if (!partCenter) {
            continue;
        }
        const distance =
            (center[0] - partCenter[0]) ** 2 +
            (center[1] - partCenter[1]) ** 2 +
            (center[2] - partCenter[2]) ** 2;
        if (distance < bestDistance) {
            nearest = part;
            bestDistance = distance;
        }
    }
    return proxyMaterialFromObject(nearest);
};

const proxyMaterialForPart = (object: PhysicsObject, bodyParts: PhysicsObject[]) => (
    object.mode === 'obstacle' ? proxyMaterialForFixedObject(object, bodyParts) : proxyMaterialFromObject(object)
);

const expandedProxyBoundsContains = (bounds: ProxyBounds, point: number[], expansionRatio = 0.04) => {
    const span = Math.max(
        bounds.maxs[0] - bounds.mins[0],
        bounds.maxs[1] - bounds.mins[1],
        bounds.maxs[2] - bounds.mins[2],
        1e-6
    );
    const pad = span * Math.max(0, expansionRatio);
    return point[0] >= bounds.mins[0] - pad && point[0] <= bounds.maxs[0] + pad &&
        point[1] >= bounds.mins[1] - pad && point[1] <= bounds.maxs[1] + pad &&
        point[2] >= bounds.mins[2] - pad && point[2] <= bounds.maxs[2] + pad;
};

const proxyStructureStiffness = (materials: ProxyMaterial[]) => {
    const material = averageProxyMaterial(materials);
    const logE = Math.log10(Math.max(material.E, 1));
    return Math.max(
        0.04,
        Math.min(
            0.9,
            0.06 + logE * 0.035 + material.rigidWeight * 0.45
        )
    );
};

const identityMatrix3 = () => [
    1, 0, 0,
    0, 1, 0,
    0, 0, 1
];

const dampProxyAffine = (matrix: number[], scale: number, limit: number) => (
    matrix.slice(0, 9).map(value => Math.max(-limit, Math.min(limit, value * scale)))
);

const dampProxyDeformation = (matrix: number[], keep: number, limit: number) => {
    const result = matrix.slice(0, 9);
    const alpha = Math.max(0, Math.min(keep, 1));
    for (let i = 0; i < 9; ++i) {
        const identity = i === 0 || i === 4 || i === 8 ? 1 : 0;
        result[i] = identity + (result[i] - identity) * alpha;
        result[i] = Math.max(identity - limit, Math.min(identity + limit, result[i]));
    }
    return result;
};

const mulMatrix3 = (a: number[], b: number[]) => [
    a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
    a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
    a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
    a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
    a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
    a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
    a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
    a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
    a[6] * b[2] + a[7] * b[5] + a[8] * b[8]
];

const clampProxyDeformation = (matrix: number[], material: ProxyMaterial) => {
    const elasticBlend = Math.max(
        0.05,
        Math.min(1, 1 - material.rigidWeight * 0.75)
    );
    const result = matrix.slice(0, 9);
    for (let i = 0; i < 9; ++i) {
        const identity = i === 0 || i === 4 || i === 8 ? 1 : 0;
        result[i] = identity + (result[i] - identity) * elasticBlend;
        const limit = i === 0 || i === 4 || i === 8 ? 0.35 : 0.22;
        result[i] = Math.max(identity - limit, Math.min(identity + limit, result[i]));
    }
    return result;
};

const localProxyFrameFromState = (state: LocalProxyState): LocalProxyFrame => {
    const count = state.indices.length;
    const positions = new Float32Array(count * 3);
    const rotations = new Float32Array(count * 4);
    const scales = new Float32Array(count * 3);

    for (let order = 0; order < count; ++order) {
        const index = state.indices[order];
        const src3 = order * 3;
        const src4 = order * 4;
        const restPoint = [
            state.restPositions[src3],
            state.restPositions[src3 + 1],
            state.restPositions[src3 + 2]
        ];
        const blends = state.skinning.get(index) ?? [[state.groupByIndex.get(index) ?? 0, 1]];
        const blendedPosition = [0, 0, 0];
        const quatAccum = [0, 0, 0, 0];
        let totalWeight = 0;
        let referenceQuat: number[] | null = null;

        for (const [groupId, rawWeight] of blends) {
            if (rawWeight <= 0 || groupId < 0 || groupId >= state.groups.length) {
                continue;
            }
            const weight = rawWeight;
            const center = state.centers[groupId];
            const restCenter = state.restCenters[groupId];
            const deltaRotation = state.rotations[groupId];
            const local = [
                restPoint[0] - restCenter[0],
                restPoint[1] - restCenter[1],
                restPoint[2] - restCenter[2]
            ];
            const rotated = rotateVectorByQuatValues(deltaRotation, local);
            blendedPosition[0] += weight * (center[0] + rotated[0]);
            blendedPosition[1] += weight * (center[1] + rotated[1]);
            blendedPosition[2] += weight * (center[2] + rotated[2]);

            if (!referenceQuat) {
                referenceQuat = deltaRotation;
            }
            const sign = referenceQuat[0] * deltaRotation[0] +
                referenceQuat[1] * deltaRotation[1] +
                referenceQuat[2] * deltaRotation[2] +
                referenceQuat[3] * deltaRotation[3] < 0 ? -1 : 1;
            quatAccum[0] += sign * weight * deltaRotation[0];
            quatAccum[1] += sign * weight * deltaRotation[1];
            quatAccum[2] += sign * weight * deltaRotation[2];
            quatAccum[3] += sign * weight * deltaRotation[3];
            totalWeight += weight;
        }

        if (totalWeight <= 1e-8) {
            blendedPosition[0] = restPoint[0];
            blendedPosition[1] = restPoint[1];
            blendedPosition[2] = restPoint[2];
            quatAccum[0] = 1;
        } else if (Math.abs(totalWeight - 1) > 1e-5) {
            blendedPosition[0] /= totalWeight;
            blendedPosition[1] /= totalWeight;
            blendedPosition[2] /= totalWeight;
        }
        normalizeQuatValues(quatAccum);

        positions[src3] = blendedPosition[0];
        positions[src3 + 1] = blendedPosition[1];
        positions[src3 + 2] = blendedPosition[2];

        const baseRotation = [
            state.restRotations[src4],
            state.restRotations[src4 + 1],
            state.restRotations[src4 + 2],
            state.restRotations[src4 + 3]
        ];
        const finalRotation = quatMulArray(quatAccum, baseRotation);
        rotations[src4] = finalRotation[0];
        rotations[src4 + 1] = finalRotation[1];
        rotations[src4 + 2] = finalRotation[2];
        rotations[src4 + 3] = finalRotation[3];

        scales[src3] = state.restScales[src3];
        scales[src3 + 1] = state.restScales[src3 + 1];
        scales[src3 + 2] = state.restScales[src3 + 2];
    }

    return {
        positions,
        rotations,
        scales,
        indices: state.indexArray,
        updateBounds: false
    };
};

const shapeCoupleProxyVelocities = (
    centers: number[][],
    velocities: number[][],
    restCenters: number[][],
    masses: number[],
    dt: number,
    stiffness: number
) => {
    if (stiffness <= 0 || centers.length === 0) {
        return velocities;
    }
    const totalMass = Math.max(masses.reduce((sum, mass) => sum + Math.max(mass, 1e-6), 0), 1e-6);
    const currentCom = [0, 0, 0];
    const restCom = [0, 0, 0];
    for (let i = 0; i < centers.length; ++i) {
        const mass = Math.max(masses[i], 1e-6);
        for (let axis = 0; axis < 3; ++axis) {
            currentCom[axis] += centers[i][axis] * mass / totalMass;
            restCom[axis] += restCenters[i][axis] * mass / totalMass;
        }
    }
    const invDt = 1 / Math.max(dt, 1e-6);
    const strength = Math.max(0, Math.min(stiffness, 0.9));
    for (let i = 0; i < centers.length; ++i) {
        for (let axis = 0; axis < 3; ++axis) {
            const target = currentCom[axis] + restCenters[i][axis] - restCom[axis];
            velocities[i][axis] += (target - centers[i][axis]) * strength * invDt;
        }
    }
    return velocities;
};

const pullProxyCentersTowardGrabbedShape = (
    centers: number[][],
    velocities: number[][],
    restCenters: number[][],
    grabbedGroup: number | null,
    grabbedTarget: number[] | null,
    grabbedVelocity: number[],
    dt: number,
    strength: number,
    maxSpeed: number
) => {
    if (grabbedGroup === null || !grabbedTarget || grabbedGroup < 0 || grabbedGroup >= centers.length) {
        return { centers, velocities };
    }
    const translation = [
        grabbedTarget[0] - restCenters[grabbedGroup][0],
        grabbedTarget[1] - restCenters[grabbedGroup][1],
        grabbedTarget[2] - restCenters[grabbedGroup][2]
    ];
    const alpha = Math.max(0, Math.min(strength, 1));
    const maxCorrection = Math.max(maxSpeed * dt * 2.8, 1e-5);
    for (let groupId = 0; groupId < centers.length; ++groupId) {
        if (groupId === grabbedGroup) {
            centers[groupId] = grabbedTarget.slice(0, 3);
            velocities[groupId] = grabbedVelocity.slice(0, 3);
            continue;
        }
        const desired = [
            restCenters[groupId][0] + translation[0],
            restCenters[groupId][1] + translation[1],
            restCenters[groupId][2] + translation[2]
        ];
        const correction = clampVectorLength([
            desired[0] - centers[groupId][0],
            desired[1] - centers[groupId][1],
            desired[2] - centers[groupId][2]
        ], maxCorrection);
        const applied = [
            correction[0] * alpha,
            correction[1] * alpha,
            correction[2] * alpha
        ];
        centers[groupId][0] += applied[0];
        centers[groupId][1] += applied[1];
        centers[groupId][2] += applied[2];
        velocities[groupId] = clampVectorLength([
            velocities[groupId][0] + applied[0] / Math.max(dt, 1e-6),
            velocities[groupId][1] + applied[1] / Math.max(dt, 1e-6),
            velocities[groupId][2] + applied[2] / Math.max(dt, 1e-6)
        ], Math.max(maxSpeed * 1.35, 1e-4));
    }
    return { centers, velocities };
};

const stepLocalProxyMpm = (
    centers: number[][],
    velocities: number[][],
    affines: number[][],
    deformations: number[][],
    masses: number[],
    materials: ProxyMaterial[],
    bounds: ProxyBounds,
    gridRes: number,
    dt: number,
    gravity: number[],
    groundZ: number | null,
    driveGroup: number | null,
    driveVelocity: number[] | null,
    kinematicGroups: Set<number> = new Set()
) => {
    const count = centers.length;
    const pad = Math.max(
        Math.max(
            bounds.maxs[0] - bounds.mins[0],
            bounds.maxs[1] - bounds.mins[1],
            bounds.maxs[2] - bounds.mins[2]
        ) * 0.12,
        1e-5
    );
    const lo = [bounds.mins[0] - pad, bounds.mins[1] - pad, bounds.mins[2] - pad];
    const hi = [bounds.maxs[0] + pad, bounds.maxs[1] + pad, bounds.maxs[2] + pad];
    const dx = [
        Math.max((hi[0] - lo[0]) / Math.max(gridRes - 1, 1), 1e-6),
        Math.max((hi[1] - lo[1]) / Math.max(gridRes - 1, 1), 1e-6),
        Math.max((hi[2] - lo[2]) / Math.max(gridRes - 1, 1), 1e-6)
    ];
    const invDx = [1 / dx[0], 1 / dx[1], 1 / dx[2]];
    const avgDx = (dx[0] + dx[1] + dx[2]) / 3;
    const nodeMass = new Map<string, number>();
    const nodeMomentum = new Map<string, number[]>();

    const weightsFor = (position: number[]) => {
        const gridPosition = [
            (position[0] - lo[0]) * invDx[0],
            (position[1] - lo[1]) * invDx[1],
            (position[2] - lo[2]) * invDx[2]
        ];
        const base = [
            Math.floor(gridPosition[0] - 0.5),
            Math.floor(gridPosition[1] - 0.5),
            Math.floor(gridPosition[2] - 0.5)
        ];
        const fx = [gridPosition[0] - base[0], gridPosition[1] - base[1], gridPosition[2] - base[2]];
        const weights = [
            [0.5 * (1.5 - fx[0]) ** 2, 0.75 - (fx[0] - 1) ** 2, 0.5 * (fx[0] - 0.5) ** 2],
            [0.5 * (1.5 - fx[1]) ** 2, 0.75 - (fx[1] - 1) ** 2, 0.5 * (fx[1] - 0.5) ** 2],
            [0.5 * (1.5 - fx[2]) ** 2, 0.75 - (fx[2] - 1) ** 2, 0.5 * (fx[2] - 0.5) ** 2]
        ];
        const gradients = [
            [fx[0] - 1.5, -2 * (fx[0] - 1), fx[0] - 0.5],
            [fx[1] - 1.5, -2 * (fx[1] - 1), fx[1] - 0.5],
            [fx[2] - 1.5, -2 * (fx[2] - 1), fx[2] - 0.5]
        ];
        return { base, weights, gradients };
    };

    for (const groupId of kinematicGroups) {
        if (groupId >= 0 && groupId < velocities.length) {
            velocities[groupId] = [0, 0, 0];
            affines[groupId] = new Array(9).fill(0);
            deformations[groupId] = identityMatrix3();
        }
    }
    if (driveGroup !== null && driveVelocity && driveGroup >= 0 && driveGroup < velocities.length) {
        velocities[driveGroup] = driveVelocity.slice(0, 3);
        affines[driveGroup] = new Array(9).fill(0);
    }

    for (let particle = 0; particle < count; ++particle) {
        const { base, weights, gradients } = weightsFor(centers[particle]);
        const material = materials[particle] ?? proxyMaterialFromObject(null);
        const isKinematic = kinematicGroups.has(particle);
        const deformation = isKinematic ? identityMatrix3() : (deformations[particle] ?? identityMatrix3());
        const strain = [
            deformation[0] - 1,
            0.5 * (deformation[1] + deformation[3]),
            0.5 * (deformation[2] + deformation[6]),
            0.5 * (deformation[3] + deformation[1]),
            deformation[4] - 1,
            0.5 * (deformation[5] + deformation[7]),
            0.5 * (deformation[6] + deformation[2]),
            0.5 * (deformation[7] + deformation[5]),
            deformation[8] - 1
        ];
        const trace = strain[0] + strain[4] + strain[8];
        const young = Math.max(0.002, Math.min(material.E / 1e7, 1.5));
        const nu = Math.max(0, Math.min(material.nu, 0.45));
        const mu = young / (2 * (1 + nu));
        const lambda = young * nu / Math.max((1 + nu) * (1 - 2 * nu), 1e-6);
        const stressScale = isKinematic ? 0 : material.stressScale;
        const stress = [
            (2 * mu * strain[0] + lambda * trace) * stressScale,
            2 * mu * strain[1] * stressScale,
            2 * mu * strain[2] * stressScale,
            2 * mu * strain[3] * stressScale,
            (2 * mu * strain[4] + lambda * trace) * stressScale,
            2 * mu * strain[5] * stressScale,
            2 * mu * strain[6] * stressScale,
            2 * mu * strain[7] * stressScale,
            (2 * mu * strain[8] + lambda * trace) * stressScale
        ];
        const volume = Math.max(masses[particle] / Math.max(material.density / 1000, 1e-6), 1e-6);
        for (let ox = 0; ox < 3; ++ox) {
            const ix = Math.max(0, Math.min(gridRes - 1, base[0] + ox));
            for (let oy = 0; oy < 3; ++oy) {
                const iy = Math.max(0, Math.min(gridRes - 1, base[1] + oy));
                for (let oz = 0; oz < 3; ++oz) {
                    const iz = Math.max(0, Math.min(gridRes - 1, base[2] + oz));
                    const weight = weights[0][ox] * weights[1][oy] * weights[2][oz];
                    if (weight <= 0) {
                        continue;
                    }
                    const key = `${ix},${iy},${iz}`;
                    const nodeWorld = [lo[0] + ix * dx[0], lo[1] + iy * dx[1], lo[2] + iz * dx[2]];
                    const dpos = [
                        nodeWorld[0] - centers[particle][0],
                        nodeWorld[1] - centers[particle][1],
                        nodeWorld[2] - centers[particle][2]
                    ];
                    const affine = affines[particle];
                    const apicVelocity = [
                        velocities[particle][0] + affine[0] * dpos[0] + affine[1] * dpos[1] + affine[2] * dpos[2],
                        velocities[particle][1] + affine[3] * dpos[0] + affine[4] * dpos[1] + affine[5] * dpos[2],
                        velocities[particle][2] + affine[6] * dpos[0] + affine[7] * dpos[1] + affine[8] * dpos[2]
                    ];
                    const mass = Math.max(masses[particle], 1e-6);
                    nodeMass.set(key, (nodeMass.get(key) ?? 0) + weight * mass);
                    const momentum = nodeMomentum.get(key) ?? [0, 0, 0];
                    momentum[0] += weight * mass * apicVelocity[0];
                    momentum[1] += weight * mass * apicVelocity[1];
                    momentum[2] += weight * mass * apicVelocity[2];
                    const gradW = [
                        gradients[0][ox] * invDx[0] * weights[1][oy] * weights[2][oz],
                        weights[0][ox] * gradients[1][oy] * invDx[1] * weights[2][oz],
                        weights[0][ox] * weights[1][oy] * gradients[2][oz] * invDx[2]
                    ];
                    momentum[0] -= dt * volume * (stress[0] * gradW[0] + stress[1] * gradW[1] + stress[2] * gradW[2]);
                    momentum[1] -= dt * volume * (stress[3] * gradW[0] + stress[4] * gradW[1] + stress[5] * gradW[2]);
                    momentum[2] -= dt * volume * (stress[6] * gradW[0] + stress[7] * gradW[1] + stress[8] * gradW[2]);
                    nodeMomentum.set(key, momentum);
                }
            }
        }
    }

    const nodeVelocity = new Map<string, number[]>();
    for (const [key, momentum] of nodeMomentum) {
        const mass = Math.max(nodeMass.get(key) ?? 0, 1e-12);
        const velocity = [
            momentum[0] / mass + gravity[0] * dt,
            momentum[1] / mass + gravity[1] * dt,
            momentum[2] / mass + gravity[2] * dt
        ];
        if (groundZ !== null) {
            const iz = Number(key.split(',')[2]);
            const nodeZ = lo[2] + iz * dx[2];
            if (nodeZ <= groundZ + dx[2] * 1.25 && velocity[2] < 0) {
                velocity[2] = 0;
            }
        }
        nodeVelocity.set(key, velocity);
    }

    const nextCenters = centers.map(center => center.slice(0, 3));
    const nextVelocities = velocities.map(velocity => velocity.slice(0, 3));
    const nextAffines = affines.map(() => new Array(9).fill(0));
    const nextDeformations = deformations.map(deformation => deformation.slice(0, 9));
    const deltaRotations: number[][] = [];
    const affineScale = 4 / Math.max(avgDx * avgDx, 1e-12);

    for (let particle = 0; particle < count; ++particle) {
        if (kinematicGroups.has(particle)) {
            const prescribedVelocity = driveGroup === particle && driveVelocity ? driveVelocity.slice(0, 3) : [0, 0, 0];
            nextVelocities[particle] = prescribedVelocity;
            nextAffines[particle] = new Array(9).fill(0);
            nextDeformations[particle] = identityMatrix3();
            nextCenters[particle] = [
                centers[particle][0] + prescribedVelocity[0] * dt,
                centers[particle][1] + prescribedVelocity[1] * dt,
                centers[particle][2] + prescribedVelocity[2] * dt
            ];
            deltaRotations.push([1, 0, 0, 0]);
            continue;
        }
        const { base, weights } = weightsFor(centers[particle]);
        const particleVelocity = [0, 0, 0];
        const particleAffine = new Array(9).fill(0);
        for (let ox = 0; ox < 3; ++ox) {
            const ix = Math.max(0, Math.min(gridRes - 1, base[0] + ox));
            for (let oy = 0; oy < 3; ++oy) {
                const iy = Math.max(0, Math.min(gridRes - 1, base[1] + oy));
                for (let oz = 0; oz < 3; ++oz) {
                    const iz = Math.max(0, Math.min(gridRes - 1, base[2] + oz));
                    const weight = weights[0][ox] * weights[1][oy] * weights[2][oz];
                    const gridVelocity = nodeVelocity.get(`${ix},${iy},${iz}`);
                    if (!gridVelocity || weight <= 0) {
                        continue;
                    }
                    const nodeWorld = [lo[0] + ix * dx[0], lo[1] + iy * dx[1], lo[2] + iz * dx[2]];
                    const dpos = [
                        nodeWorld[0] - centers[particle][0],
                        nodeWorld[1] - centers[particle][1],
                        nodeWorld[2] - centers[particle][2]
                    ];
                    for (let axis = 0; axis < 3; ++axis) {
                        particleVelocity[axis] += weight * gridVelocity[axis];
                    }
                    for (let row = 0; row < 3; ++row) {
                        for (let col = 0; col < 3; ++col) {
                            particleAffine[row * 3 + col] += affineScale * weight * gridVelocity[row] * dpos[col];
                        }
                    }
                }
            }
        }
        if (driveGroup !== null && driveVelocity && particle === driveGroup) {
            particleVelocity[0] = driveVelocity[0];
            particleVelocity[1] = driveVelocity[1];
            particleVelocity[2] = driveVelocity[2];
            particleAffine.fill(0);
        }
        nextVelocities[particle] = particleVelocity;
        nextAffines[particle] = particleAffine;
        const incremental = [
            1 + dt * particleAffine[0], dt * particleAffine[1], dt * particleAffine[2],
            dt * particleAffine[3], 1 + dt * particleAffine[4], dt * particleAffine[5],
            dt * particleAffine[6], dt * particleAffine[7], 1 + dt * particleAffine[8]
        ];
        nextDeformations[particle] = clampProxyDeformation(
            mulMatrix3(incremental, deformations[particle] ?? identityMatrix3()),
            materials[particle] ?? proxyMaterialFromObject(null)
        );
        for (let axis = 0; axis < 3; ++axis) {
            nextCenters[particle][axis] = centers[particle][axis] + particleVelocity[axis] * dt;
        }
        if (groundZ !== null && nextCenters[particle][2] < groundZ) {
            nextCenters[particle][2] = groundZ;
            nextVelocities[particle][2] = Math.max(nextVelocities[particle][2], 0);
        }
        deltaRotations.push(quatFromAngularVelocity([
            0.5 * (particleAffine[7] - particleAffine[5]),
            0.5 * (particleAffine[2] - particleAffine[6]),
            0.5 * (particleAffine[3] - particleAffine[1])
        ], dt));
    }

    return { centers: nextCenters, velocities: nextVelocities, affines: nextAffines, deformations: nextDeformations, deltaRotations };
};

const normalizedAngle = (degrees: number) => {
    if (!Number.isFinite(degrees)) {
        return 0;
    }
    let result = ((degrees + 180) % 360 + 360) % 360 - 180;
    if (Math.abs(result) < 1e-4) {
        result = 0;
    }
    return result;
};

const physicsRotationFor = (splat: Splat | null) => {
    if (!splat) {
        return {
            degree: [0],
            axis: [0]
        };
    }

    splat.entity.getLocalRotation().getEulerAngles(eulerScratch);
    const eulers = [
        normalizedAngle(eulerScratch.x),
        normalizedAngle(eulerScratch.y),
        normalizedAngle(eulerScratch.z)
    ];
    const degree: number[] = [];
    const axis: number[] = [];
    eulers.forEach((value, index) => {
        if (value !== 0) {
            degree.push(value);
            axis.push(index);
        }
    });

    return {
        degree: degree.length ? degree : [0],
        axis: axis.length ? axis : [0]
    };
};

const registerPhysicsSessionEvents = (events: Events) => {
    const objects = new Map<number, PhysicsObject>();
    let nextObjectId = 1;
    let nextBodyId = 1;
    let selectedCount = 0;
    let simArea: number[] | null = null;
    let simAreaLocal: { min: number[], max: number[] } | null = null;
    let simAreaIndices: number[] | null = null;
    let scale = 1;
    let nGrid = 50;
    let opacityThreshold = 0.02;
    let modelId = '';
    let modelInfo: any = null;
    let loadedBasePlyUrl = '';
    let apiBase = '';
    let officialConfigEnabled = false;
    let solver: PhysicsPayload['solver'] = 'explicit-mpm';
    let currentRunId = '';
    let lastRunId = '';
    let simulation: PhysicsPayload['simulation'] = {
        gravityEnabled: true,
        gravity: [0, 0, -9.8],
        groundEnabled: true,
        groundHeight: 0.5,
        boundingBoxEnabled: true,
        frame_dt: 0.02,
        frame_num: 30,
        substep_dt: 1e-4,
        damping: 0.9999,
        newtonMaxIter: 16,
        pbmpm: {
            n_min: 3,
            n_max: 25,
            plastic_mode: 0,
            yield_min: 0.55,
            yield_max: 1.85
        }
    };
    const syncedFiles = new WeakMap<File, string>();
    let voxelGroupTimer: number | null = null;
    let scheduleVoxelGroups = (_payload?: PhysicsPayload) => {};
    let liveProxyState: LocalProxyState | null = null;
    let localProxyCache = new Map<number, LocalProxyTemplate>();
    let localProxyCacheTimer: number | null = null;
    let scheduleLocalProxyCache = (_delay?: number) => {};

    const selectedSplat = () => {
        const fallback = () => {
            if (!events.functions.has('scene.splats')) {
                return null;
            }
            return ((events.invoke('scene.splats') as Splat[]) ?? []).find(splat => splat.visible) ?? null;
        };
        if (!events.functions.has('selection')) {
            return fallback();
        }
        const splat = events.invoke('selection') as Splat;
        return splat?.visible ? splat : fallback();
    };

    const refreshSelectedCount = () => {
        const splat = selectedSplat();
        selectedCount = splat ? selectedIndicesFor(splat).length : 0;
        events.fire('physics.selectionCount', selectedCount);
    };

    const toPhysicsAabbFromIndices = (indices?: number[]) => {
        const splat = selectedSplat();
        if (!splat || !indices?.length) {
            return null;
        }

        const x = splat.splatData.getProp('x') as Float32Array;
        const y = splat.splatData.getProp('y') as Float32Array;
        const z = splat.splatData.getProp('z') as Float32Array;
        physicsRotationScratch.copy(splat.entity.getLocalRotation());
        const min = [Infinity, Infinity, Infinity];
        const max = [-Infinity, -Infinity, -Infinity];
        let found = false;
        for (const index of indices) {
            if (index < 0 || index >= x.length) {
                continue;
            }
            pointScratch.set(x[index], y[index], z[index]);
            physicsRotationScratch.transformVector(pointScratch, rotatedScratch);
            min[0] = Math.min(min[0], rotatedScratch.x);
            min[1] = Math.min(min[1], rotatedScratch.y);
            min[2] = Math.min(min[2], rotatedScratch.z);
            max[0] = Math.max(max[0], rotatedScratch.x);
            max[1] = Math.max(max[1], rotatedScratch.y);
            max[2] = Math.max(max[2], rotatedScratch.z);
            found = true;
        }

        return found ? { min, max } : null;
    };

    const toPhysicsAabb = (aabbLocal?: { min: number[], max: number[] }) => {
        const splat = selectedSplat();
        if (!splat || !aabbLocal) {
            return null;
        }

        const lo = aabbLocal.min;
        const hi = aabbLocal.max;
        if (!Array.isArray(lo) || !Array.isArray(hi) || lo.length !== 3 || hi.length !== 3) {
            return null;
        }

        physicsRotationScratch.copy(splat.entity.getLocalRotation());
        const min = [Infinity, Infinity, Infinity];
        const max = [-Infinity, -Infinity, -Infinity];
        for (const x of [lo[0], hi[0]]) {
            for (const y of [lo[1], hi[1]]) {
                for (const z of [lo[2], hi[2]]) {
                    pointScratch.set(x, y, z);
                    physicsRotationScratch.transformVector(pointScratch, rotatedScratch);
                    min[0] = Math.min(min[0], rotatedScratch.x);
                    min[1] = Math.min(min[1], rotatedScratch.y);
                    min[2] = Math.min(min[2], rotatedScratch.z);
                    max[0] = Math.max(max[0], rotatedScratch.x);
                    max[1] = Math.max(max[1], rotatedScratch.y);
                    max[2] = Math.max(max[2], rotatedScratch.z);
                }
            }
        }

        return { min, max };
    };

    const refreshPhysicsAabbs = () => {
        for (const object of objects.values()) {
            const aabbWorld = toPhysicsAabbFromIndices(object.indices) ?? toPhysicsAabb(object.aabbLocal) ?? object.aabbWorld;
            object.aabbWorld = aabbWorld;
        }
    };

    const boundsToSimArea = (aabb: Bounds3 | null) => (
        aabb ? [
            aabb.min[0],
            aabb.max[0],
            aabb.min[1],
            aabb.max[1],
            aabb.min[2],
            aabb.max[2]
        ] : null
    );

    const unionPhysicsAabb = (items: PhysicsObject[]) => {
        const min = [Infinity, Infinity, Infinity];
        const max = [-Infinity, -Infinity, -Infinity];
        let found = false;
        for (const item of items) {
            const aabb = item.aabbWorld;
            if (!aabb?.min || !aabb?.max) {
                continue;
            }
            found = true;
            for (let axis = 0; axis < 3; ++axis) {
                min[axis] = Math.min(min[axis], Number(aabb.min[axis]));
                max[axis] = Math.max(max[axis], Number(aabb.max[axis]));
            }
        }
        return found ? { min, max } : null;
    };

    const gridNodeStats = (bounds: Bounds3, gridDx: number) => {
        const counts = [0, 0, 0];
        const indexMin = [0, 0, 0];
        const indexMax = [-1, -1, -1];
        for (let axis = 0; axis < 3; ++axis) {
            let first = -1;
            let last = -1;
            for (let index = 0; index < Math.max(0, Math.floor(nGrid)); ++index) {
                const coord = index * gridDx;
                if (bounds.min[axis] < coord && coord < bounds.max[axis]) {
                    if (first < 0) {
                        first = index;
                    }
                    last = index;
                    counts[axis]++;
                }
            }
            indexMin[axis] = first < 0 ? 0 : first;
            indexMax[axis] = last;
        }
        return {
            counts,
            count: counts[0] * counts[1] * counts[2],
            indexBounds: {
                min: indexMin,
                max: indexMax
            }
        };
    };

    const buildVelocityCuboidPreview = (
        object: PhysicsObject,
        simAreaForConstraints: number[]
    ): BoundaryConstraintPreview | null => {
        if (object.material !== 'obstacle' && object.mode !== 'obstacle') {
            return null;
        }
        const aabb = object.aabbWorld;
        if (!aabb?.min || !aabb?.max || simAreaForConstraints.length !== 6) {
            return null;
        }

        const simMin = [simAreaForConstraints[0], simAreaForConstraints[2], simAreaForConstraints[4]];
        const simMax = [simAreaForConstraints[1], simAreaForConstraints[3], simAreaForConstraints[5]];
        const simCenter = [
            (simMin[0] + simMax[0]) * 0.5,
            (simMin[1] + simMax[1]) * 0.5,
            (simMin[2] + simMax[2]) * 0.5
        ];
        const maxDiff = Math.max(simMax[0] - simMin[0], simMax[1] - simMin[1], simMax[2] - simMin[2]);
        if (!Number.isFinite(maxDiff) || maxDiff <= 1e-12) {
            return null;
        }
        const factor = scale / maxDiff;
        const toMpm = (value: number, axis: number) => (value - simCenter[axis]) * factor + 1.0;
        const mpmMin = [
            toMpm(Number(aabb.min[0]), 0),
            toMpm(Number(aabb.min[1]), 1),
            toMpm(Number(aabb.min[2]), 2)
        ];
        const mpmMax = [
            toMpm(Number(aabb.max[0]), 0),
            toMpm(Number(aabb.max[1]), 1),
            toMpm(Number(aabb.max[2]), 2)
        ];
        const gridDx = mpmGridLim / Math.max(Math.floor(nGrid), 1);
        const paddingGridNodes = Math.max(0, Number(object.cuboidPaddingGridNodes ?? 2) || 0);
        const padding = gridDx * paddingGridNodes;
        const paddedMin = [0, 1, 2].map(axis => Math.min(mpmMin[axis], mpmMax[axis]) - padding);
        const paddedMax = [0, 1, 2].map(axis => Math.max(mpmMin[axis], mpmMax[axis]) + padding);
        for (let axis = 0; axis < 3; ++axis) {
            const center = (paddedMin[axis] + paddedMax[axis]) * 0.5;
            const minHalf = Math.max(gridDx * paddingGridNodes, gridDx * 1.01);
            if ((paddedMax[axis] - paddedMin[axis]) * 0.5 < minHalf) {
                paddedMin[axis] = center - minHalf;
                paddedMax[axis] = center + minHalf;
            }
        }

        const paddedBounds = { min: paddedMin, max: paddedMax };
        const stats = gridNodeStats(paddedBounds, gridDx);
        const hasLinearDrive = object.drive.linearEnabled &&
            object.drive.linearForce.some(value => Math.abs(Number(value) || 0) > 1e-12);
        const velocity = hasLinearDrive ? object.drive.linearForce.slice(0, 3) : [0, 0, 0];
        const start = hasLinearDrive ? object.drive.linearStart : 0;
        const end = hasLinearDrive ? start + Math.max(1, object.drive.linearNumDt) * simulation.substep_dt : 1e3;

        return {
            type: 'velocity_cuboid',
            constraint_type: 'set_velocity_on_cuboid',
            source: 'obstacle_part',
            objectId: object.objectId,
            bodyId: object.bodyId ?? object.objectId,
            coordinateSpace: 'mpm_shifted_normalized',
            sceneBounds: {
                min: aabb.min.slice(0, 3),
                max: aabb.max.slice(0, 3)
            },
            simArea: simAreaForConstraints.slice(0, 6),
            mpmBounds: {
                min: [0, 1, 2].map(axis => Math.min(mpmMin[axis], mpmMax[axis])),
                max: [0, 1, 2].map(axis => Math.max(mpmMin[axis], mpmMax[axis]))
            },
            paddedMpmBounds: paddedBounds,
            point: [0, 1, 2].map(axis => (paddedMin[axis] + paddedMax[axis]) * 0.5),
            size: [0, 1, 2].map(axis => (paddedMax[axis] - paddedMin[axis]) * 0.5),
            scale: [0, 1, 2].map(axis => paddedMax[axis] - paddedMin[axis]),
            velocity,
            start_time: start,
            end_time: end,
            reset: 1,
            gridDx,
            padding,
            paddingGridNodes,
            estimatedGridNodeCounts: stats.counts,
            estimatedGridNodeCount: stats.count,
            gridIndexBounds: stats.indexBounds
        };
    };

    const setSimAreaFromAabb = (aabb: { min: number[], max: number[] }, indices?: number[]) => {
        const min = aabb.min;
        const max = aabb.max;
        simArea = [min[0], max[0], min[1], max[1], min[2], max[2]];
        simAreaLocal = null;
        simAreaIndices = indices?.slice() ?? null;
        if (indices) {
            selectedCount = indices.length;
            events.fire('physics.selectionCount', selectedCount);
        }
        emit();
    };

    const buildPayload = (): PhysicsPayload => {
        const splat = selectedSplat();
        const rotation = physicsRotationFor(splat);
        refreshPhysicsAabbs();
        const activeSimArea = toPhysicsAabbFromIndices(simAreaIndices ?? undefined) ?? (simAreaLocal ? toPhysicsAabb(simAreaLocal) : null);
        if (activeSimArea) {
            simArea = [
                activeSimArea.min[0],
                activeSimArea.max[0],
                activeSimArea.min[1],
                activeSimArea.max[1],
                activeSimArea.min[2],
                activeSimArea.max[2]
            ];
        }
        const objectList = Array.from(objects.values()).map(object => ({
            ...object,
            drive: JSON.parse(JSON.stringify(object.drive)),
            indices: object.indices.slice(),
            boundaryDebug: null as BoundaryConstraintPreview | null
        }));
        const constraintSimArea = boundsToSimArea(activeSimArea) ?? boundsToSimArea(unionPhysicsAabb(objectList));
        const boundaryConstraints = constraintSimArea
            ? objectList
                .map(object => buildVelocityCuboidPreview(object, constraintSimArea))
                .filter((constraint): constraint is BoundaryConstraintPreview => !!constraint)
            : [];
        const constraintByObject = new Map(boundaryConstraints.map(constraint => [constraint.objectId, constraint]));
        for (const object of objectList) {
            object.boundaryDebug = constraintByObject.get(object.objectId) ?? null;
        }
        return {
            version: 'phys-ui-v1',
            solver,
            source: {
                splat: splat?.name ?? '',
                numSplats: splat?.splatData.numSplats ?? 0
            },
            modelId,
            officialConfig: {
                enabled: officialConfigEnabled,
                available: !!modelInfo?.officialConfigAvailable,
                name: modelInfo?.officialConfigName ?? '',
                values: modelInfo?.officialConfigValues ?? null
            },
            preprocessing: {
                rotation_degree: rotation.degree.slice(),
                rotation_axis: rotation.axis.slice(),
                sim_area: simArea?.slice() ?? null,
                scale,
                n_grid: nGrid,
                opacity_threshold: opacityThreshold
            },
            simulation: JSON.parse(JSON.stringify(simulation)),
            boundaryConstraints,
            objects: objectList,
            selectedCount
        };
    };

    const emit = () => {
        const payload = buildPayload();
        events.fire('physics.payload', payload);
        scheduleVoxelGroups(payload);
    };

    events.function('physics.payload', () => buildPayload());

    events.on('scene.clear', () => {
        loadedBasePlyUrl = '';
        localProxyCache.clear();
        liveProxyState = null;
        if (localProxyCacheTimer !== null) {
            window.clearTimeout(localProxyCacheTimer);
            localProxyCacheTimer = null;
        }
    });

    const apiUrl = (path: string) => {
        const base = apiBase.trim().replace(/\/+$/, '');
        if (!base) {
            return path;
        }
        if (base.endsWith('/api') && path.startsWith('/api/')) {
            return `${base}${path.slice(4)}`;
        }
        return `${base}${path}`;
    };

    const assetUrl = (url: string) => {
        if (!url) {
            return '';
        }
        if (/^https?:\/\//i.test(url)) {
            return url;
        }
        const base = apiBase.trim().replace(/\/api\/?$/, '').replace(/\/+$/, '') || window.location.origin;
        return new URL(url, `${base}/`).href;
    };

    scheduleVoxelGroups = (payload = buildPayload()) => {
        if (voxelGroupTimer !== null) {
            window.clearTimeout(voxelGroupTimer);
        }
        if (!payload.modelId) {
            return;
        }
        voxelGroupTimer = window.setTimeout(async () => {
            voxelGroupTimer = null;
            try {
                const response = await fetch(apiUrl('/api/preview/voxel-groups'), {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(buildPayload())
                });
                if (!response.ok) {
                    return;
                }
                events.fire('physics.voxelGroups', await response.json());
            } catch (_error) {
                // Grouping is a preview diagnostic; simulation submission should
                // not be blocked if the server is temporarily busy.
            }
        }, 500);
    };

    const applyOfficialConfigValues = (values: any) => {
        if (!values) {
            return;
        }
        const preprocessing = values.preprocessing ?? {};
        const officialSimulation = values.simulation ?? {};
        if (preprocessing.opacity_threshold !== undefined) {
            opacityThreshold = preprocessing.opacity_threshold;
        }
        if (preprocessing.n_grid !== undefined) {
            nGrid = preprocessing.n_grid;
        }
        if (preprocessing.scale !== undefined) {
            scale = preprocessing.scale;
        }
        simArea = Array.isArray(preprocessing.sim_area) ? preprocessing.sim_area.slice() : null;
        simAreaLocal = null;
        simAreaIndices = null;
        simulation = {
            ...simulation,
            gravityEnabled: !!officialSimulation.gravityEnabled,
            gravity: Array.isArray(officialSimulation.gravity) ? officialSimulation.gravity.slice() : [0, 0, 0],
            frame_dt: officialSimulation.frame_dt ?? simulation.frame_dt,
            substep_dt: officialSimulation.substep_dt ?? simulation.substep_dt,
            damping: officialSimulation.damping ?? officialSimulation.grid_v_damping_scale ?? simulation.damping
        };
    };

    const applyModelRecord = (record: any) => {
        modelInfo = record ?? null;
        modelId = record?.modelId ?? modelId;
        if (!modelInfo?.officialConfigAvailable) {
            officialConfigEnabled = false;
        }
        if (officialConfigEnabled) {
            applyOfficialConfigValues(modelInfo?.officialConfigValues);
        }
        events.fire('physics.modelInfo', modelInfo);
        emit();
        return modelInfo;
    };

    const loadMotionFromRun = async (run: any, options: { frameCount?: number, availableFrames?: number, includeBase?: boolean, includeProxy?: boolean } = {}) => {
        const result = run?.result ?? run;
        const runId = run?.runId ?? result?.runId;
        if (runId) {
            lastRunId = runId;
        }
        const manifestUrl = result?.manifestUrl ? assetUrl(result.manifestUrl) : '';
        const binaryUrl = runId ? assetUrl(`/api/runs/${encodeURIComponent(runId)}/motion.bin`) : result?.binaryUrl ? assetUrl(result.binaryUrl) : '';
        const indicesUrl = runId ? assetUrl(`/api/runs/${encodeURIComponent(runId)}/indices.bin`) : result?.indicesUrl ? assetUrl(result.indicesUrl) : '';
        const proxyMotionUrl = runId ? assetUrl(`/api/runs/${encodeURIComponent(runId)}/proxy_motion.bin`) : result?.proxyMotionUrl ? assetUrl(result.proxyMotionUrl) : '';
        const proxySkinningUrl = runId ? assetUrl(`/api/runs/${encodeURIComponent(runId)}/proxy_skinning.bin`) : result?.proxySkinningUrl ? assetUrl(result.proxySkinningUrl) : '';
        const basePlyUrl = result?.basePlyUrl ? assetUrl(result.basePlyUrl) : '';
        const files: { filename: string, url?: string, contents?: File, skipReorder?: boolean }[] = [];

        const shouldLoadBase = !!basePlyUrl && options.includeBase !== false && basePlyUrl !== loadedBasePlyUrl;
        if (shouldLoadBase) {
            events.fire('scene.clear');
            files.push({
                filename: result.basePlyName ?? 'point_cloud.ply',
                url: basePlyUrl,
                skipReorder: true
            });
            loadedBasePlyUrl = basePlyUrl;
        }
        if (manifestUrl && binaryUrl) {
            let manifestFile: { filename: string, url?: string, contents?: File } = {
                filename: 'motion.physmotion.json',
                url: manifestUrl
            };
            const limitedAvailableFrames = options.availableFrames ?? options.frameCount;
            if (limitedAvailableFrames !== undefined) {
                const response = await fetch(manifestUrl);
                if (!response.ok) {
                    throw new Error(`failed to read motion manifest: ${response.status} ${response.statusText}`);
                }
                const manifest = await response.json();
                const fullFrameCount = Number(manifest.frameCount) || limitedAvailableFrames;
                manifest.availableFrames = Math.max(1, Math.min(limitedAvailableFrames, fullFrameCount));
                manifestFile = {
                    filename: 'motion.physmotion.json',
                    contents: new File([JSON.stringify(manifest)], 'motion.physmotion.json', { type: 'application/json' })
                };
            }
            files.push(
                manifestFile,
                { filename: 'motion.bin', url: binaryUrl }
            );
            if (indicesUrl) {
                files.push({ filename: 'indices.bin', url: indicesUrl });
            }
            if (options.includeProxy === true && proxyMotionUrl && proxySkinningUrl) {
                files.push(
                    { filename: 'proxy_motion.bin', url: proxyMotionUrl },
                    { filename: 'proxy_skinning.bin', url: proxySkinningUrl }
                );
            }
            await events.invoke('physmotion.loadPackage', files);
            if (shouldLoadBase) {
                invalidateLocalProxyCache(250);
            }
        }
    };

    const projectDragMotion = (preview: DragPreviewRequest): DragProjection | null => {
        const splat = selectedSplat();
        if (!splat) {
            return null;
        }
        const splatData = splat.splatData;
        const x = splatData.getProp('x') as Float32Array;
        const y = splatData.getProp('y') as Float32Array;
        const z = splatData.getProp('z') as Float32Array;
        const hitIndex = Number(preview?.hitIndex);
        const fallbackWorld = Array.isArray(preview?.velocity) ? preview.velocity.slice(0, 3) : [0, 0, 0];
        const canProjectDrag = Number.isFinite(hitIndex) &&
            hitIndex >= 0 &&
            hitIndex < (x?.length ?? 0) &&
            Array.isArray(preview.screenStart) &&
            Array.isArray(preview.screenEnd) &&
            Array.isArray(preview.canvasSize);
        if (!canProjectDrag) {
            const duration = Math.max(Number(preview.duration) || 0.05, 1e-3);
            const displacementWorld = fallbackWorld.map(value => value * duration);
            return {
                displacement: vectorToSplatLocal(splat, displacementWorld),
                displacementWorld,
                velocity: vectorToSplatLocal(splat, fallbackWorld),
                velocityWorld: fallbackWorld,
                startWorld: null,
                endWorld: null,
                mode: 'fallback-localized',
                depth: null,
                worldPerPixel: null
            };
        }

        const pose = events.invoke('camera.getPose') as any;
        const cameraPosition = [
            Number(pose?.position?.x) || 0,
            Number(pose?.position?.y) || 0,
            Number(pose?.position?.z) || 0
        ];
        const cameraTarget = [
            Number(pose?.target?.x) || 0,
            Number(pose?.target?.y) || 0,
            Number(pose?.target?.z) || 0
        ];
        let forward = [
            cameraTarget[0] - cameraPosition[0],
            cameraTarget[1] - cameraPosition[1],
            cameraTarget[2] - cameraPosition[2]
        ];
        const forwardLength = Math.max(vec3Length(forward), 1e-12);
        forward[0] /= forwardLength;
        forward[1] /= forwardLength;
        forward[2] /= forwardLength;

        forward = normalizedPoseVector(pose?.forward, forward);
        let right = normalizedPoseVector(pose?.right, [forward[1], -forward[0], 0]);
        const rightLength = vec3Length(right);
        if (rightLength <= 1e-8) {
            right = [1, 0, 0];
        } else {
            right[0] /= rightLength;
            right[1] /= rightLength;
            right[2] /= rightLength;
        }
        let up = normalizedPoseVector(pose?.up, [
            right[1] * forward[2] - right[2] * forward[1],
            right[2] * forward[0] - right[0] * forward[2],
            right[0] * forward[1] - right[1] * forward[0]
        ]);
        const upLength = Math.max(vec3Length(up), 1e-12);
        up[0] /= upLength;
        up[1] /= upLength;
        up[2] /= upLength;

        pointScratch.set(x[hitIndex], y[hitIndex], z[hitIndex]);
        splat.worldTransform.transformPoint(pointScratch, rotatedScratch);
        const hitWorld = [rotatedScratch.x, rotatedScratch.y, rotatedScratch.z];
        const cameraToHit = [
            hitWorld[0] - cameraPosition[0],
            hitWorld[1] - cameraPosition[1],
            hitWorld[2] - cameraPosition[2]
        ];
        const depth = Math.max(
            cameraToHit[0] * forward[0] + cameraToHit[1] * forward[1] + cameraToHit[2] * forward[2],
            1e-4
        );
        const width = Math.max(Number(preview.canvasSize?.[0]) || 1, 1);
        const height = Math.max(Number(preview.canvasSize?.[1]) || 1, 1);
        const fov = Number(pose?.fov) || 60;
        const worldPerPixel = 2 * depth * Math.tan(fov * Math.PI / 360) / Math.max(height, 1);
        const duration = Math.max(Number(preview.duration) || 0.05, 1e-3);
        const dx = Number(preview.screenEnd?.[0]) - Number(preview.screenStart?.[0]);
        const dy = Number(preview.screenEnd?.[1]) - Number(preview.screenStart?.[1]);
        const gain = Math.max(0, Number(preview.dragGain) || 0.002) / 0.002;
        const displacementWorld = [
            (right[0] * dx - up[0] * dy) * worldPerPixel * gain,
            (right[1] * dx - up[1] * dy) * worldPerPixel * gain,
            (right[2] * dx - up[2] * dy) * worldPerPixel * gain
        ];
        const velocityWorld = [
            displacementWorld[0] / duration,
            displacementWorld[1] / duration,
            displacementWorld[2] / duration
        ];
        const startWorld = [
            hitWorld[0],
            hitWorld[1],
            hitWorld[2]
        ];
        const endWorld = [
            hitWorld[0] + displacementWorld[0],
            hitWorld[1] + displacementWorld[1],
            hitWorld[2] + displacementWorld[2]
        ];
        return {
            displacement: vectorToSplatLocal(splat, displacementWorld),
            displacementWorld,
            velocity: vectorToSplatLocal(splat, velocityWorld),
            velocityWorld,
            startWorld,
            endWorld,
            mode: 'screen-plane-displacement',
            depth,
            worldPerPixel
        };
    };

    const projectDragVelocity = (preview: DragPreviewRequest) => projectDragMotion(preview);

    events.function('physics.projectDragVelocity', (preview: DragPreviewRequest) => projectDragVelocity(preview));

    const resolveBodyIdForPreview = (preview: DragPreviewRequest) => {
        const hitIndex = Number(preview?.hitIndex);
        let objectId = Number(preview?.objectId);
        let bodyId = Number(preview?.bodyId);
        if (Number.isFinite(hitIndex) && hitIndex >= 0) {
            for (const object of objects.values()) {
                if (object.indices.includes(hitIndex)) {
                    objectId = object.objectId;
                    bodyId = object.bodyId ?? object.objectId;
                    break;
                }
            }
        }
        if (!Number.isFinite(bodyId) || bodyId <= 0) {
            const selectedObject = Number.isFinite(objectId) && objectId > 0 ? objects.get(objectId) : null;
            bodyId = selectedObject ? selectedObject.bodyId ?? selectedObject.objectId : 0;
        }
        return Number.isFinite(bodyId) && bodyId > 0 ? bodyId : 0;
    };

    const refreshProxyTemplateGeometry = (
        template: LocalProxyTemplate,
        x: Float32Array,
        y: Float32Array,
        z: Float32Array
    ) => {
        const groups = template.groups.map((group) => {
            const center = [0, 0, 0];
            for (const index of group.indices) {
                center[0] += x[index];
                center[1] += y[index];
                center[2] += z[index];
            }
            const inv = 1 / Math.max(group.indices.length, 1);
            center[0] *= inv;
            center[1] *= inv;
            center[2] *= inv;
            return {
                indices: group.indices,
                center
            };
        });
        return {
            groups,
            bounds: proxyBoundsForIndices(x, y, z, template.indices)
        };
    };

    const buildLocalProxyTemplate = (splat: Splat, requestedBodyId: number): LocalProxyTemplate | null => {
        const splatData = splat.splatData;
        const x = splatData.getProp('x') as Float32Array;
        const y = splatData.getProp('y') as Float32Array;
        const z = splatData.getProp('z') as Float32Array;
        if (!x || !y || !z || !x.length) {
            return null;
        }

        const bodyId = Number.isFinite(requestedBodyId) && requestedBodyId > 0 ? requestedBodyId : 0;
        const bodyParts = bodyId > 0
            ? Array.from(objects.values()).filter(object => (object.bodyId ?? object.objectId) === bodyId)
            : [];
        const nonObstacleParts = bodyParts.filter(object => object.mode !== 'obstacle');

        const sourceParts = bodyParts.length > 0 ? bodyParts : [];
        const sourceMaterials = (nonObstacleParts.length > 0 ? nonObstacleParts : sourceParts)
            .map(object => proxyMaterialForPart(object, bodyParts));
        const fallbackMaterial = averageProxyMaterial(sourceMaterials);
        const indexSet = new Set<number>();
        const indexMaterial = new Map<number, ProxyMaterial>();
        const fixedIndexSet = new Set<number>();
        const blockedIndices = new Set<number>();
        const movableBodyIds = new Set<number>();
        for (const object of objects.values()) {
            const partBodyId = object.bodyId ?? object.objectId;
            if (object.mode !== 'obstacle') {
                movableBodyIds.add(partBodyId);
            }
            if (partBodyId !== bodyId) {
                for (const index of object.indices) {
                    blockedIndices.add(index);
                }
            }
        }
        if (sourceParts.length > 0) {
            for (const object of sourceParts) {
                const material = proxyMaterialForPart(object, bodyParts);
                for (const index of object.indices) {
                    if (index >= 0 && index < x.length) {
                        indexSet.add(index);
                        indexMaterial.set(index, material);
                        if (object.mode === 'obstacle') {
                            fixedIndexSet.add(index);
                        }
                    }
                }
            }
            const sourceIndices = Array.from(indexSet);
            const bodyBounds = sourceIndices.length ? proxyBoundsForIndices(x, y, z, sourceIndices) : null;
            const adoptAllUnassigned = nonObstacleParts.length > 0 && movableBodyIds.size <= 1;
            for (let index = 0; index < x.length; ++index) {
                if (indexSet.has(index) || blockedIndices.has(index)) {
                    continue;
                }
                if (adoptAllUnassigned || (bodyBounds && expandedProxyBoundsContains(bodyBounds, [x[index], y[index], z[index]], 0.06))) {
                    indexSet.add(index);
                    indexMaterial.set(index, fallbackMaterial);
                }
            }
        } else {
            for (let index = 0; index < x.length; ++index) {
                indexSet.add(index);
                indexMaterial.set(index, fallbackMaterial);
            }
        }
        const indices = Array.from(indexSet).sort((a, b) => a - b);
        if (indices.length < 1) {
            return null;
        }

        const targetGroups = Math.max(24, Math.min(384, Math.round(Math.cbrt(indices.length) * 6)));
        const { groups, groupByIndex, bounds, voxelSize, kinematicGroups } = buildLocalProxyGrouping(x, y, z, indices, fixedIndexSet, targetGroups);
        if (!groups.length) {
            return null;
        }
        const restCenters = groups.map(group => group.center.slice(0, 3));
        const materials = groups.map(group => averageProxyMaterial(
            group.indices.map(index => indexMaterial.get(index) ?? fallbackMaterial)
        ));
        const masses = groups.map((group, groupId) => {
            const density = Math.max(materials[groupId]?.density ?? fallbackMaterial.density, 1);
            return Math.max(group.indices.length * density / 1000, 1e-4);
        });
        const blendCount = Math.max(1, Math.min(4, groups.length));
        return {
            bodyId,
            indices,
            groups,
            groupByIndex,
            materials,
            masses,
            kinematicGroups,
            bounds,
            voxelSize,
            gridRes: localProxyGridResolution(groups.length, voxelSize, bounds),
            blendCount,
            skinning: localProxySkinning(x, y, z, indices, restCenters, groupByIndex, blendCount, 2),
            shapeStiffness: proxyStructureStiffness(materials)
        };
    };

    const bodyIdsForLocalProxyCache = () => {
        if (!objects.size) {
            return [0];
        }
        return Array.from(new Set(
            Array.from(objects.values())
                .map(object => object.bodyId ?? object.objectId)
        )).sort((a, b) => a - b);
    };

    const rebuildLocalProxyCaches = async () => {
        const splat = selectedSplat();
        if (!splat) {
            localProxyCache.clear();
            return;
        }
        const nextCache = new Map<number, LocalProxyTemplate>();
        const bodyIds = bodyIdsForLocalProxyCache();
        for (const bodyId of bodyIds) {
            const template = buildLocalProxyTemplate(splat, bodyId);
            if (template) {
                nextCache.set(bodyId, template);
            }
            await sleep(0);
        }
        localProxyCache = nextCache;
        if (nextCache.size > 0) {
            const groupCount = Array.from(nextCache.values()).reduce((sum, template) => sum + template.groups.length, 0);
            events.fire('physics.status', `proxy cache ready: ${nextCache.size} bodies / ${groupCount} groups`);
        }
    };

    scheduleLocalProxyCache = (delay = 120) => {
        if (localProxyCacheTimer !== null) {
            window.clearTimeout(localProxyCacheTimer);
        }
        localProxyCacheTimer = window.setTimeout(() => {
            localProxyCacheTimer = null;
            void rebuildLocalProxyCaches();
        }, delay);
    };

    const invalidateLocalProxyCache = (delay = 120) => {
        localProxyCache.clear();
        liveProxyState = null;
        scheduleLocalProxyCache(delay);
    };

    const createLocalProxyState = (preview: DragPreviewRequest): LocalProxyState | null => {
        const splat = selectedSplat();
        if (!splat) {
            return null;
        }
        const splatData = splat.splatData;
        const x = splatData.getProp('x') as Float32Array;
        const y = splatData.getProp('y') as Float32Array;
        const z = splatData.getProp('z') as Float32Array;
        if (!x || !y || !z || !x.length) {
            return null;
        }

        const hitIndex = Number(preview?.hitIndex);
        const bodyId = resolveBodyIdForPreview(preview);
        let template = localProxyCache.get(bodyId);
        if (!template) {
            template = buildLocalProxyTemplate(splat, bodyId);
            if (template) {
                localProxyCache.set(bodyId, template);
            }
        }
        if (!template) {
            events.fire('physics.status', 'live proxy skipped: no movable proxy body');
            return null;
        }

        const indices = template.indices;
        const { groups, bounds } = refreshProxyTemplateGeometry(template, x, y, z);
        const groupByIndex = template.groupByIndex;

        let hitGroup: number | null = null;
        const hasHit = Number.isFinite(hitIndex) && hitIndex >= 0;
        if (!hasHit) {
            events.fire('physics.status', 'live proxy skipped: click a Gaussian to grab');
            return null;
        }
        if (hasHit) {
            hitGroup = groupByIndex.get(hitIndex) ?? null;
            if (hitGroup === null) {
                events.fire('physics.status', 'live proxy skipped: hit Gaussian is not in the movable body');
                return null;
            }
        }

        const restCenters = groups.map(group => group.center.slice(0, 3));

        const restPositions = new Float32Array(indices.length * 3);
        const restRotations = new Float32Array(indices.length * 4);
        const restScales = new Float32Array(indices.length * 3);
        const r0 = splatData.getProp('rot_0') as Float32Array;
        const r1 = splatData.getProp('rot_1') as Float32Array;
        const r2 = splatData.getProp('rot_2') as Float32Array;
        const r3 = splatData.getProp('rot_3') as Float32Array;
        const s0 = splatData.getProp('scale_0') as Float32Array;
        const s1 = splatData.getProp('scale_1') as Float32Array;
        const s2 = splatData.getProp('scale_2') as Float32Array;
        for (let order = 0; order < indices.length; ++order) {
            const index = indices[order];
            const dst3 = order * 3;
            const dst4 = order * 4;
            restPositions[dst3] = x[index];
            restPositions[dst3 + 1] = y[index];
            restPositions[dst3 + 2] = z[index];
            restRotations[dst4] = r0[index];
            restRotations[dst4 + 1] = r1[index];
            restRotations[dst4 + 2] = r2[index];
            restRotations[dst4 + 3] = r3[index];
            restScales[dst3] = s0[index];
            restScales[dst3 + 1] = s1[index];
            restScales[dst3 + 2] = s2[index];
        }

        const diag = Math.hypot(
            bounds.maxs[0] - bounds.mins[0],
            bounds.maxs[1] - bounds.mins[1],
            bounds.maxs[2] - bounds.mins[2]
        );
        const gravityWorld = simulation.gravityEnabled && Array.isArray(simulation.gravity) ? simulation.gravity.slice(0, 3) : [0, 0, 0];
        const gravity = vectorToSplatLocal(splat, gravityWorld);
        const groundHeight = Number(simulation.groundHeight);
        const localRotation = splat.entity.getLocalRotation();
        const canUseLocalGround = Math.hypot(localRotation.x, localRotation.y, localRotation.z) <= 1e-4;
        const groundZ = canUseLocalGround &&
            simulation.groundEnabled &&
            Number.isFinite(groundHeight) &&
            groundHeight >= bounds.mins[2] - diag * 0.2 &&
            groundHeight <= bounds.maxs[2] + diag * 0.2
            ? groundHeight
            : null;

        return {
            splat,
            indices,
            indexArray: new Uint32Array(indices),
            groups,
            groupByIndex,
            restCenters,
            centers: restCenters.map(center => center.slice(0, 3)),
            velocities: groups.map(() => [0, 0, 0]),
            affines: groups.map(() => new Array(9).fill(0)),
            deformations: groups.map(() => identityMatrix3()),
            rotations: groups.map(() => [1, 0, 0, 0]),
            masses: template.masses,
            materials: template.materials.map(material => ({
                ...material,
                stressScale: Math.max(0.004, Math.min(material.stressScale * 0.1, 0.035))
            })),
            bounds,
            voxelSize: template.voxelSize,
            gridRes: localProxyGridResolution(groups.length, template.voxelSize, bounds),
            blendCount: template.blendCount,
            skinning: template.skinning,
            restPositions,
            restRotations,
            restScales,
            hitGroup,
            kinematicGroups: new Set(template.kinematicGroups),
            gravity,
            groundZ,
            shapeStiffness: Math.max(0.035, Math.min(template.shapeStiffness * 0.28, 0.14)),
            maxDriveSpeed: Math.max(diag * 0.28, 1e-4),
            dragDisplacement: [0, 0, 0],
            lastTime: performance.now(),
            liveBusy: false
        };
    };

    const stepLiveProxyState = async (preview: DragPreviewRequest) => {
        const state = liveProxyState;
        if (!state || state.liveBusy) {
            return false;
        }
        const now = performance.now();
        const elapsed = Math.max((now - state.lastTime) / 1000, 1 / 120);
        state.lastTime = now;
        const frameDt = Math.max(1 / 240, Math.min(elapsed, 1 / 24));
        const substeps = Math.max(1, Math.min(4, Math.ceil(frameDt / (1 / 120))));
        const subDt = frameDt / substeps;
        const diag = Math.hypot(
            state.bounds.maxs[0] - state.bounds.mins[0],
            state.bounds.maxs[1] - state.bounds.mins[1],
            state.bounds.maxs[2] - state.bounds.mins[2]
        );
        const projected = projectDragMotion(preview);
        const fallbackVelocity = vectorToSplatLocal(
            state.splat,
            Array.isArray(preview?.velocity) ? preview.velocity.slice(0, 3) : [0, 0, 0]
        );
        const dragDelta = clampVectorLength(
            projected?.displacement ?? fallbackVelocity.map(value => value * frameDt),
            Math.max(diag * 0.035, 1e-5)
        );
        state.dragDisplacement = clampVectorLength([
            state.dragDisplacement[0] + dragDelta[0],
            state.dragDisplacement[1] + dragDelta[1],
            state.dragDisplacement[2] + dragDelta[2]
        ], Math.max(diag * 0.22, 1e-4));
        const mouseVelocity = clampVectorLength(
            projected?.velocity ?? dragDelta.map(value => value / Math.max(Number(preview.duration) || frameDt, 1e-3)),
            state.maxDriveSpeed
        );
        const grabbedGroup = state.hitGroup;
        const grabbedTarget = grabbedGroup === null ? null : clampPointToProxyBounds([
            state.restCenters[grabbedGroup][0] + state.dragDisplacement[0],
            state.restCenters[grabbedGroup][1] + state.dragDisplacement[1],
            state.restCenters[grabbedGroup][2] + state.dragDisplacement[2]
        ], state.bounds, 0.08);

        state.liveBusy = true;
        try {
            for (let stepId = 0; stepId < substeps; ++stepId) {
                const driveVelocity = grabbedGroup === null ? null : mouseVelocity;
                if (grabbedGroup !== null && grabbedTarget) {
                    state.centers[grabbedGroup][0] = grabbedTarget[0];
                    state.centers[grabbedGroup][1] = grabbedTarget[1];
                    state.centers[grabbedGroup][2] = grabbedTarget[2];
                    state.velocities[grabbedGroup][0] = mouseVelocity[0];
                    state.velocities[grabbedGroup][1] = mouseVelocity[1];
                    state.velocities[grabbedGroup][2] = mouseVelocity[2];
                }
                const step = stepLocalProxyMpm(
                    state.centers,
                    state.velocities,
                    state.affines,
                    state.deformations,
                    state.masses,
                    state.materials,
                    state.bounds,
                    state.gridRes,
                    subDt,
                    state.gravity,
                    state.groundZ,
                    state.hitGroup,
                    driveVelocity,
                    state.kinematicGroups
                );
                state.centers = step.centers.map(center => clampPointToProxyBounds(center, state.bounds, 0.1));
                state.velocities = shapeCoupleProxyVelocities(
                    state.centers,
                    step.velocities.map(velocity => clampVectorLength(velocity, Math.max(state.maxDriveSpeed * 1.25, 1e-4))),
                    state.restCenters,
                    state.masses,
                    subDt,
                    state.shapeStiffness
                );
                state.velocities = state.velocities.map(velocity => clampVectorLength(velocity, Math.max(state.maxDriveSpeed * 1.25, 1e-4)));
                const pulled = pullProxyCentersTowardGrabbedShape(
                    state.centers,
                    state.velocities,
                    state.restCenters,
                    grabbedGroup,
                    grabbedTarget,
                    mouseVelocity,
                    subDt,
                    0.10,
                    state.maxDriveSpeed
                );
                state.centers = pulled.centers.map(center => clampPointToProxyBounds(center, state.bounds, 0.1));
                state.velocities = pulled.velocities.map(velocity => clampVectorLength(velocity, Math.max(state.maxDriveSpeed * 1.35, 1e-4)));
                state.affines = step.affines.map(matrix => dampProxyAffine(matrix, 0.18, 8));
                state.deformations = step.deformations.map(matrix => dampProxyDeformation(matrix, 0.35, 0.12));
                state.rotations = state.rotations.map(() => [1, 0, 0, 0]);
                if (grabbedGroup !== null && grabbedTarget) {
                    state.centers[grabbedGroup][0] = grabbedTarget[0];
                    state.centers[grabbedGroup][1] = grabbedTarget[1];
                    state.centers[grabbedGroup][2] = grabbedTarget[2];
                    state.velocities[grabbedGroup][0] = mouseVelocity[0];
                    state.velocities[grabbedGroup][1] = mouseVelocity[1];
                    state.velocities[grabbedGroup][2] = mouseVelocity[2];
                }
                for (const groupId of state.kinematicGroups) {
                    if (groupId === grabbedGroup) {
                        continue;
                    }
                    state.centers[groupId] = state.restCenters[groupId].slice(0, 3);
                    state.velocities[groupId] = [0, 0, 0];
                    state.affines[groupId] = new Array(9).fill(0);
                    state.deformations[groupId] = identityMatrix3();
                }
            }
            await state.splat.applyMotionFrame(localProxyFrameFromState(state));
        } finally {
            state.liveBusy = false;
        }
        const maxSpeed = state.velocities.reduce((currentMax, velocity) => Math.max(currentMax, vec3Length(velocity)), 0);
        return {
            active: true,
            maxSpeed,
            settled: maxSpeed <= Math.max(state.maxDriveSpeed * 0.018, 1e-5)
        };
    };

    events.function('physics.liveProxyDragStart', (preview: DragPreviewRequest) => {
        liveProxyState = createLocalProxyState(preview);
        if (!liveProxyState) {
            return null;
        }
        events.fire(
            'physics.status',
            `live proxy ready: ${liveProxyState.groups.length} groups, grid ${liveProxyState.gridRes}, ${liveProxyState.indices.length} splats, material-aware`
        );
        return {
            groupCount: liveProxyState.groups.length,
            gridRes: liveProxyState.gridRes,
            splatCount: liveProxyState.indices.length,
            hitGroup: liveProxyState.hitGroup
        };
    });

    events.function('physics.liveProxyDragMove', (preview: DragPreviewRequest) => stepLiveProxyState(preview));

    events.function('physics.liveProxyDragEnd', () => {
        const hadState = !!liveProxyState;
        liveProxyState = null;
        return hadState;
    });

    const tryLocalProxyDragPreview = async (preview: DragPreviewRequest) => {
        const splat = selectedSplat();
        if (!splat) {
            return false;
        }
        const splatData = splat.splatData;
        const x = splatData.getProp('x') as Float32Array;
        const y = splatData.getProp('y') as Float32Array;
        const z = splatData.getProp('z') as Float32Array;
        if (!x || !y || !z || !x.length) {
            return false;
        }

        const hitIndex = Number(preview?.hitIndex);
        let objectId = Number(preview?.objectId);
        let bodyId = Number(preview?.bodyId);
        if (Number.isFinite(hitIndex) && hitIndex >= 0) {
            for (const object of objects.values()) {
                if (object.indices.includes(hitIndex)) {
                    objectId = object.objectId;
                    bodyId = object.bodyId ?? object.objectId;
                    break;
                }
            }
        }
        if (!Number.isFinite(bodyId) || bodyId <= 0) {
            const selectedObject = Number.isFinite(objectId) && objectId > 0 ? objects.get(objectId) : null;
            bodyId = selectedObject ? selectedObject.bodyId ?? selectedObject.objectId : 0;
        }

        const bodyParts = bodyId > 0
            ? Array.from(objects.values()).filter(object => (object.bodyId ?? object.objectId) === bodyId)
            : [];
        const nonObstacleParts = bodyParts.filter(object => object.mode !== 'obstacle');

        const indexSet = new Set<number>();
        const sourceParts = bodyParts.length > 0 ? bodyParts : [];
        const sourceMaterials = (nonObstacleParts.length > 0 ? nonObstacleParts : sourceParts)
            .map(object => proxyMaterialForPart(object, bodyParts));
        const fallbackMaterial = averageProxyMaterial(sourceMaterials);
        const indexMaterial = new Map<number, ProxyMaterial>();
        const fixedIndexSet = new Set<number>();
        if (sourceParts.length > 0) {
            for (const object of sourceParts) {
                const material = proxyMaterialForPart(object, bodyParts);
                for (const index of object.indices) {
                    if (index >= 0 && index < x.length) {
                        indexSet.add(index);
                        indexMaterial.set(index, material);
                        if (object.mode === 'obstacle') {
                            fixedIndexSet.add(index);
                        }
                    }
                }
            }
        } else {
            for (let index = 0; index < x.length; ++index) {
                indexSet.add(index);
                indexMaterial.set(index, fallbackMaterial);
            }
        }
        const indices = Array.from(indexSet).sort((a, b) => a - b);
        if (indices.length < 1) {
            return false;
        }

        const targetGroups = Math.max(24, Math.min(384, Math.round(Math.cbrt(indices.length) * 6)));
        const { groups, groupByIndex, bounds, voxelSize, kinematicGroups } = buildLocalProxyGrouping(x, y, z, indices, fixedIndexSet, targetGroups);
        if (!groups.length) {
            return false;
        }
        const groupCenters = groups.map(group => group.center.slice(0, 3));
        const restCenters = groupCenters.map(center => center.slice(0, 3));
        const groupMaterials = groups.map(group => averageProxyMaterial(
            group.indices.map(index => indexMaterial.get(index) ?? fallbackMaterial)
        ));
        const masses = groups.map((group, groupId) => {
            const density = Math.max(groupMaterials[groupId]?.density ?? fallbackMaterial.density, 1);
            return Math.max(group.indices.length * density / 1000, 1e-4);
        });
        const shapeStiffness = proxyStructureStiffness(groupMaterials);
        const blendCount = Math.max(1, Math.min(4, groups.length));
        const skinning = localProxySkinning(x, y, z, indices, restCenters, groupByIndex, blendCount, 2);
        let hitGroup: number | null = null;
        const hasHit = Number.isFinite(hitIndex) && hitIndex >= 0;
        if (hasHit) {
            hitGroup = groupByIndex.get(hitIndex) ?? null;
            if (hitGroup === null) {
                events.fire('physics.status', 'local proxy skipped: hit Gaussian is not in the movable body');
                return false;
            }
        }

        const frameDt = Math.max(1 / 90, Math.min(simulation.frame_dt || 1 / 30, 1 / 24));
        const frameCount = 18;
        const totalTime = Math.max((frameCount - 1) * frameDt, frameDt);
        const diag = Math.hypot(
            bounds.maxs[0] - bounds.mins[0],
            bounds.maxs[1] - bounds.mins[1],
            bounds.maxs[2] - bounds.mins[2]
        );
        const projected = projectDragMotion(preview);
        const rawVelocity = projected?.velocity ?? vectorToSplatLocal(
            splat,
            Array.isArray(preview?.velocity) ? preview.velocity.slice(0, 3) : [0, 0, 0]
        );
        const velocityMode = projected?.mode ?? 'fallback-localized';
        const displacement = clampVectorLength(
            projected?.displacement ?? [rawVelocity[0] * totalTime, rawVelocity[1] * totalTime, rawVelocity[2] * totalTime],
            Math.max(diag * 0.28, 1e-4)
        );
        const driveVelocity = displacement.map(value => value / totalTime);
        const gravityWorld = simulation.gravityEnabled && Array.isArray(simulation.gravity) ? simulation.gravity.slice(0, 3) : [0, 0, 0];
        const gravity = vectorToSplatLocal(splat, gravityWorld);
        const groundHeight = Number(simulation.groundHeight);
        const localRotation = splat.entity.getLocalRotation();
        const canUseLocalGround = Math.hypot(localRotation.x, localRotation.y, localRotation.z) <= 1e-4;
        const groundZ = canUseLocalGround &&
            simulation.groundEnabled &&
            Number.isFinite(groundHeight) &&
            groundHeight >= bounds.mins[2] - diag * 0.2 &&
            groundHeight <= bounds.maxs[2] + diag * 0.2
            ? groundHeight
            : null;
        const gridRes = localProxyGridResolution(groups.length, voxelSize, bounds);

        let centers = groupCenters.map(center => center.slice(0, 3));
        let velocities = groups.map(() => hitGroup === null ? driveVelocity.slice(0, 3) : [0, 0, 0]);
        let affines = groups.map(() => new Array(9).fill(0));
        let deformations = groups.map(() => identityMatrix3());
        let rotations = groups.map(() => [1, 0, 0, 0]);
        const proxyFrames = new Float32Array(frameCount * groups.length * 7);
        const writeProxyFrame = (frame: number) => {
            const frameOffset = frame * groups.length * 7;
            for (let groupId = 0; groupId < groups.length; ++groupId) {
                const centerOffset = frameOffset + groupId * 3;
                proxyFrames[centerOffset] = centers[groupId][0];
                proxyFrames[centerOffset + 1] = centers[groupId][1];
                proxyFrames[centerOffset + 2] = centers[groupId][2];
                const rotationOffset = frameOffset + groups.length * 3 + groupId * 4;
                proxyFrames[rotationOffset] = rotations[groupId][0];
                proxyFrames[rotationOffset + 1] = rotations[groupId][1];
                proxyFrames[rotationOffset + 2] = rotations[groupId][2];
                proxyFrames[rotationOffset + 3] = rotations[groupId][3];
            }
        };
        writeProxyFrame(0);
        for (let frame = 1; frame < frameCount; ++frame) {
            const step = stepLocalProxyMpm(
                centers,
                velocities,
                affines,
                deformations,
                masses,
                groupMaterials,
                bounds,
                gridRes,
                frameDt,
                gravity,
                groundZ,
                hitGroup,
                hitGroup === null ? null : driveVelocity,
                kinematicGroups
            );
            centers = step.centers;
            velocities = shapeCoupleProxyVelocities(
                centers,
                step.velocities,
                restCenters,
                masses,
                frameDt,
                shapeStiffness
            );
            affines = step.affines;
            deformations = step.deformations;
            rotations = rotations.map((rotation, groupId) => quatMulArray(step.deltaRotations[groupId], rotation));
            writeProxyFrame(frame);
        }

        const groupIds = new Uint32Array(indices.length * blendCount);
        const weights = new Float32Array(indices.length * blendCount);
        for (let order = 0; order < indices.length; ++order) {
            const blends = (skinning.get(indices[order]) ?? [[groupByIndex.get(indices[order]) ?? 0, 1]]) as [number, number][];
            for (let blend = 0; blend < blendCount; ++blend) {
                const item = blends[blend] ?? [blends[blends.length - 1]?.[0] ?? 0, 0];
                groupIds[order * blendCount + blend] = Math.max(0, item[0]);
                weights[order * blendCount + blend] = item[1];
            }
        }
        const skinningBuffer = new ArrayBuffer(groupIds.byteLength + weights.byteLength);
        new Uint32Array(skinningBuffer, 0, groupIds.length).set(groupIds);
        new Float32Array(skinningBuffer, groupIds.byteLength, weights.length).set(weights);

        const manifest = {
            format: 'phys-motion-v1',
            binary: 'motion.bin',
            indices: 'indices.bin',
            proxy: {
                motion: 'proxy_motion.bin',
                skinning: 'proxy_skinning.bin',
                groupCount: groups.length,
                blendCount,
                motionStrideBytes: groups.length * 7 * 4,
                skinningStrideBytes: blendCount * 8
            },
            frameCount,
            frameRate: Math.round(1 / frameDt),
            numSplats: splatData.numSplats,
            attributes: ['position', 'rotation', 'scale'],
            frameStrideBytes: indices.length * 10 * 4,
            updateBounds: false
        };
        const files = [
            new File([JSON.stringify(manifest)], 'motion.physmotion.json', { type: 'application/json' }),
            new File([new ArrayBuffer(0)], 'motion.bin'),
            new File([new Uint32Array(indices).buffer], 'indices.bin'),
            new File([proxyFrames.buffer], 'proxy_motion.bin'),
            new File([skinningBuffer], 'proxy_skinning.bin')
        ].map(file => ({ filename: file.name, contents: file }));

        await events.invoke('physmotion.loadPackage', files);
        events.fire(
            'physics.status',
            `local proxy preview: ${groups.length} groups, grid ${gridRes}, ${indices.length} splats, ${velocityMode}, material-aware`
        );
        return true;
    };

    const pollRun = async (runId: string, options: { includeProxy?: boolean } = {}) => {
        currentRunId = runId;
        let streamLoaded = false;
        let streamedFrames = 0;
        for (let i = 0; i < 3600; ++i) {
            const response = await fetch(apiUrl(`/api/runs/${encodeURIComponent(runId)}`));
            if (!response.ok) {
                throw new Error(`读取运行状态失败：${response.status} ${response.statusText}`);
            }
            const run = await response.json();
            const statusText = run.status === 'completed' ? '完成' :
                run.status === 'failed' ? '失败' :
                    run.status === 'running' ? '运行中' :
                        run.status === 'queued' ? '排队中' :
                            run.status;
            const progressText = run.progress === 'physgaussian' ? 'PhysGaussian 求解中' :
                run.progress === 'motion-ready' ? '动画已生成' :
                    run.progress === 'failed' ? '求解失败' :
                        run.progress;
            events.fire('physics.status', `${statusText}${progressText ? `：${progressText}` : ''}`);

            const availableFrames = Math.max(0, Math.min(run.availableFrames ?? 0, run.frameCount ?? Number.MAX_SAFE_INTEGER));
            const shouldRefreshStream = run.manifestReady &&
                run.indicesReady &&
                availableFrames > 0 &&
                (!streamLoaded || availableFrames >= streamedFrames + 4);
            if (shouldRefreshStream) {
                if (!streamLoaded) {
                    await loadMotionFromRun(run, {
                        availableFrames,
                        includeBase: true,
                        includeProxy: options.includeProxy === true
                    });
                } else {
                    events.invoke('physmotion.extendAvailableFrames', availableFrames);
                }
                streamLoaded = true;
                streamedFrames = availableFrames;
                events.fire('physics.status', `streaming preview: ${availableFrames}/${run.frameCount ?? '?'} frames`);
            }

            if (run.status === 'completed') {
                if (streamLoaded) {
                    events.invoke('physmotion.extendAvailableFrames', run.frameCount ?? availableFrames);
                } else {
                    await loadMotionFromRun(run, {
                        includeBase: true,
                        includeProxy: options.includeProxy === true
                    });
                }
                currentRunId = '';
                return run;
            }
            if (run.status === 'failed') {
                currentRunId = '';
                throw new Error(run.error ? `仿真失败：${run.error}` : '仿真失败');
            }
            if (run.status === 'cancelled') {
                currentRunId = '';
                throw new Error('simulation cancelled');
            }
            await sleep(2000);
        }
        currentRunId = '';
        throw new Error('等待仿真超时');
    };

    const selectObjectInScene = (objectId: number) => {
        const object = objects.get(Number(objectId));
        const splat = selectedSplat();
        if (!object || !splat) {
            return;
        }
        const mask = new Uint8Array(splat.splatData.numSplats);
        for (const index of object.indices) {
            if (index >= 0 && index < mask.length) {
                mask[index] = 255;
            }
        }
        events.fire('select.mask', 'set', mask);
        selectedCount = object.indices.length;
        events.fire('physics.selectionCount', selectedCount);
    };

    events.on('physics.setPreprocess', (patch: Partial<PhysicsPayload['preprocessing']>) => {
        if (patch.sim_area !== undefined) {
            simArea = patch.sim_area ? patch.sim_area.slice() : null;
            simAreaLocal = null;
            simAreaIndices = null;
        }
        if (patch.scale !== undefined) scale = patch.scale;
        if (patch.n_grid !== undefined) nGrid = patch.n_grid;
        if (patch.opacity_threshold !== undefined) opacityThreshold = patch.opacity_threshold;
        emit();
        invalidateLocalProxyCache(120);
    });

    events.on('physics.setApiBase', (value: string) => {
        apiBase = value ?? '';
        events.fire('physics.status', apiBase ? `接口：${apiBase}` : '');
    });

    events.on('physics.setModelId', (value: string) => {
        modelId = value ?? '';
        emit();
    });

    events.on('physics.setSolver', (value: PhysicsPayload['solver']) => {
        solver = value ?? 'explicit-mpm';
        events.fire('physics.status', `求解器：${solver}`);
        emit();
    });

    events.on('physics.setOfficialConfig', (enabled: boolean) => {
        officialConfigEnabled = !!enabled && !!modelInfo?.officialConfigAvailable;
        if (enabled && !modelInfo?.officialConfigAvailable) {
            events.fire('physics.status', '当前模型没有绑定官方配置');
        }
        if (officialConfigEnabled) {
            applyOfficialConfigValues(modelInfo?.officialConfigValues);
            events.fire('physics.status', `已启用官方配置：${modelInfo?.officialConfigName ?? ''}`);
        }
        emit();
    });

    events.function('physics.listModels', async () => {
        const response = await fetch(apiUrl('/api/models'));
        if (!response.ok) {
            throw new Error(`读取云端模型失败：${response.status} ${response.statusText}`);
        }
        const records = await response.json();
        events.fire('physics.models', records);
        return records;
    });

    events.function('physics.loadModelById', async (value?: string) => {
        const targetModelId = (value ?? modelId ?? '').trim();
        if (!targetModelId) {
            throw new Error('模型ID为空');
        }
        events.fire('physics.status', `正在加载模型 ${targetModelId}...`);
        const response = await fetch(apiUrl(`/api/models/${encodeURIComponent(targetModelId)}`));
        if (!response.ok) {
            throw new Error(`查找模型失败：${response.status} ${response.statusText}`);
        }
        const result = await response.json();
        applyModelRecord(result);

        if (!result.basePlyUrl) {
            throw new Error(`模型 ${modelId} 没有预览 PLY`);
        }
        const nextBasePlyUrl = assetUrl(result.basePlyUrl);
        if (nextBasePlyUrl !== loadedBasePlyUrl) {
            events.fire('scene.clear');
            await events.invoke('import', [{
                filename: result.basePlyName ?? 'point_cloud.ply',
                url: nextBasePlyUrl,
                skipReorder: true
            }]);
            loadedBasePlyUrl = nextBasePlyUrl;
        }
        invalidateLocalProxyCache(250);
        events.fire('physics.status', `模型已加载：${modelId}`);
        return result;
    });

    events.on('physics.setSimulation', (patch: Partial<PhysicsPayload['simulation']>) => {
        simulation = {
            ...simulation,
            ...patch,
            pbmpm: {
                ...(simulation.pbmpm ?? {}),
                ...(patch.pbmpm ?? {})
            }
        };
        emit();
    });

    events.function('physics.uploadModel', async (file: File, options: { importPreview?: boolean } = {}) => {
        const importPreview = options.importPreview ?? true;
        if (syncedFiles.has(file)) {
            modelId = syncedFiles.get(file);
            emit();
            events.fire('physics.status', modelId ? `模型：${modelId}` : '模型已同步');
            return { modelId };
        }
        const form = new FormData();
        form.append('file', file);
        events.fire('physics.status', '正在上传模型...');
        const response = await fetch(apiUrl('/api/models'), {
            method: 'POST',
            body: form
        });
        if (!response.ok) {
            throw new Error(`上传模型失败：${response.status} ${response.statusText}`);
        }
        const result = await response.json();
        applyModelRecord(result);
        if (modelId) {
            syncedFiles.set(file, modelId);
        }

        if (importPreview && result.basePlyUrl) {
            const nextBasePlyUrl = assetUrl(result.basePlyUrl);
            events.fire('physics.status', '正在加载预览 PLY...');
            await events.invoke('import', [{
                filename: result.basePlyName ?? 'point_cloud.ply',
                url: nextBasePlyUrl,
                skipReorder: true
            }]);
            loadedBasePlyUrl = nextBasePlyUrl;
            invalidateLocalProxyCache(250);
        }

        events.fire('physics.status', modelId ? `模型：${modelId}` : '模型已上传');
        return result;
    });

    events.function('physics.syncImportedFile', async (file: { filename: string, contents?: File }) => {
        const contents = file?.contents;
        if (!contents || !file.filename.toLowerCase().endsWith('.ply')) {
            return null;
        }
        try {
            events.fire('physics.status', `正在同步 ${file.filename}...`);
            return await events.invoke('physics.uploadModel', contents, { importPreview: false });
        } catch (error) {
            events.fire('physics.status', `同步失败：${String(error?.message ?? error)}`);
            return null;
        }
    });

    events.on('physics.useSelectionAabb', () => {
        const summary = events.invoke('phys.selectionSummary') as any;
        if (!summary || !summary.aabbWorld) {
            return;
        }
        const indices = selectedSplat() ? selectedIndicesFor(selectedSplat()) : [];
        const physicsAabb = toPhysicsAabbFromIndices(indices) ?? toPhysicsAabb(summary.aabbLocal) ?? summary.aabbWorld;
        setSimAreaFromAabb(physicsAabb, indices);
    });

    events.on('physics.useCurrentBodyAabb', (objectId: number) => {
        const object = objects.get(Number(objectId));
        if (!object) {
            return;
        }
        const bodyId = object.bodyId ?? object.objectId;
        const bodyParts = Array.from(objects.values()).filter(part => (part.bodyId ?? part.objectId) === bodyId);
        const indices = Array.from(new Set(bodyParts.flatMap(part => part.indices))).sort((a, b) => a - b);
        const aabb = toPhysicsAabbFromIndices(indices);
        if (!aabb) {
            return;
        }
        setSimAreaFromAabb(aabb, indices);
        events.fire('physics.status', `已使用 Body ${bodyId} 的 AABB`);
    });

    events.on('physics.createObjectFromSelection', (options: { name?: string, material: MaterialPreset, fill: boolean, bodyId?: number | string, density?: number }) => {
        const splat = selectedSplat();
        if (!splat) {
            return;
        }
        const indices = selectedIndicesFor(splat);
        if (!indices.length) {
            return;
        }

        const material = options.material ?? 'jelly';
        const defaults = materialDefaults[material] ?? materialDefaults.jelly;
        const objectId = nextObjectId++;
        const newIndexSet = new Set(indices);
        let conflictCount = 0;
        for (const [existingId, existing] of Array.from(objects)) {
            const filtered = existing.indices.filter(index => !newIndexSet.has(index));
            conflictCount += existing.indices.length - filtered.length;
            if (filtered.length === 0) {
                objects.delete(existingId);
                continue;
            }
            if (filtered.length !== existing.indices.length) {
                objects.set(existingId, {
                    ...existing,
                    indices: filtered,
                    count: filtered.length,
                    aabbWorld: toPhysicsAabbFromIndices(filtered) ?? existing.aabbWorld
                });
            }
        }
        const requestedBodyId = Number(options.bodyId);
        const bodyId = Number.isFinite(requestedBodyId) && requestedBodyId > 0 ? requestedBodyId : nextBodyId++;
        nextBodyId = Math.max(nextBodyId, bodyId + 1);
        const summary = events.invoke('phys.selectionSummary') as any;
        const physicsAabb = toPhysicsAabbFromIndices(indices) ?? toPhysicsAabb(summary?.aabbLocal) ?? summary?.aabbWorld;
        objects.set(objectId, {
            objectId,
            bodyId,
            name: options.name || `object_${objectId}`,
            material,
            density: Number.isFinite(Number(options.density)) ? Math.max(Number(options.density), 0) : defaults.density,
            E: defaults.E,
            nu: defaults.nu,
            mode: defaults.mode,
            constraintMode: material === 'obstacle' ? 'grid-velocity-cuboid' : undefined,
            cuboidPaddingGridNodes: material === 'obstacle' ? 2 : undefined,
            fill: material === 'obstacle' ? false : !!options.fill,
            drive: defaultDrive(),
            count: indices.length,
            indices,
            aabbWorld: physicsAabb,
            aabbLocal: summary?.aabbLocal
        });
        emit();
        invalidateLocalProxyCache(120);
        if (conflictCount > 0) {
            events.fire('physics.status', `新 Part 覆盖了旧 Part 中的 ${conflictCount} 个 Gaussian`);
        }
    });

    events.on('physics.updateObject', (objectId: number, patch: Partial<PhysicsObject>) => {
        const object = objects.get(objectId);
        if (!object) {
            return;
        }
        const material = (patch.material ?? object.material) as MaterialPreset;
        const defaults = materialDefaults[material] ?? materialDefaults.jelly;
        const requestedBodyId = Number(patch.bodyId ?? object.bodyId ?? objectId);
        const bodyId = Number.isFinite(requestedBodyId) && requestedBodyId > 0 ? requestedBodyId : object.bodyId ?? objectId;
        nextBodyId = Math.max(nextBodyId, bodyId + 1);
        const nextDrive = {
            ...object.drive,
            ...(patch.drive ?? {})
        };
        if (material === 'obstacle') {
            nextDrive.spinEnabled = false;
            nextDrive.spinAngular = 0;
        }
        const nextObject = {
            ...object,
            ...patch,
            bodyId,
            material,
            fill: material === 'obstacle' ? false : (patch.fill ?? object.fill),
            E: patch.E ?? (patch.material ? defaults.E : object.E),
            nu: patch.nu ?? (patch.material ? defaults.nu : object.nu),
            density: patch.density ?? (patch.material ? defaults.density : object.density),
            mode: patch.mode ?? (patch.material ? defaults.mode : object.mode),
            constraintMode: material === 'obstacle' ? (patch.constraintMode ?? object.constraintMode ?? 'grid-velocity-cuboid') : undefined,
            cuboidPaddingGridNodes: material === 'obstacle' ? (patch.cuboidPaddingGridNodes ?? object.cuboidPaddingGridNodes ?? 2) : undefined,
            drive: nextDrive
        };
        objects.set(objectId, nextObject);
        if (patch.drive) {
            for (const [otherId, other] of objects) {
                if (otherId !== objectId && (other.bodyId ?? other.objectId) === bodyId) {
                    objects.set(otherId, {
                        ...other,
                        drive: JSON.parse(JSON.stringify(nextObject.drive))
                    });
                }
            }
        }
        emit();
        invalidateLocalProxyCache(120);
    });

    events.on('physics.deleteObject', (objectId: number) => {
        if (!objects.delete(Number(objectId))) {
            return;
        }
        emit();
        invalidateLocalProxyCache(120);
    });

    events.on('physics.selectObject', selectObjectInScene);

    events.on('physics.clearObjects', () => {
        objects.clear();
        nextObjectId = 1;
        nextBodyId = 1;
        emit();
        invalidateLocalProxyCache(120);
    });

    events.function('physics.submitSimulation', async (url = '/api/simulate') => {
        const payload = buildPayload();
        events.fire('physics.status', '正在提交...');
        const response = await fetch(url.startsWith('/api/') ? apiUrl(url) : url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        if (!response.ok) {
            const errorBody = await response.json().catch((): null => null);
            const detail = errorBody?.detail ? String(errorBody.detail) : response.statusText;
            throw new Error(`提交仿真失败：${response.status} ${detail}`);
        }
        const result = await response.json().catch(() => ({}));
        if (result.runId) {
            return pollRun(result.runId);
        }
        await loadMotionFromRun(result);
        return result;
    });

    events.function('physics.submitDragPreview', async (preview: DragPreviewRequest) => {
        try {
            if (await tryLocalProxyDragPreview(preview)) {
                return {
                    status: 'completed',
                    local: true,
                    progress: 'local-proxy-preview-ready'
                };
            }
        } catch (error) {
            const message = String((error as any)?.message ?? error);
            events.fire('physics.status', `local proxy preview failed, falling back to server: ${message}`);
        }

        const payload = buildPayload();
        const hitIndex = Number(preview?.hitIndex);
        let objectId = Number(preview?.objectId);
        let bodyId = Number(preview?.bodyId);
        if (Number.isFinite(hitIndex) && hitIndex >= 0) {
            for (const object of objects.values()) {
                if (object.indices.includes(hitIndex)) {
                    objectId = object.objectId;
                    bodyId = object.bodyId ?? object.objectId;
                    break;
                }
            }
        }
        payload.simulation.preview = {
            ...(payload.simulation.preview ?? {}),
            dragHitIndex: Number.isFinite(hitIndex) ? hitIndex : undefined,
            dragVelocity: Array.isArray(preview?.velocity) ? preview.velocity.slice(0, 3) : [0, 0, 0],
            dragObjectId: Number.isFinite(objectId) && objectId > 0 ? objectId : undefined,
            dragBodyId: Number.isFinite(bodyId) && bodyId > 0 ? bodyId : undefined,
            adaptiveGrid: true,
            targetVoxelGroups: 192,
            proxyBlendCount: 4,
            proxyBlendPower: 2,
            velocitySpreadIterations: 10,
            velocitySpreadStrength: 0.65,
            shapeCoupling: 0.2
        };
        events.fire('physics.status', '正在提交拖拽快速预览...');
        const response = await fetch(apiUrl('/api/preview/voxel-rigid'), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        if (!response.ok) {
            const errorBody = await response.json().catch((): null => null);
            const detail = errorBody?.detail ? String(errorBody.detail) : response.statusText;
            throw new Error(`拖拽预览提交失败：${response.status} ${detail}`);
        }
        const result = await response.json().catch(() => ({}));
        if (result.status === 'completed' && result.result) {
            events.fire('physics.status', `拖拽预览已生成：${result.runId ?? ''}`);
            await loadMotionFromRun(result, { includeProxy: true });
            return result;
        }
        if (result.runId) {
            events.fire('physics.status', `拖拽预览任务已提交：${result.runId}`);
            return pollRun(result.runId, { includeProxy: true });
        }
        await loadMotionFromRun(result, { includeProxy: true });
        return result;
    });

    events.function('physics.cancelSimulation', async () => {
        if (!currentRunId) {
            return null;
        }
        const runId = currentRunId;
        const response = await fetch(apiUrl(`/api/runs/${encodeURIComponent(runId)}/cancel`), {
            method: 'POST'
        });
        if (!response.ok) {
            throw new Error(`cancel failed: ${response.status} ${response.statusText}`);
        }
        currentRunId = '';
        events.fire('physics.status', 'simulation cancelled');
        return response.json();
    });

    events.function('physics.downloadSolverTrace', async () => {
        const runId = currentRunId || lastRunId;
        if (!runId) {
            throw new Error('no completed simulation run is available for trace export');
        }
        const response = await fetch(apiUrl(`/api/runs/${encodeURIComponent(runId)}/solver_trace.json`));
        if (!response.ok) {
            const detail = await response.text().catch(() => response.statusText);
            throw new Error(`trace export failed: ${response.status} ${detail}`);
        }
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `solver_trace_${runId}.json`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
        events.fire('physics.status', `solver trace exported: ${runId}`);
        return { runId };
    });

    events.on('splat.stateChanged', refreshSelectedCount);
    events.on('selection.changed', refreshSelectedCount);
    void events.invoke('physics.listModels').catch((): null => null);
};

export { registerPhysicsSessionEvents, MaterialPreset, PhysicsPayload };
