import { BooleanInput, Button, Container, Label, NumericInput, SelectInput, TextAreaInput, TextInput } from '@playcanvas/pcui';
import { Vec3 } from 'playcanvas';

import { Events } from '../events';
import { MaterialPreset, PhysicsPayload } from '../physics-session';
import { Tooltips } from './tooltips';

const materialOptions: { v: MaterialPreset, t: string }[] = [
    { v: 'jelly', t: '果冻' },
    { v: 'metal', t: '金属' },
    { v: 'sand', t: '沙' },
    { v: 'foam', t: '泡沫' },
    { v: 'snow', t: '雪' },
    { v: 'plasticine', t: '橡皮泥' },
    { v: 'rigid', t: '近刚体' },
    { v: 'obstacle', t: '固定锚点' }
];

materialOptions.find(option => option.v === 'obstacle')!.t = 'Grid velocity cuboid';

const materialDensityDefaults: Record<MaterialPreset, number> = {
    jelly: 200,
    metal: 2700,
    sand: 1500,
    foam: 80,
    snow: 400,
    plasticine: 1300,
    rigid: 1000,
    obstacle: 200
};

const materialParamDefaults: Record<MaterialPreset, {
    E: number;
    nu: number;
    density: number;
}> = {
    jelly: { E: 1e5, nu: 0.3, density: 200 },
    metal: { E: 2e6, nu: 0.3, density: 2700 },
    sand: { E: 1e5, nu: 0.2, density: 1500 },
    foam: { E: 1e4, nu: 0.1, density: 80 },
    snow: { E: 1.4e5, nu: 0.2, density: 400 },
    plasticine: { E: 4e4, nu: 0.35, density: 1300 },
    rigid: { E: 1e7, nu: 0.25, density: 1000 },
    obstacle: { E: 2e6, nu: 0.4, density: 200 }
};

const roundValue = (value: number) => Number.isFinite(value) ? Number(value.toPrecision(6)) : value;

const formatPayload = (payload: PhysicsPayload) => JSON.stringify({
    solver: payload.solver,
    modelId: payload.modelId,
    officialConfig: payload.officialConfig,
    preprocessing: payload.preprocessing,
    simulation: payload.simulation,
    boundaryConstraints: payload.boundaryConstraints,
    objects: payload.objects.map(object => ({
        objectId: object.objectId,
        bodyId: object.bodyId,
        name: object.name,
        material: object.material,
        mode: object.mode,
        constraintMode: object.constraintMode,
        cuboidPaddingGridNodes: object.cuboidPaddingGridNodes,
        fill: object.fill,
        count: object.count,
        E: object.E,
        nu: object.nu,
        density: object.density,
        drive: object.drive,
        aabbWorld: object.aabbWorld,
        boundaryDebug: object.boundaryDebug
    })),
    selectedCount: payload.selectedCount
}, null, 2);

const createRow = (label: string, child: any) => {
    const row = new Container({ class: 'physics-row' });
    row.append(new Label({ class: 'physics-row-label', text: label }));
    row.append(child);
    return row;
};

const createParamRow = (label: string, child: any, help: string) => {
    const group = new Container({ class: 'physics-param' });
    group.append(createRow(label, child));
    group.append(new Label({ class: 'physics-param-help', text: help }));
    return group;
};

const createSection = (label: string) => {
    const section = new Container({ class: 'physics-section' });
    section.append(new Container({ class: 'physics-line' }));
    section.append(new Label({ class: 'physics-section-label', text: label }));
    return section;
};

const createFoldout = (label: string, open = true) => {
    const root = new Container({ class: 'physics-foldout' });
    const header = new Button({
        class: 'physics-foldout-header',
        text: `${open ? '[-]' : '[+]'} ${label}`
    });
    const body = new Container({
        class: 'physics-foldout-body',
        hidden: !open
    });
    header.on('click', () => {
        body.hidden = !body.hidden;
        header.text = `${body.hidden ? '[+]' : '[-]'} ${label}`;
    });
    root.append(header);
    root.append(body);
    return { root, body };
};

const createTabPages = (tabs: { id: string; label: string }[], activeId: string) => {
    const root = new Container({ class: 'physics-tabs' });
    const content = new Container({ class: 'physics-tab-content' });
    const pages = new Map<string, Container>();
    const buttons = new Map<string, Button>();

    const show = (id: string) => {
        pages.forEach((page, pageId) => {
            page.hidden = pageId !== id;
        });
        buttons.forEach((button, buttonId) => {
            if (buttonId === id) {
                button.class.add('active');
            } else {
                button.class.remove('active');
            }
        });
    };

    tabs.forEach((tab) => {
        const button = new Button({
            class: ['physics-button', 'physics-tab-button', tab.id === activeId ? 'active' : 'subtle'],
            text: tab.label
        });
        const page = new Container({
            class: 'physics-tab-page',
            hidden: tab.id !== activeId
        });
        button.on('click', () => show(tab.id));
        buttons.set(tab.id, button);
        pages.set(tab.id, page);
        root.append(button);
        content.append(page);
    });

    return { root, content, pages, show };
};

const numeric = (value: number, precision: number, min: number, max: number) => new NumericInput({
    class: 'physics-number',
    value,
    precision,
    min,
    max
});

class PhysicsPanel extends Container {
    constructor(events: Events, tooltips: Tooltips, args = {}) {
        args = {
            ...args,
            id: 'physics-panel',
            class: 'panel',
            hidden: true
        };

        super(args);

        ['pointerdown', 'pointerup', 'pointermove', 'wheel', 'dblclick'].forEach((eventName) => {
            this.dom.addEventListener(eventName, (event: Event) => event.stopPropagation());
        });

        const header = new Container({ class: 'panel-header' });
        header.append(new Label({ text: '\uE136', class: 'panel-header-icon' }));
        header.append(new Label({ text: '物理仿真', class: 'panel-header-label' }));

        const selectedLabel = new Label({ class: 'physics-muted', text: '已选：0' });
        const objectLabel = new Label({ class: 'physics-muted', text: 'Part：0' });
        const objectInfo = new Label({ class: 'physics-muted', text: '当前：无' });
        const modelStatus = new Label({ class: 'physics-muted', text: '预览模型：未同步' });
        const status = new Label({ class: 'physics-status', text: '' });

        const apiBase = new TextInput({
            class: 'physics-input',
            placeholder: '留空=同源'
        });

        const refreshCloudModels = new Button({
            class: ['physics-button', 'subtle'],
            text: '刷新官方样例'
        });
        const modelList = new Container({
            class: 'physics-model-list'
        });

        const solverHint = new Label({
            class: 'physics-muted',
            text: '当前可运行：显式 MPM / 隐式 MPM / PBMPM'
        });
        const solverList = new Container({
            class: 'physics-solver-list'
        });
        const solverButtons: Button[] = [];
        const syncSolverButtons = (activeSolver: PhysicsPayload['solver']) => {
            solverButtons.forEach((item) => {
                if (String((item as any)._solverValue) === activeSolver) {
                    item.class.add('active');
                    item.class.remove('subtle');
                } else {
                    item.class.remove('active');
                    item.class.add('subtle');
                }
            });
        };
        const solverOptions: { value: PhysicsPayload['solver']; text: string; enabled: boolean; note?: string }[] = [
            { value: 'explicit-mpm', text: '显式 MPM', enabled: true },
            { value: 'implicit-mpm', text: '隐式 MPM', enabled: true },
            { value: 'pbmpm', text: 'PBMPM Local-Global', enabled: true }
        ];
        solverOptions.forEach(({ value, text, enabled, note }) => {
            const button = new Button({
                class: ['physics-button', 'physics-solver-button', value === 'explicit-mpm' ? 'active' : 'subtle'],
                text: `${text}${note ? `（${note}）` : ''}`,
                enabled: !!enabled
            });
            (button as any)._solverValue = value;
            button.on('click', () => {
                syncSolverButtons(value);
                events.fire('physics.setSolver', value);
            });
            solverButtons.push(button);
            solverList.append(button);
        });

        const scale = numeric(1, 3, 0.01, 2);
        const nGrid = numeric(50, 0, 16, 512);
        const opacity = numeric(0.02, 3, 0, 1);
        const simAreaLabel = new Label({ class: 'physics-muted', text: '仿真区：自动 / 选区AABB' });
        const voxelGroupInfo = new Label({ class: 'physics-muted', text: '体素分组：等待模型/Part' });
        const officialConfig = new BooleanInput({
            class: 'boolean',
            type: 'toggle',
            value: false
        });
        const officialConfigHint = new Label({ class: 'physics-muted', text: '开启后使用模型绑定的官方 JSON；场景和对象按官方配置锁定，求解与时间仍可手动调整' });

        const gravityEnabled = new BooleanInput({
            class: 'boolean',
            type: 'toggle',
            value: true
        });
        const gravityX = numeric(0, 3, -100, 100);
        const gravityY = numeric(0, 3, -100, 100);
        const gravityZ = numeric(-9.8, 3, -100, 100);
        const groundEnabled = new BooleanInput({
            class: 'boolean',
            type: 'toggle',
            value: true
        });
        const groundHeight = numeric(0.5, 3, 0, 2);
        const boundingBoxEnabled = new BooleanInput({
            class: 'boolean',
            type: 'toggle',
            value: true
        });

        const objectName = new TextInput({
            class: 'physics-input',
            placeholder: 'Part 名称'
        });
        const createBodyId = numeric(0, 0, 0, 100000);
        const createBodyHint = new Label({ class: 'physics-muted', text: 'Body ID=0 表示自动：有当前 Part 时沿用其 Body，否则新建 Body' });
        const objectBodyId = numeric(1, 0, 1, 100000);
        const objectBodyHint = new Label({ class: 'physics-muted', text: '同一 Body 的多个 Part 共享仿真尺度、填充范围和整体驱动' });
        const createMaterial = new SelectInput({
            class: 'physics-input',
            defaultValue: 'jelly',
            options: materialOptions
        });
        const createDensity = numeric(materialDensityDefaults.jelly, 2, 0, 100000);
        const createFill = new BooleanInput({
            class: 'boolean',
            type: 'toggle',
            value: false
        });

        const objectSelect = new SelectInput({
            class: 'physics-input',
            defaultValue: '',
            options: [{ v: '', t: '未选择 Part' }]
        });
        const objectMaterial = new SelectInput({
            class: 'physics-input',
            defaultValue: 'jelly',
            options: materialOptions
        });
        const objectFill = new BooleanInput({
            class: 'boolean',
            type: 'toggle',
            value: false
        });
        const objectDensity = numeric(200, 2, 0, 100000);
        const objectE = numeric(1e5, 1, 0, 1e10);
        const objectNu = numeric(0.3, 3, 0, 0.49);

        const linearEnabled = new BooleanInput({
            class: 'boolean',
            type: 'toggle',
            value: false
        });
        const linearFx = numeric(0, 4, -1000, 1000);
        const linearFy = numeric(0, 4, -1000, 1000);
        const linearFz = numeric(0, 4, -1000, 1000);
        const linearNumDt = numeric(1, 0, 1, 10000);
        const linearStart = numeric(0, 4, 0, 10000);

        const spinEnabled = new BooleanInput({
            class: 'boolean',
            type: 'toggle',
            value: false
        });
        const spinAxisX = numeric(0, 3, -1, 1);
        const spinAxisY = numeric(0, 3, -1, 1);
        const spinAxisZ = numeric(1, 3, -1, 1);
        const spinAngular = numeric(0, 4, -100, 100);
        const spinTranslation = numeric(0, 4, -100, 100);
        const spinStart = numeric(0, 4, 0, 10000);
        const spinEnd = numeric(0.2, 4, 0, 10000);

        const dragGain = numeric(0.002, 5, 0, 1);
        const captureLinear = new Button({
            class: 'physics-button',
            text: '手掌抓取预览'
        });

        const useAabb = new Button({
            class: 'physics-button',
            text: '用当前选择设 AABB'
        });
        const useBodyAabb = new Button({
            class: 'physics-button',
            text: '用当前 Body 设 AABB'
        });
        const addObject = new Button({
            class: 'physics-button',
            text: '创建 Part'
        });
        const deleteObject = new Button({
            class: ['physics-button', 'subtle'],
            text: '删除当前 Part'
        });
        const clearObjects = new Button({
            class: ['physics-button', 'subtle'],
            text: '清空全部 Part'
        });

        const frameDt = numeric(0.02, 4, 1e-4, 1);
        const frameNum = numeric(30, 0, 1, 10000);
        const substepDt = numeric(1e-4, 6, 1e-6, 1);
        const damping = numeric(0.9999, 4, 0, 1);
        const implicitNewtonMaxIter = numeric(16, 0, 1, 128);
        const pbmpmNMin = numeric(3, 0, 1, 256);
        const pbmpmNMax = numeric(25, 0, 1, 512);
        const pbmpmPlasticMode = numeric(0, 0, 0, 1);
        const pbmpmYieldMin = numeric(0.55, 4, 0.01, 10);
        const pbmpmYieldMax = numeric(1.85, 4, 0.01, 10);
        const timeWarning = new Label({ class: 'physics-warning', text: '' });

        const copyPayload = new Button({
            class: 'physics-button',
            text: '复制JSON'
        });
        const submit = new Button({
            class: ['physics-button', 'primary'],
            text: '开始仿真'
        });
        const cancel = new Button({
            class: ['physics-button', 'subtle'],
            text: '停止任务'
        });
        const exportTrace = new Button({
            class: ['physics-button', 'subtle'],
            text: '\u5bfc\u51fa Trace JSON'
        });
        const payloadView = new TextAreaInput({
            class: 'physics-json',
            readOnly: true
        });

        let currentPayload: PhysicsPayload | null = null;
        let selectedObjectId = 0;
        let syncingObjectUi = false;
        let syncingOfficialUi = false;
        let submitting = false;
        let cloudModelRecords: any[] = [];

        const currentObject = () => currentPayload?.objects.find(object => object.objectId === selectedObjectId) ?? null;

        const applyMaterialDefaultsToPartControls = (material: MaterialPreset) => {
            const defaults = materialParamDefaults[material] ?? materialParamDefaults.jelly;
            objectDensity.value = defaults.density;
            objectE.value = defaults.E;
            objectNu.value = defaults.nu;
            if (material === 'obstacle') {
                objectFill.value = false;
            }
        };

        const updatePreprocess = () => {
            if (syncingOfficialUi || officialConfig.value) {
                return;
            }
            events.fire('physics.setPreprocess', {
                scale: scale.value,
                n_grid: nGrid.value,
                opacity_threshold: opacity.value
            });
        };

        const updateSimulation = () => {
            if (syncingOfficialUi) {
                return;
            }
            const patch = {
                frame_dt: frameDt.value,
                frame_num: frameNum.value,
                substep_dt: substepDt.value,
                damping: damping.value,
                newtonMaxIter: implicitNewtonMaxIter.value,
                pbmpm: {
                    n_min: pbmpmNMin.value,
                    n_max: pbmpmNMax.value,
                    plastic_mode: pbmpmPlasticMode.value,
                    yield_min: pbmpmYieldMin.value,
                    yield_max: pbmpmYieldMax.value
                }
            } as Partial<PhysicsPayload['simulation']>;
            Object.assign(patch, {
                gravityEnabled: gravityEnabled.value,
                gravity: [gravityX.value, gravityY.value, gravityZ.value]
            });
            if (!officialConfig.value) {
                Object.assign(patch, {
                    groundEnabled: groundEnabled.value,
                    groundHeight: groundHeight.value,
                    boundingBoxEnabled: boundingBoxEnabled.value
                });
            }
            events.fire('physics.setSimulation', patch);
        };

        const updateTimeWarning = () => {
            const frame = Number(frameDt.value);
            const substep = Number(substepDt.value);
            if (!Number.isFinite(frame) || !Number.isFinite(substep) || frame <= 0 || substep <= 0) {
                timeWarning.text = '警告：时间参数需要为正数';
                return;
            }
            if (substep >= frame) {
                timeWarning.text = '警告：子步长不小于帧间隔时，每帧只推进 1 个大步，显式求解极易爆炸';
                return;
            }
            if (substep >= 1e-2) {
                timeWarning.text = '警告：子步长远大于官方设置，显式 MPM 很可能不稳定';
                return;
            }
            if (substep >= 1e-3) {
                timeWarning.text = '提示：子步长较大，速度会更快但误差会明显增加';
                return;
            }
            timeWarning.text = '';
        };

        const updateObject = () => {
            if (syncingObjectUi || officialConfig.value || !selectedObjectId) {
                return;
            }
            events.fire('physics.updateObject', selectedObjectId, {
                bodyId: objectBodyId.value,
                material: objectMaterial.value as MaterialPreset,
                fill: objectFill.value,
                density: objectDensity.value,
                E: objectE.value,
                nu: objectNu.value,
                drive: {
                    linearEnabled: linearEnabled.value,
                    linearForce: [linearFx.value, linearFy.value, linearFz.value],
                    linearNumDt: linearNumDt.value,
                    linearStart: linearStart.value,
                    spinEnabled: spinEnabled.value,
                    spinAxis: [spinAxisX.value, spinAxisY.value, spinAxisZ.value],
                    spinAngular: spinAngular.value,
                    spinTranslation: spinTranslation.value,
                    spinStart: spinStart.value,
                    spinEnd: spinEnd.value
                }
            });
        };

        const manualControls = [
            scale,
            nGrid,
            opacity,
            gravityEnabled,
            gravityX,
            gravityY,
            gravityZ,
            groundEnabled,
            groundHeight,
            boundingBoxEnabled,
            useAabb,
            useBodyAabb,
            objectName,
            createBodyId,
            createMaterial,
            createDensity,
            createFill,
            addObject,
            objectSelect,
            objectBodyId,
            objectMaterial,
            objectFill,
            objectDensity,
            objectE,
            objectNu,
            deleteObject,
            clearObjects,
            linearEnabled,
            linearFx,
            linearFy,
            linearFz,
            linearNumDt,
            linearStart,
            dragGain,
            captureLinear,
            spinEnabled,
            spinAxisX,
            spinAxisY,
            spinAxisZ,
            spinAngular,
            spinTranslation,
            spinStart,
            spinEnd
        ];

        const applyOfficialUiState = (payload: PhysicsPayload) => {
            const available = !!payload.officialConfig?.available;
            const enabled = !!payload.officialConfig?.enabled && available;
            officialConfig.enabled = available;
            if (officialConfig.value !== enabled) {
                officialConfig.value = enabled;
            }
            manualControls.forEach(control => {
                control.enabled = !enabled;
            });
            officialConfigHint.text = available ?
                `官方配置：${payload.officialConfig?.name ?? '已绑定'}${enabled ? '（已启用：场景和对象按官方 JSON 锁定，求解与时间参数仍可自定义）' : ''}` :
                '当前模型没有绑定官方配置；上传匹配官方样例的 PLY 或选择云端官方模型后可用';

            syncingOfficialUi = true;
            const currentSimulation: Partial<PhysicsPayload['simulation']> = payload.simulation ?? {};
            if (currentSimulation.frame_dt !== undefined) {
                frameDt.value = currentSimulation.frame_dt;
            }
            if (currentSimulation.frame_num !== undefined) {
                frameNum.value = currentSimulation.frame_num;
            }
            if (currentSimulation.substep_dt !== undefined) {
                substepDt.value = currentSimulation.substep_dt;
            }
            if (currentSimulation.damping !== undefined) {
                damping.value = currentSimulation.damping;
            }
            if (currentSimulation.newtonMaxIter !== undefined) {
                implicitNewtonMaxIter.value = currentSimulation.newtonMaxIter;
            }
            if (currentSimulation.pbmpm) {
                if (currentSimulation.pbmpm.n_min !== undefined) pbmpmNMin.value = currentSimulation.pbmpm.n_min;
                if (currentSimulation.pbmpm.n_max !== undefined) pbmpmNMax.value = currentSimulation.pbmpm.n_max;
                if (currentSimulation.pbmpm.plastic_mode !== undefined) pbmpmPlasticMode.value = currentSimulation.pbmpm.plastic_mode;
                if (currentSimulation.pbmpm.yield_min !== undefined) pbmpmYieldMin.value = currentSimulation.pbmpm.yield_min;
                if (currentSimulation.pbmpm.yield_max !== undefined) pbmpmYieldMax.value = currentSimulation.pbmpm.yield_max;
            }
            updateTimeWarning();
            if (Array.isArray(currentSimulation.gravity)) {
                const g = currentSimulation.gravity;
                gravityEnabled.value = !!currentSimulation.gravityEnabled;
                gravityX.value = g[0] ?? 0;
                gravityY.value = g[1] ?? 0;
                gravityZ.value = g[2] ?? 0;
            }
            const values = payload.officialConfig?.values;
            if (!values) {
                syncingOfficialUi = false;
                return;
            }
            const preprocessing = values.preprocessing ?? {};
            if (preprocessing.scale !== undefined) scale.value = preprocessing.scale;
            if (preprocessing.n_grid !== undefined) nGrid.value = preprocessing.n_grid;
            if (preprocessing.opacity_threshold !== undefined) opacity.value = preprocessing.opacity_threshold;
            const officialSimulation = values.simulation ?? {};
            if (enabled && Array.isArray(officialSimulation.gravity)) {
                const g = officialSimulation.gravity;
                gravityEnabled.value = !!officialSimulation.gravityEnabled;
                gravityX.value = g[0] ?? 0;
                gravityY.value = g[1] ?? 0;
                gravityZ.value = g[2] ?? 0;
            }
            syncingOfficialUi = false;
        };

        const syncObjectControls = () => {
            const object = currentObject();
            syncingObjectUi = true;
            if (!object) {
                objectInfo.text = '当前：无';
            } else {
                objectInfo.text = `${object.name} / Body ${object.bodyId ?? object.objectId} / ${object.count} 个 Gaussian`;
                objectBodyId.value = object.bodyId ?? object.objectId;
                objectMaterial.value = object.material;
                objectFill.value = object.fill;
                objectDensity.value = object.density;
                objectE.value = object.E;
                objectNu.value = object.nu;
                linearEnabled.value = object.drive.linearEnabled;
                linearFx.value = object.drive.linearForce[0] ?? 0;
                linearFy.value = object.drive.linearForce[1] ?? 0;
                linearFz.value = object.drive.linearForce[2] ?? 0;
                linearNumDt.value = object.drive.linearNumDt;
                linearStart.value = object.drive.linearStart;
                spinEnabled.value = object.drive.spinEnabled;
                spinAxisX.value = object.drive.spinAxis[0] ?? 0;
                spinAxisY.value = object.drive.spinAxis[1] ?? 0;
                spinAxisZ.value = object.drive.spinAxis[2] ?? 1;
                spinAngular.value = object.drive.spinAngular;
                spinTranslation.value = object.drive.spinTranslation;
                spinStart.value = object.drive.spinStart;
                spinEnd.value = object.drive.spinEnd;
            }
            syncingObjectUi = false;
        };

        [scale, nGrid, opacity].forEach(input => input.on('change', updatePreprocess));
        officialConfig.on('change', () => {
            events.fire('physics.setOfficialConfig', officialConfig.value);
        });
        [
            gravityEnabled,
            gravityX,
            gravityY,
            gravityZ,
            groundEnabled,
            groundHeight,
            boundingBoxEnabled,
            frameDt,
            frameNum,
            substepDt,
            damping,
            implicitNewtonMaxIter,
            pbmpmNMin,
            pbmpmNMax,
            pbmpmPlasticMode,
            pbmpmYieldMin,
            pbmpmYieldMax
        ].forEach(input => input.on('change', () => {
            updateTimeWarning();
            updateSimulation();
        }));
        [
            objectBodyId,
            objectFill,
            objectDensity,
            objectE,
            objectNu,
            linearEnabled,
            linearFx,
            linearFy,
            linearFz,
            linearNumDt,
            linearStart,
            spinEnabled,
            spinAxisX,
            spinAxisY,
            spinAxisZ,
            spinAngular,
            spinTranslation,
            spinStart,
            spinEnd
        ].forEach(input => input.on('change', updateObject));

        objectMaterial.on('change', () => {
            if (syncingObjectUi || officialConfig.value || !selectedObjectId) {
                return;
            }
            syncingObjectUi = true;
            applyMaterialDefaultsToPartControls(objectMaterial.value as MaterialPreset);
            syncingObjectUi = false;
            updateObject();
        });

        const refreshModels = () => {
            status.text = '正在刷新官方样例...';
            void events.invoke('physics.listModels').then(() => {
                status.text = cloudModelRecords.length ? `已找到 ${cloudModelRecords.length} 个官方样例` : '云端没有可用官方样例';
            }).catch((error: any) => {
                status.text = String(error?.message ?? error);
            });
        };

        const loadCloudModelById = async (id: string) => {
            if (!id) {
                status.text = '请选择官方样例';
                return;
            }
            try {
                status.text = `正在加载 ${id}...`;
                await events.invoke('physics.loadModelById', id);
            } catch (error) {
                status.text = String(error?.message ?? error);
            }
        };

        const renderModelButtons = (records: any[]) => {
            modelList.clear();
            const sorted = (records ?? [])
                .filter(record => record.officialConfigAvailable || String(record.kind ?? '').includes('official'))
                .slice()
                .sort((a, b) => String(a.modelId).localeCompare(String(b.modelId)));
            if (!sorted.length) {
                modelList.append(new Label({ class: 'physics-muted', text: '没有从后端读取到官方样例' }));
                return;
            }
            sorted.forEach((record) => {
                const id = String(record.modelId ?? '');
                const title = id.replace('-sample-7000', ' 7000').replace('-sample', '').replace(/-/g, ' ');
                const text = `${title}${record.officialConfigName ? ` / ${record.officialConfigName}` : ''}`;
                const button = new Button({
                    class: ['physics-button', 'physics-model-button'],
                    text
                });
                button.on('click', () => {
                    void loadCloudModelById(id);
                });
                modelList.append(button);
            });
        };

        apiBase.on('change', () => {
            events.fire('physics.setApiBase', apiBase.value);
            refreshModels();
        });

        refreshCloudModels.on('click', () => {
            refreshModels();
        });

        objectSelect.on('change', () => {
            selectedObjectId = Number(objectSelect.value) || 0;
            syncObjectControls();
            if (selectedObjectId) {
                events.fire('physics.selectObject', selectedObjectId);
            }
        });

        createMaterial.on('change', () => {
            const material = createMaterial.value as MaterialPreset;
            createDensity.value = materialDensityDefaults[material] ?? materialDensityDefaults.jelly;
            if (material === 'obstacle') {
                createFill.value = false;
            }
        });

        useAabb.on('click', () => {
            events.fire('physics.useSelectionAabb');
            const payload = events.invoke('physics.payload') as PhysicsPayload;
            if (payload?.preprocessing.sim_area) {
                payload.preprocessing.sim_area = payload.preprocessing.sim_area.map(roundValue);
                events.fire('physics.setPreprocess', { sim_area: payload.preprocessing.sim_area });
            }
        });

        useBodyAabb.on('click', () => {
            if (!selectedObjectId) {
                status.text = '请先选择一个 Part';
                return;
            }
            events.fire('physics.useCurrentBodyAabb', selectedObjectId);
        });

        addObject.on('click', () => {
            const inheritedBodyId = selectedObjectId ? (currentObject()?.bodyId ?? selectedObjectId) : 0;
            const requestedBodyId = Number(createBodyId.value) > 0 ? createBodyId.value : inheritedBodyId;
            events.fire('physics.createObjectFromSelection', {
                name: objectName.value,
                material: createMaterial.value as MaterialPreset,
                fill: createFill.value,
                bodyId: requestedBodyId,
                density: createDensity.value
            });
            objectName.value = '';
            createBodyId.value = 0;
        });

        deleteObject.on('click', () => {
            if (!selectedObjectId) {
                status.text = '请先选择一个 Part';
                return;
            }
            const deletedId = selectedObjectId;
            selectedObjectId = 0;
            events.fire('physics.deleteObject', deletedId);
            status.text = `已删除 Part ${deletedId}`;
        });

        clearObjects.on('click', () => {
            selectedObjectId = 0;
            events.fire('physics.clearObjects');
        });

        copyPayload.on('click', async () => {
            await navigator.clipboard.writeText(JSON.stringify(events.invoke('physics.payload'), null, 2));
            status.text = '已复制';
            setTimeout(() => {
                status.text = '';
            }, 1200);
        });

        submit.on('click', async () => {
            if (submitting) {
                return;
            }
            submitting = true;
            submit.enabled = false;
            status.text = '正在提交...';
            try {
                await events.invoke('physics.submitSimulation');
                status.text = '动画已加载';
            } catch (error) {
                const message = String(error?.message ?? error);
                status.text = message.length > 220 ? `${message.slice(0, 220)}...` : message;
            } finally {
                submitting = false;
                submit.enabled = true;
            }
        });

        cancel.on('click', async () => {
            try {
                await events.invoke('physics.cancelSimulation');
                status.text = '仿真已中断';
            } catch (error) {
                const message = String(error?.message ?? error);
                status.text = message.length > 220 ? `${message.slice(0, 220)}...` : message;
            }
        });

        exportTrace.on('click', async () => {
            try {
                await events.invoke('physics.downloadSolverTrace');
                status.text = 'Trace JSON \u5df2\u5bfc\u51fa';
            } catch (error) {
                const message = String(error?.message ?? error);
                status.text = message.length > 220 ? `${message.slice(0, 220)}...` : message;
            }
        });

        type DragCapture = {
            x: number;
            y: number;
            lastX: number;
            lastY: number;
            liveX: number;
            liveY: number;
            t: number;
            lastT: number;
            pointerId: number;
            hitIndex: number | null;
            pickDone: boolean;
            pickPromise: Promise<void> | null;
            liveStarted: boolean;
        };

        let waitingForDrag = false;
        let dragAutoPreview = false;
        let dragStart: DragCapture | null = null;
        let dragPreviousTool: string | null = null;
        let dragPickToken = 0;
        let dragLiveFrame: number | null = null;
        let dragLiveLastTick = 0;
        const worldUp = new Vec3(0, 0, 1);
        const forward = new Vec3();
        const right = new Vec3();
        const up = new Vec3();
        const pointerForce = new Vec3();
        const dragOverlay = document.createElement('div');
        dragOverlay.className = 'physics-drag-overlay';
        dragOverlay.innerHTML = '<div class="physics-drag-dot"></div><div class="physics-drag-tip"></div>';
        document.body.appendChild(dragOverlay);
        const dragDot = dragOverlay.querySelector('.physics-drag-dot') as HTMLElement;
        const dragTip = dragOverlay.querySelector('.physics-drag-tip') as HTMLElement;
        const canvas = () => document.getElementById('canvas') as HTMLCanvasElement | null;
        const isCanvasPointer = (event: PointerEvent) => {
            const targetCanvas = canvas();
            return !!targetCanvas && (event.target === targetCanvas || event.composedPath().includes(targetCanvas));
        };
        const consumeDragEvent = (event: PointerEvent) => {
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
        };
        const setDragOverlay = (x0: number, y0: number, x1: number, y1: number, text: string, state: 'pending' | 'hit' | 'miss' | 'dragging') => {
            dragOverlay.classList.add('visible');
            dragOverlay.classList.toggle('pending', state === 'pending');
            dragOverlay.classList.toggle('hit', state === 'hit' || state === 'dragging');
            dragOverlay.classList.toggle('miss', state === 'miss');
            dragDot.style.left = `${x1}px`;
            dragDot.style.top = `${y1}px`;
            dragTip.style.left = `${x1}px`;
            dragTip.style.top = `${y1}px`;
            dragTip.textContent = text;
        };
        const hideDragOverlay = () => {
            dragOverlay.classList.remove('visible', 'pending', 'hit', 'miss');
            dragTip.textContent = '';
        };
        const restoreDragTool = (hideOverlayNow = true) => {
            const previous = dragPreviousTool;
            dragPreviousTool = null;
            document.body.classList.remove('physics-drag-capturing');
            if (hideOverlayNow) {
                hideDragOverlay();
            }
            if (previous) {
                events.fire(`tool.${previous}`);
            } else {
                events.fire('tool.activated', events.invoke('tool.active'));
            }
        };
        const stopLiveTick = () => {
            if (dragLiveFrame !== null) {
                window.cancelAnimationFrame(dragLiveFrame);
                dragLiveFrame = null;
            }
            dragLiveLastTick = 0;
        };
        const invokeLiveProxyStep = async (capture: DragCapture, targetX: number, targetY: number, dt: number) => {
            const activeCanvas = canvas();
            const result = await events.invoke('physics.liveProxyDragMove', {
                hitIndex: capture.hitIndex,
                velocity: [0, 0, 0],
                screenStart: [capture.liveX, capture.liveY],
                screenEnd: [targetX, targetY],
                duration: Math.max(dt, 1 / 120),
                canvasSize: activeCanvas ? [activeCanvas.clientWidth, activeCanvas.clientHeight] : undefined,
                dragGain: dragGain.value,
                objectId: selectedObjectId || undefined
            }) as any;
            capture.liveX = targetX;
            capture.liveY = targetY;
            return result;
        };
        const scheduleLiveTick = () => {
            if (dragLiveFrame !== null) {
                return;
            }
            dragLiveFrame = window.requestAnimationFrame(async (time) => {
                dragLiveFrame = null;
                const capture = dragStart;
                if (!waitingForDrag || !capture?.liveStarted) {
                    dragLiveLastTick = 0;
                    return;
                }
                const previousTime = dragLiveLastTick || time;
                dragLiveLastTick = time;
                const dt = Math.max(1 / 120, Math.min((time - previousTime) / 1000, 1 / 24));
                await invokeLiveProxyStep(capture, capture.lastX, capture.lastY, dt).catch((): null => null);
                if (waitingForDrag && dragStart === capture && capture.liveStarted) {
                    scheduleLiveTick();
                }
            });
        };
        const settleLiveProxy = async (capture: DragCapture, maxFrames = 45) => {
            stopLiveTick();
            let settledFrames = 0;
            for (let frame = 0; frame < maxFrames; ++frame) {
                const result = await invokeLiveProxyStep(capture, capture.lastX, capture.lastY, 1 / 60).catch((): null => null) as any;
                const settled = !!result?.settled;
                settledFrames = settled ? settledFrames + 1 : 0;
                if (frame >= 8 && settledFrames >= 3) {
                    break;
                }
                await new Promise<void>(resolve => window.requestAnimationFrame(() => resolve()));
            }
        };

        const beginDragCapture = (autoPreview = false) => {
            waitingForDrag = true;
            dragAutoPreview = autoPreview;
            dragStart = null;
            stopLiveTick();
            dragPickToken++;
            dragPreviousTool = events.invoke('tool.active') as string | null;
            events.fire('tool.deactivate');
            document.body.classList.add('physics-drag-capturing');
            if (autoPreview) {
                events.fire('tool.activated', 'physicsDragImpulse');
            }
            status.text = '手掌抓取已激活：在画布上按住 Gaussian，proxy 位置和速度会跟随鼠标';
        };

        captureLinear.on('click', () => beginDragCapture(false));
        events.on('physics.dragImpulsePreview', () => beginDragCapture(true));

        window.addEventListener('pointerdown', (event: PointerEvent) => {
            if (!waitingForDrag || !isCanvasPointer(event) || dragStart) {
                return;
            }
            consumeDragEvent(event);
            if (event.button !== 0) {
                status.text = '手掌抓取只响应鼠标左键；按 Esc 或再次点击手掌按钮可重新开始';
                return;
            }
            const targetCanvas = canvas();
            const rect = targetCanvas?.getBoundingClientRect();
            const point = rect ? {
                x: (event.clientX - rect.left) / Math.max(rect.width, 1),
                y: (event.clientY - rect.top) / Math.max(rect.height, 1),
                radiusPixels: 14
            } : null;
            const token = ++dragPickToken;
            dragStart = {
                x: event.clientX,
                y: event.clientY,
                lastX: event.clientX,
                lastY: event.clientY,
                liveX: event.clientX,
                liveY: event.clientY,
                t: performance.now(),
                lastT: performance.now(),
                pointerId: event.pointerId,
                hitIndex: null,
                pickDone: false,
                pickPromise: null,
                liveStarted: false
            };
            targetCanvas?.setPointerCapture?.(event.pointerId);
            setDragOverlay(event.clientX, event.clientY, event.clientX + 1, event.clientY, '正在命中 Gaussian...', 'pending');
            status.text = '已进入手掌抓取，不会触发相机/选区/位移；正在命中 Gaussian...';
            const pickPromise = (async () => {
                try {
                    const pick = point ? await events.invoke('select.pickIndex', point) as any : null;
                    if (!dragStart || dragStart.pointerId !== event.pointerId || token !== dragPickToken) {
                        return;
                    }
                    const hitIndex = Number.isFinite(Number(pick?.index)) ? Number(pick.index) : null;
                    dragStart.hitIndex = hitIndex;
                    dragStart.pickDone = true;
                    const activeCanvas = canvas();
                    const liveStart = events.invoke('physics.liveProxyDragStart', {
                        hitIndex,
                        velocity: [0, 0, 0],
                        screenStart: [dragStart.x, dragStart.y],
                        screenEnd: [dragStart.x, dragStart.y],
                        duration: 1 / 60,
                        canvasSize: activeCanvas ? [activeCanvas.clientWidth, activeCanvas.clientHeight] : undefined,
                        dragGain: dragGain.value,
                        objectId: selectedObjectId || undefined
                    }) as any;
                    dragStart.liveStarted = !!liveStart;
                    if (dragStart.liveStarted) {
                        dragLiveLastTick = performance.now();
                        scheduleLiveTick();
                    }
                    const text = hitIndex === null ? '未命中：将作用到当前 Body / 全点云' : `命中 Gaussian #${hitIndex}`;
                    setDragOverlay(dragStart.x, dragStart.y, dragStart.lastX, dragStart.lastY, text, hitIndex === null ? 'miss' : 'hit');
                    status.text = `${text}；继续拖动会把对应 proxy 绑定到鼠标`;
                } catch (error) {
                    if (!dragStart || dragStart.pointerId !== event.pointerId || token !== dragPickToken) {
                        return;
                    }
                    dragStart.pickDone = true;
                    dragStart.hitIndex = null;
                    const message = String(error?.message ?? error);
                    setDragOverlay(dragStart.x, dragStart.y, dragStart.lastX, dragStart.lastY, '命中失败：将作用到当前 Body / 全点云', 'miss');
                    status.text = message.length > 160 ? `${message.slice(0, 160)}...` : message;
                }
            })();
            dragStart.pickPromise = pickPromise;
        }, true);

        window.addEventListener('pointermove', (event: PointerEvent) => {
            if (!waitingForDrag || !dragStart || event.pointerId !== dragStart.pointerId) {
                return;
            }
            consumeDragEvent(event);
            const previousX = dragStart.lastX;
            const previousY = dragStart.lastY;
            const previousT = dragStart.lastT;
            const now = performance.now();
            dragStart.lastX = event.clientX;
            dragStart.lastY = event.clientY;
            dragStart.lastT = now;
            const dt = Math.max(1 / 120, (now - previousT) / 1000);
            const dx = (event.clientX - previousX) / dt;
            const dy = (event.clientY - previousY) / dt;
            const hitText = dragStart.pickDone ?
                (dragStart.hitIndex === null ? '当前 Body / 全点云' : `Gaussian #${dragStart.hitIndex}`) :
                '命中检测中';
            setDragOverlay(dragStart.x, dragStart.y, event.clientX, event.clientY, `${hitText}  速度 ${roundValue(dx)}, ${roundValue(dy)}`, 'dragging');
            status.text = `手掌抓取中：屏幕速度 ${roundValue(dx)}, ${roundValue(dy)}`;
        }, true);

        window.addEventListener('pointerup', async (event: PointerEvent) => {
            if (!waitingForDrag || !dragStart || event.pointerId !== dragStart.pointerId) {
                return;
            }
            consumeDragEvent(event);
            const activeDrag = dragStart;
            if (!activeDrag.pickDone && activeDrag.pickPromise) {
                await Promise.race([
                    activeDrag.pickPromise,
                    new Promise(resolve => setTimeout(resolve, 200))
                ]);
            }
            const dt = Math.max(0.05, (performance.now() - activeDrag.t) / 1000);
            const dx = (event.clientX - activeDrag.x) / dt;
            const dy = (event.clientY - activeDrag.y) / dt;
            const dragHitIndex = activeDrag.hitIndex;
            const activeCanvas = canvas();
            const pose = events.invoke('camera.getPose') as any;
            const position = new Vec3(pose?.position?.x ?? 0, pose?.position?.y ?? -1, pose?.position?.z ?? 0);
            const target = new Vec3(pose?.target?.x ?? 0, pose?.target?.y ?? 0, pose?.target?.z ?? 0);

            forward.sub2(target, position).normalize();
            right.cross(forward, worldUp);
            if (right.lengthSq() < 1e-8) {
                right.set(1, 0, 0);
            } else {
                right.normalize();
            }
            up.cross(right, forward).normalize();
            pointerForce.copy(right).mulScalar(dx * dragGain.value);
            pointerForce.add(up.mulScalar(-dy * dragGain.value));

            const projected = events.invoke('physics.projectDragVelocity', {
                hitIndex: dragHitIndex,
                velocity: [pointerForce.x, pointerForce.y, pointerForce.z],
                screenStart: [activeDrag.x, activeDrag.y],
                screenEnd: [event.clientX, event.clientY],
                duration: dt,
                canvasSize: activeCanvas ? [activeCanvas.clientWidth, activeCanvas.clientHeight] : undefined,
                dragGain: dragGain.value,
                objectId: selectedObjectId || undefined
            }) as any;
            const projectedVelocity = Array.isArray(projected?.velocity) ? projected.velocity : [pointerForce.x, pointerForce.y, pointerForce.z];
            linearFx.value = roundValue(projectedVelocity[0]);
            linearFy.value = roundValue(projectedVelocity[1]);
            linearFz.value = roundValue(projectedVelocity[2]);
            if (selectedObjectId) {
                linearEnabled.value = true;
                updateObject();
            }

            waitingForDrag = false;
            stopLiveTick();
            activeCanvas?.releasePointerCapture?.(activeDrag.pointerId);
            dragStart = null;
            dragPickToken++;
            status.text = `手掌抓取结束：${linearFx.value}, ${linearFy.value}, ${linearFz.value}`;
            setDragOverlay(activeDrag.x, activeDrag.y, event.clientX, event.clientY, '手掌抓取结束', dragHitIndex === null ? 'miss' : 'hit');
            if (dragAutoPreview) {
                if (activeDrag.liveStarted) {
                    status.text = 'live proxy settling after grab...';
                    await settleLiveProxy(activeDrag);
                    void events.invoke('physics.liveProxyDragEnd');
                    dragAutoPreview = false;
                    status.text = 'live proxy preview finished; click Run Simulation for the full PhysGaussian solve';
                    hideDragOverlay();
                    restoreDragTool(false);
                    return;
                }
                dragAutoPreview = false;
                status.text = '未命中可抓取 Gaussian，未启动手掌抓取';
                hideDragOverlay();
                restoreDragTool(false);
                return;
            } else {
                if (activeDrag.liveStarted) {
                    status.text = 'live proxy settling after grab...';
                    await settleLiveProxy(activeDrag);
                    void events.invoke('physics.liveProxyDragEnd');
                }
                dragAutoPreview = false;
                restoreDragTool();
            }
        }, true);

        window.addEventListener('pointercancel', (event: PointerEvent) => {
            if (!waitingForDrag || !dragStart || event.pointerId !== dragStart.pointerId) {
                return;
            }
            consumeDragEvent(event);
            canvas()?.releasePointerCapture?.(dragStart.pointerId);
            waitingForDrag = false;
            dragAutoPreview = false;
            stopLiveTick();
            if (dragStart.liveStarted) {
                void events.invoke('physics.liveProxyDragEnd');
            }
            dragStart = null;
            dragPickToken++;
            restoreDragTool();
            status.text = '拖拽已取消';
        }, true);

        window.addEventListener('keydown', (event: KeyboardEvent) => {
            if (!waitingForDrag || event.key !== 'Escape') {
                return;
            }
            event.preventDefault();
            event.stopPropagation();
            dragPickToken++;
            waitingForDrag = false;
            dragAutoPreview = false;
            stopLiveTick();
            if (dragStart) {
                canvas()?.releasePointerCapture?.(dragStart.pointerId);
                if (dragStart.liveStarted) {
                    void events.invoke('physics.liveProxyDragEnd');
                }
            }
            dragStart = null;
            restoreDragTool();
            status.text = '物理拖拽已取消';
        }, true);

        events.on('physics.selectionCount', (count: number) => {
            selectedLabel.text = `已选：${count}`;
        });

        events.on('physics.status', (message: string) => {
            status.text = message;
        });

        events.on('physics.voxelGroups', (summary: any) => {
            const bodies = Array.isArray(summary?.bodies) ? summary.bodies : [];
            if (!bodies.length) {
                voxelGroupInfo.text = '体素分组：无可分组 Body';
                return;
            }
            const totalGroups = bodies.reduce((sum: number, body: any) => sum + Number(body.groupCount ?? 0), 0);
            const totalTracked = bodies.reduce((sum: number, body: any) => sum + Number(body.selectedCount ?? 0), 0);
            voxelGroupInfo.text = `体素分组：${bodies.length} 个 Body / ${totalGroups} 组 / ${totalTracked} 点`;
        });

        events.on('physics.models', (records: any[]) => {
            cloudModelRecords = (records ?? []).filter(record => record.officialConfigAvailable || String(record.kind ?? '').includes('official'));
            renderModelButtons(cloudModelRecords);
        });

        events.on('physics.payload', (payload: PhysicsPayload) => {
            currentPayload = payload;
            applyOfficialUiState(payload);
            syncSolverButtons(payload.solver);
            objectLabel.text = `Part：${payload.objects.length}`;
            simAreaLabel.text = payload.preprocessing.sim_area ?
                `仿真区：${payload.preprocessing.sim_area.map(roundValue).join(', ')}` :
                '仿真区：自动 / Body 并集';
            modelStatus.text = payload.modelId ? `预览模型：已同步 ${payload.modelId}` : '预览模型：未同步，请先用 SuperSplat 打开/拖入 PLY';

            syncingOfficialUi = true;
            implicitNewtonMaxIter.value = payload.simulation.newtonMaxIter ?? 16;
            damping.value = payload.simulation.damping ?? 0.9999;
            pbmpmNMin.value = payload.simulation.pbmpm?.n_min ?? 3;
            pbmpmNMax.value = payload.simulation.pbmpm?.n_max ?? 25;
            pbmpmPlasticMode.value = payload.simulation.pbmpm?.plastic_mode ?? 0;
            pbmpmYieldMin.value = payload.simulation.pbmpm?.yield_min ?? 0.55;
            pbmpmYieldMax.value = payload.simulation.pbmpm?.yield_max ?? 1.85;
            syncingOfficialUi = false;

            const objectOptions = payload.objects.length ?
                payload.objects.map(object => ({ v: object.objectId, t: `Part ${object.objectId}: ${object.name} / Body ${object.bodyId ?? object.objectId}` })) :
                [{ v: '', t: '未选择 Part' }];
            objectSelect.options = objectOptions;
            if (!payload.objects.some(object => object.objectId === selectedObjectId)) {
                selectedObjectId = payload.objects[payload.objects.length - 1]?.objectId ?? 0;
                objectSelect.value = selectedObjectId || '';
            }
            syncObjectControls();
            payloadView.value = formatPayload(payload);
        });

        const setVisible = (visible: boolean) => {
            if (visible === this.hidden) {
                this.hidden = !visible;
                events.fire('physicsPanel.visible', visible);
            }
        };

        events.function('physicsPanel.visible', () => {
            return !this.hidden;
        });

        events.on('physicsPanel.setVisible', (visible: boolean) => {
            setVisible(visible);
            if (visible) {
                void events.invoke('physics.listModels').catch((error: any) => {
                    status.text = String(error?.message ?? error);
                });
            }
        });

        events.on('physicsPanel.toggleVisible', () => {
            setVisible(this.hidden);
        });

        events.on('viewPanel.visible', (visible: boolean) => {
            if (visible) {
                setVisible(false);
            }
        });

        events.on('colorPanel.visible', (visible: boolean) => {
            if (visible) {
                setVisible(false);
            }
        });

        const modelFoldout = createFoldout('模型与官方预设', true);
        modelFoldout.body.append(createRow('接口', apiBase));
        modelFoldout.body.append(refreshCloudModels);
        modelFoldout.body.append(modelList);
        modelFoldout.body.append(modelStatus);
        modelFoldout.body.append(createRow('官方配置', officialConfig));
        modelFoldout.body.append(officialConfigHint);

        const solverFoldout = createFoldout('求解器', true);
        solverFoldout.body.append(solverHint);
        solverFoldout.body.append(solverList);
        const implicitInfo = new Label({
            class: 'physics-param-help',
            text: '隐式 MPM 使用 Newmark + Newton-GMRES + EW 默认设置；只需要在运行页调整帧间隔、帧数和子步长。'
        });
        const implicitFoldout = createFoldout('隐式 MPM', false);
        implicitFoldout.body.append(createParamRow('Newton 最大迭代', implicitNewtonMaxIter, '每个 implicit substep 内 Newton 外层最多迭代次数；默认 16，增大更稳但更慢。'));
        const pbmpmFoldout = createFoldout('PBMPM Local-Global', false);
        pbmpmFoldout.body.append(createParamRow('N min', pbmpmNMin, 'PBMPM 自动映射 local/global 内步迭代下限；默认 3。'));
        pbmpmFoldout.body.append(createParamRow('N max', pbmpmNMax, 'PBMPM 自动映射 local/global 内步迭代上限；默认 25。'));
        pbmpmFoldout.body.append(createParamRow('Plastic mode', pbmpmPlasticMode, '0 使用显式/隐式 MPM 同源 material return mapping；1 使用 PBMPM stretch clamp。'));
        pbmpmFoldout.body.append(createParamRow('Yield min', pbmpmYieldMin, 'PBMPM 主方向伸缩下界，不是 material yield_stress。'));
        pbmpmFoldout.body.append(createParamRow('Yield max', pbmpmYieldMax, 'PBMPM 主方向伸缩上界，不是 material yield_stress。'));
        solverFoldout.body.append(implicitInfo);
        solverFoldout.body.append(implicitFoldout.root);
        solverFoldout.body.append(pbmpmFoldout.root);

        const environmentFoldout = createFoldout('环境 / 网格 / 归一化', true);
        environmentFoldout.body.append(createRow('重力', gravityEnabled));
        environmentFoldout.body.append(createRow('重力 X', gravityX));
        environmentFoldout.body.append(createRow('重力 Y', gravityY));
        environmentFoldout.body.append(createRow('重力 Z', gravityZ));
        environmentFoldout.body.append(createRow('地面', groundEnabled));
        environmentFoldout.body.append(createRow('地面 Z', groundHeight));
        environmentFoldout.body.append(createRow('边界盒', boundingBoxEnabled));
        environmentFoldout.body.append(createRow('网格', nGrid));
        environmentFoldout.body.append(simAreaLabel);
        environmentFoldout.body.append(voxelGroupInfo);
        environmentFoldout.body.append(createRow('尺度', scale));
        environmentFoldout.body.append(createRow('透明阈值', opacity));
        environmentFoldout.body.append(useAabb);
        environmentFoldout.body.append(useBodyAabb);

        const createPartFoldout = createFoldout('创建 Part', true);
        createPartFoldout.body.append(selectedLabel);
        createPartFoldout.body.append(objectLabel);
        createPartFoldout.body.append(createRow('Part 名', objectName));
        createPartFoldout.body.append(createRow('Body ID', createBodyId));
        createPartFoldout.body.append(createBodyHint);
        createPartFoldout.body.append(createRow('材料', createMaterial));
        createPartFoldout.body.append(createRow('填充', createFill));
        createPartFoldout.body.append(createRow('\u5bc6\u5ea6', createDensity));
        createPartFoldout.body.append(addObject);

        const partFoldout = createFoldout('Part / Body 参数', true);
        partFoldout.body.append(createRow('Part', objectSelect));
        partFoldout.body.append(objectInfo);
        partFoldout.body.append(createRow('Body ID', objectBodyId));
        partFoldout.body.append(objectBodyHint);
        partFoldout.body.append(createRow('材料', objectMaterial));
        partFoldout.body.append(createRow('填充', objectFill));
        partFoldout.body.append(createRow('密度', objectDensity));
        partFoldout.body.append(createRow('E', objectE));
        partFoldout.body.append(createRow('泊松比', objectNu));
        partFoldout.body.append(deleteObject);
        partFoldout.body.append(clearObjects);

        const driveFoldout = createFoldout('驱动 / 初始冲量', true);
        driveFoldout.body.append(createSection('线性冲量'));
        driveFoldout.body.append(createRow('启用', linearEnabled));
        driveFoldout.body.append(createRow('力 X', linearFx));
        driveFoldout.body.append(createRow('力 Y', linearFy));
        driveFoldout.body.append(createRow('力 Z', linearFz));
        driveFoldout.body.append(createRow('步数', linearNumDt));
        driveFoldout.body.append(createRow('起始', linearStart));
        driveFoldout.body.append(createRow('抓取增益', dragGain));
        driveFoldout.body.append(captureLinear);
        driveFoldout.body.append(createSection('旋转驱动'));
        driveFoldout.body.append(createRow('启用', spinEnabled));
        driveFoldout.body.append(createRow('轴 X', spinAxisX));
        driveFoldout.body.append(createRow('轴 Y', spinAxisY));
        driveFoldout.body.append(createRow('轴 Z', spinAxisZ));
        driveFoldout.body.append(createRow('角速度', spinAngular));
        driveFoldout.body.append(createRow('平移', spinTranslation));
        driveFoldout.body.append(createRow('起始', spinStart));
        driveFoldout.body.append(createRow('结束', spinEnd));

        const runFoldout = createFoldout('时间 / 运行', true);
        runFoldout.body.append(createRow('帧间隔', frameDt));
        runFoldout.body.append(createRow('帧数', frameNum));
        runFoldout.body.append(createRow('子步长', substepDt));
        runFoldout.body.append(createParamRow('Damping', damping, '网格速度保留比例；1 表示不衰减，越小衰减越强。'));
        runFoldout.body.append(timeWarning);
        runFoldout.body.append(submit);
        runFoldout.body.append(cancel);
        runFoldout.body.append(exportTrace);
        runFoldout.body.append(status);

        const debugFoldout = createFoldout('调试 JSON', false);
        debugFoldout.body.append(copyPayload);
        debugFoldout.body.append(payloadView);

        const tabPages = createTabPages([
            { id: 'model', label: '模型' },
            { id: 'scene', label: '场景' },
            { id: 'object', label: '对象' },
            { id: 'drive', label: '驱动' },
            { id: 'solver', label: '求解' },
            { id: 'run', label: '运行' },
            { id: 'debug', label: '调试' }
        ], 'object');
        const tabPage = (id: string) => tabPages.pages.get(id) as Container;
        tabPage('model').append(modelFoldout.root);
        tabPage('scene').append(environmentFoldout.root);
        tabPage('object').append(createPartFoldout.root);
        tabPage('object').append(partFoldout.root);
        tabPage('drive').append(driveFoldout.root);
        tabPage('solver').append(solverFoldout.root);
        tabPage('run').append(runFoldout.root);
        tabPage('debug').append(debugFoldout.root);

        this.append(header);
        this.append(tabPages.root);
        this.append(tabPages.content);

        tooltips.register(officialConfig, '使用模型自带的 PhysGaussian 官方配置文件，适合复现论文案例', 'right');
        tooltips.register(useAabb, '用当前高亮 Gaussian 生成仿真区域 AABB，适合手动框定整体仿真范围', 'right');
        tooltips.register(useBodyAabb, '用当前 Part 所属 Body 的全部 Part 并集生成仿真区域 AABB', 'right');
        tooltips.register(addObject, '从当前选区创建 Part；Body ID=0 会优先沿用当前 Part 的 Body，没有当前 Part 时才新建 Body；重叠 Gaussian 归新 Part', 'right');
        tooltips.register(deleteObject, '删除当前下拉框选中的 Part，不影响同一 Body 的其他 Part', 'right');
        tooltips.register(captureLinear, '在画布上按住 Gaussian，用手掌抓取方式实时驱动它所在 Body 的 proxy', 'right');
        tooltips.register(submit, '提交后端仿真，完成后自动加载离线动画', 'right');
        tooltips.register(copyPayload, '复制提交给后端的 JSON 配置', 'right');

        tooltips.register(exportTrace, '\u5bfc\u51fa\u6700\u8fd1\u4e00\u6b21\u4eff\u771f\u7684 solver_trace.json\uff0c\u7528\u4e8e\u5206\u6790 Newton / GMRES / \u65f6\u95f4\u6b65\u8bb0\u5f55', 'right');

        refreshModels();
        updateTimeWarning();
        updatePreprocess();
        updateSimulation();
    }
}

export { PhysicsPanel };
