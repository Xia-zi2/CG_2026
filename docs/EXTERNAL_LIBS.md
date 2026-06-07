# External Libraries And Overlay Sync

This project is an integration overlay. Keep upstream repositories outside this
Git repository and copy only the project-owned override files into them.

## Recommended Layout

```text
workspace/
  upload-repo/
  external/
    supersplat/
    PhysGaussian/
```

## SuperSplat

- Upstream: https://github.com/playcanvas/supersplat
- Recommended source: current upstream `main`, or the exact commit used by your
  experiment if you need bit-for-bit reproduction.
- Override source in this repo: `src-overrides/supersplat-src/`
- Target in upstream checkout: `external/supersplat/src/`

Sync example:

```bash
rsync -av src-overrides/supersplat-src/ external/supersplat/src/
```

Main modified areas:

- Physics panel UI and app controls.
- Physics session lifecycle.
- Splat metadata and motion handling.
- Worker-based motion execution.

## PhysGaussian

- Upstream: https://github.com/XPandora/PhysGaussian
- Recommended source: upstream release/commit matching your environment.
- Override source in this repo: `src-overrides/physgaussian-src/`
- Target in upstream checkout: `external/PhysGaussian/`

Sync example:

```bash
rsync -av src-overrides/physgaussian-src/ external/PhysGaussian/
```

Main modified areas:

- MPM solver stepping and utility behavior.
- Implicit/explicit integration contracts.
- Parameter decoding used by the backend bridge.

## Patch Workflow

If you prefer patches instead of copying files:

```bash
cd external/supersplat
git diff -- src > ../../supersplat-integration.patch

cd ../PhysGaussian
git diff > ../../physgaussian-integration.patch
```

Apply later with:

```bash
git apply supersplat-integration.patch
git apply physgaussian-integration.patch
```

Keep generated patch files small and review them before upload. Do not include
full third-party repositories unless there is a deliberate project decision to
maintain a fork.
