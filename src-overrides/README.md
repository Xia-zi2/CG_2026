# Source Overrides

The folders here contain project-owned overlay files for upstream repositories.
They are intentionally stored as partial trees, not full third-party checkouts.

## Folders

- `supersplat-src/`: files to copy into the upstream SuperSplat `src/` folder.
- `physgaussian-src/`: files to copy into the upstream PhysGaussian repository
  root.

## Sync

```bash
rsync -av src-overrides/supersplat-src/ external/supersplat/src/
rsync -av src-overrides/physgaussian-src/ external/PhysGaussian/
```

Review `docs/EXTERNAL_LIBS.md` before uploading or publishing patches.
