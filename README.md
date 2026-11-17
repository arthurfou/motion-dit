# motion-dit

Text-to-motion generation with diffusion transformers (DiT: in-context, AdaLN, cross-attention), trained on HumanML3D.
Based on lab 5b of CSC_52002 (Generative AI, École polytechnique).

## Setup

```bash
pixi install
# put (or symlink) the dataset at ./humanml3d-data
ln -s /path/to/humanml3d-data ./humanml3d-data
```

## Usage

```bash
pixi run visualize                     # render an SMPL sample
pixi run generate batch_size=1 seed=2  # generate a motion (Hydra overrides accepted)
pixi run train
pixi run evaluate
pixi run lab                           # open the notebook in JupyterLab
```

`PYTHONPATH` is set to the project root automatically inside the pixi environment.
