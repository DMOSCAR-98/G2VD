# Model Zoo and Checkpoints

This release contains the training and evaluation code for G2VD. Model
checkpoints are not stored in git.

## Checkpoint Layout

Training writes checkpoints under seed-specific directories:

```text
seed_<seed>/
  train_results/
    <method>/<backbone_or_model>/<experiment_name>/
      *.pth
```

Evaluation loads the checkpoint specified by `det_checkpoint_path` in the test
config. Update that field to point to the model checkpoint you want to evaluate.
For staged training, also update `det_model.checkpoint_path` in the next-stage
training config to the checkpoint produced by the previous stage.

## Pretrained Backbones

Backbone implementations use Hugging Face Transformers or local pretrained
weight directories depending on the model. The default local paths use:

```text
pretrained_weights/
```

Several backbone wrappers call `from_pretrained(..., local_files_only=True)`.
Therefore, either place the corresponding Hugging Face snapshots under
`pretrained_weights/` or change `hf_repo` to a valid local path in your config.

Common local snapshot names used by the default code include:

```text
pretrained_weights/
  models--openai--clip-vit-base-patch16/
  models--microsoft--xclip-base-patch16/
  models--facebook--timesformer-base-finetuned-k400/
  models--MCG-NJU--videomae-base/
  models--google--vivit-b-16x2-kinetics400/
```

Some reproduced baselines use additional ImageNet/backbone weights, such as
Xception, ResNet, or SCNet checkpoints. Place them under `pretrained_weights/`
or adjust the corresponding model code/config for your local setup.

## VAE Pool Weights

CFIPipeline uses a pool of pretrained VAE models. The default local layout is:

```text
pretrained_weights/
  vae_pool/
    taehv/
      *.pth
    taesdv/
      taesdv.pth
    videovaeplus/
      *.ckpt
```

These weights are not redistributed in this repository. Download them from the
corresponding upstream projects and follow their licenses and usage terms.

## Public Weights

Public G2VD checkpoints will be listed here once released.

| Model | Backbone | Checkpoint | Notes |
| --- | --- | --- | --- |
| G2VD | CLIP | Coming soon | Paper setting |
| G2VD | XCLIP | Coming soon | Paper setting |
| G2VD | DeMamba-CLIP | Coming soon | Paper setting |
| G2VD | DeMamba-XCLIP | Coming soon | Paper setting |
