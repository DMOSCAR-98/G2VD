# Configuration Templates

This directory contains editable templates rather than the full internal
experiment grid. Copy a template, replace dataset paths/source names/checkpoint
paths with your local setup, and then run `main.py` or `test.py`.

Typical staged training:

1. `train_g2vd_wo_cfi_cd_clip.yaml`: train the backbone detector.
2. `train_g2vd_wo_cd_clip.yaml`: warm-start from stage 1 and enable CFIPipeline.
3. `train_g2vd_clip.yaml`: warm-start from stage 2 and enable CFIPipeline + CD.

The same pattern can be adapted to `xclip`, `demamba_clip`, and
`demamba_xclip` by changing `det_model.params.video_backbone` and the checkpoint
paths.

For staged training, set `det_model.checkpoint_path` to the checkpoint produced
by the previous stage. For evaluation, set `det_checkpoint_path` to the
checkpoint you want to test.

The templates intentionally use neutral split names. They are meant as release
examples, not as a full reproduction of every internal experiment command.
