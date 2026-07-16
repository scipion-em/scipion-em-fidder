#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Scipion Team
# *
# * National Center of Biotechnology, CSIC, Spain
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************
"""
Batch fidder worker — runs INSIDE the ``fidder-0.0.8`` conda environment.

Why this exists
---------------
The ``fidder`` CLI loads the U-Net checkpoint and rebuilds a PyTorch-Lightning
``Trainer`` (CUDA context) on *every* invocation, and the protocol launches it
once per tilt-image (``fidder predict`` + ``fidder erase``). For a 40-60 tilt
series that is ~80-120 process cold starts, each reloading the model + CUDA.
That startup — not the inference — dominates the wall-clock.

This worker collapses a whole tilt-series (main + even + odd) into ONE process:
the model is loaded once, moved to the GPU once, and then every image is
processed in a loop. ``fidder.predict.predict.predict_fiducial_mask`` cannot be
reused as-is because it reloads the checkpoint and creates a fresh ``Trainer``
internally on each call; so its ~15-line body is reproduced here VERBATIM with
the model hoisted out of the loop (and ``Trainer.predict`` replaced by a direct
``model.predict_step`` call, which needs no Trainer). The predict pre/post
processing and the erase call are kept byte-for-byte equivalent to fidder 0.0.8
(``predict/predict.py``, ``predict/cli.py`` and ``erase/cli.py``) so the
scientific output is unchanged.

Input
-----
A single JSON manifest path. Schema::

    {
      "pixel_spacing": <float Å/px>,
      "probability_threshold": <float in (0, 1]>,
      "items": [
        {"input": "<in.mrc>", "erased_out": "<out.mrc>", "mask_out": "<mask.mrc>|null"},
        ...
      ]
    }

``mask_out`` is written only when present (used by the advanced
"save segmented stack" option); otherwise the mask is kept in memory and used
only to erase, saving a disk round-trip per image. A non-zero exit code signals
failure to the calling protocol (``runJob`` raises -> the TS is marked failed).
"""
import json
import sys

import einops
import mrcfile
import numpy as np
import torch
from einops import rearrange

from fidder.model import Fidder, get_latest_checkpoint
from fidder.utils import (calculate_resampling_factor, rescale_2d_bicubic,
                          rescale_2d_nearest)
from fidder.constants import TRAINING_PIXEL_SIZE, PIXELS_PER_FIDUCIAL
from fidder.predict.probabilities_to_mask import probabilities_to_mask
from fidder.erase import erase_masked_region


def _predict_mask(model, image, pixel_spacing, probability_threshold):
    """Reproduce fidder.predict.predict.predict_fiducial_mask with the model
    passed in (loaded once by the caller) instead of reloaded per call, and
    ``model.predict_step`` instead of ``Trainer(...).predict`` (same result,
    no per-image Trainer/CUDA setup). ``image`` is a single ``(h, w)`` tensor.
    """
    # prepare image (verbatim from predict_fiducial_mask)
    if image.ndim == 2:
        image = rearrange(image, "h w -> 1 1 h w")
    elif image.ndim == 3:
        image = rearrange(image, "b h w -> b 1 h w")
    image = torch.as_tensor(image, dtype=torch.float)
    h, w = image.shape[-2:]
    downscale_factor = calculate_resampling_factor(
        source=pixel_spacing, target=TRAINING_PIXEL_SIZE
    )
    image = rescale_2d_bicubic(image, factor=downscale_factor)
    image = rearrange(image, "1 1 h w -> 1 h w")

    # predict — model is already loaded, eval()'d and on the right device.
    # Trainer.predict(model, image) iterates `image` (1, h, w) over dim 0 and
    # calls predict_step on the single (h, w) image; do exactly that directly.
    probabilities = model.predict_step(image.squeeze(0))
    mask = probabilities_to_mask(
        probabilities=probabilities,
        threshold=probability_threshold,
        connected_pixel_count_threshold=(PIXELS_PER_FIDUCIAL // 4),
    )

    # rescale for output (verbatim from predict_fiducial_mask)
    probabilities = rearrange(probabilities, "h w -> 1 1 h w")
    probabilities = rescale_2d_bicubic(probabilities, size=(h, w))
    probabilities = torch.clamp(probabilities, min=0, max=1)
    rearrange(probabilities, "1 1 h w -> h w")
    mask = rearrange(mask, "h w -> 1 1 h w")
    mask = rescale_2d_nearest(mask, size=(h, w))
    mask = rearrange(mask, "1 1 h w -> h w")
    return mask, probabilities


def _erase(image, mask, pixel_spacing, output_image):
    """Reproduce fidder.erase.cli.erase_masked_region for a single image, using
    the in-memory mask (already (h, w)) instead of reading it back from disk.
    """
    image = torch.as_tensor(image).squeeze().float()
    mask = torch.as_tensor(mask, dtype=torch.bool).squeeze()
    if image.shape != mask.shape:
        raise ValueError('Shape mismatch between image and mask.')
    image, ps = einops.pack([image], pattern='* h w')
    mask, ps = einops.pack([mask], pattern='* h w')

    erased_images = torch.empty_like(image, dtype=torch.float32)
    for idx, (img, msk) in enumerate(zip(image, mask)):
        erased_images[idx] = erase_masked_region(
            image=img,
            mask=msk,
            background_intensity_model_resolution=(8, 8),
            background_intensity_model_samples=25000,
        )
    [erased_images] = einops.unpack(erased_images, pattern='* h w', packed_shapes=ps)
    mrcfile.write(
        name=output_image,
        data=np.array(erased_images, dtype=np.float32),
        voxel_size=pixel_spacing,
        overwrite=True,
    )


def main(manifest_path):
    with open(manifest_path) as f:
        job = json.load(f)

    pixel_spacing = float(job['pixel_spacing'])
    probability_threshold = float(job['probability_threshold'])
    items = job['items']

    # ---- LOAD THE MODEL ONCE (the whole point of this worker) ----
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Fidder.load_from_checkpoint(get_latest_checkpoint(), map_location='cpu')
    model.eval()
    model.to(device)
    print(f'[fidder_batch] model loaded once on {device}; '
          f'processing {len(items)} image(s)', flush=True)

    out_ps = (1, pixel_spacing, pixel_spacing)  # mask voxel size, as in predict/cli.py
    with torch.no_grad():
        for i, item in enumerate(items):
            in_img = item['input']
            erased_out = item['erased_out']
            mask_out = item.get('mask_out')

            images = torch.tensor(mrcfile.read(in_img)).float()
            images, pack_shapes = einops.pack([images], pattern='* h w')

            # one (h, w) image per file (packed -> (1, h, w))
            mask, _ = _predict_mask(model, images[0], pixel_spacing,
                                    probability_threshold)

            if mask_out:
                # int8 mask MRC, byte-equivalent to `fidder predict` output
                m = mask.to(torch.int8).cpu().numpy()
                [m] = einops.unpack(m[None, ...], pattern='* h w',
                                    packed_shapes=pack_shapes)
                mrcfile.write(name=mask_out, data=m.astype(np.int8),
                              voxel_size=out_ps, overwrite=True)

            _erase(images[0], mask, pixel_spacing, erased_out)
            print(f'[fidder_batch] {i + 1}/{len(items)} done: {erased_out}',
                  flush=True)

    print('[fidder_batch] finished OK', flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.stderr.write('usage: fidder_batch.py <manifest.json>\n')
        sys.exit(2)
    main(sys.argv[1])