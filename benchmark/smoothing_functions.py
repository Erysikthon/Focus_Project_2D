import numpy as np

def apply_min_duration_filter(preds, min_duration=5, background_class=0):
    """
    Remove short predicted runs of non-background classes.
    Any contiguous run shorter than min_duration frames is replaced by
    the preceding class (or background if at the start).
    """
    preds = np.array(preds, dtype=int)
    i = 0
    while i < len(preds):
        cls = preds[i]
        if cls == background_class:
            i += 1
            continue
        # Find end of this run
        j = i
        while j < len(preds) and preds[j] == cls:
            j += 1
        run_length = j - i
        if run_length < min_duration:
            replacement = preds[i - 1] if i > 0 else background_class
            preds[i:j] = replacement
        i = j
    return preds

def apply_gap_fill(preds, max_gap=5, background_class = 0):
    """
    Fill short background gaps between identical behaviors.
    If background appears for <= max_gap frames between the same behavior on both sides,
    replace the background with that behavior.
    """
    preds = np.array(preds, dtype=int)
    i = 0
    while i < len(preds):
        if preds[i] != background_class:
            i += 1
            continue
        # Found a background run — find its extent
        j = i
        while j < len(preds) and preds[j] == background_class:
            j += 1
        gap_length = j - i
        # Check if gap is short enough and flanked by the same behavior
        if gap_length <= max_gap and i > 0 and j < len(preds):
            before = preds[i - 1]
            after = preds[j]
            if before == after and before != background_class:
                preds[i:j] = before
        i = j
    return preds