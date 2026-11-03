# sf2vab.py

Convert a SoundFont (`.sf2`) or DLS level 1/2 sound bank (`.dls`) into a PlayStation Psy-Q/libsnd VAB sound bank. The script is standalone and uses only the Python standard library.

## Requirements

- Python 3
- An input `.sf2` or `.dls` file
- For `--seq`, a PlayStation Psy-Q/libsnd `.seq` file

## Basic use

```text
python sf2vab.py soundfont.sf2
```

By default, the script converts bank 0 and writes `soundfont.vab`. Specify a different file name with `-o`:

```text
python sf2vab.py soundfont.sf2 -o instruments.vab
```

A VAB contains a header and tone/sample data. Use `--split` to produce the separate `.vh` header and `.vb` body required by APIs such as `SsVabOpenHead` and `SsVabTransBody`:

```text
python sf2vab.py soundfont.sf2 --split -o instruments
```

This writes `instruments.vh` and `instruments.vb`.

## Select the VAB layout

These layout options are mutually exclusive.

### One SoundFont bank

```text
python sf2vab.py soundfont.sf2 --sf2-bank 5 -o bank5.vab
```

`--sf2-bank` selects the preset bank (default 0); program numbers in the VAB correspond to MIDI program numbers. For DLS, the bank is selected using its bank-select MSB, or its LSB when the MSB is zero. DLS drum instruments use bank 128.

### Fixed GM layout

```text
python sf2vab.py soundfont.sf2 --gm -o gm.vab
```

Programs 0–123 hold melodic GM programs. Programs 124–127 hold the selected drum kit (default kit program 0), split into four 16-note groups covering notes 24–87. Change the kit with `--drum-kit N`. Missing melodic programs alias another available program; GM programs 124–127 are reserved for the drum groups and are not included as melodic programs.

Pair this with `mid2seqp.py --gm` for a matching SEQ program layout.

### Merge all banks

```text
python sf2vab.py soundfont.sf2 --all-banks -o all.vab
```

This merges presets from all banks into one VAB and also writes `all.vmap` by default. Override the map path with `--voice-map`. The `.vmap` maps source SoundFont bank/program pairs to VAB program numbers and includes key ranges for drum kits split across multiple VAB programs.

Use the map to convert a MIDI consistently:

```text
python mid2seqp.py --voice-map all.vmap song.mid song.seq
```

### Build a VAB for one SEQ

```text
python mid2seqp.py song.mid song.seq
python sf2vab.py soundfont.sf2 --seq song.seq
```

The `--seq` mode uses the SEQ's appended instrument table (written by default by `mid2seqp.py`) to preserve the song's program assignments and include only zones covering notes used by the song. Its default output is `song.vab`. A SEQ without that table is interpreted as GM: melodic programs use bank 0, and programs played only on MIDI channel 10 are treated as drum kits.

## Options

```text
python sf2vab.py input-file [options]
```

### Output and mapping

- `-o, --output PATH`: output VAB path. Default is the input name with `.vab`; with `--seq`, the default is based on the SEQ filename.
- `--split`: write `.vh` and `.vb` files instead of one `.vab`.
- `--sf2-bank N`: select a preset bank (default 0).
- `--gm`: use the fixed GM layout.
- `--drum-kit N`: choose the percussion preset for `--gm` (default 0).
- `--all-banks`: merge banks and write a voice map.
- `--voice-map PATH`: choose the voice-map output path for `--all-banks`.
- `--seq PATH`: build the VAB for the specified SEQ.

### Sound and memory

- `--mono`: mix stereo zones down to one centered tone to reduce SPU voice use.
- `--no-reverb`: do not flag tones for reverb and reserve no SPU RAM for a reverb work area.
- `--reverb-type TYPE`: choose the runtime reverb work-area size; choices are `off`, `room`, `studio-a`, `studio-b`, `studio-c`, `hall`, `space`, `echo`, `delay`, and `pipe` (default `space`).
- `--spu-budget SIZE`: override available VB budget, e.g. `300k`, `0x40000`, or a byte count. The tool still caps this at the VAB body maximum.
- `--compact`: resample larger samples as needed to fit the budget.
- `--min-rate HZ`: lowest sample rate allowed during compaction (default 8000 Hz).
- `--keep-small SIZE`: leave samples smaller than this 16-bit PCM size untouched during compaction (default `2k`).
- `--atten-scale VALUE`: scale instrument attenuation. Defaults are 0.4 for SF2 and 1.0 for DLS.
- `--bend-range N`: pitch-bend range in semitones written to tones (default 2).
- `--no-pitch-fix`: disable resampling used to keep notes within the SPU's pitch limit.

Run `python sf2vab.py --help` for the current argparse help.

## Conversion notes and limits

- The VAB format supports up to 128 programs, 16 tones per program, and 254 VAG samples. Oversized programs are reduced by mixing stereo pairs, merging adjacent key splits, or dropping tones; excess samples may also require merging or dropping tones.
- VAB tones do not provide velocity ranges. The converter keeps zones that cover velocity 100, or falls back to the available zones if none cover it.
- Missing presets can be aliased or replaced by a fallback, depending on the selected layout. Warnings are printed for missing or unusable presets and for resource limits.
- SF2 samples with ROM flags are skipped. DLS waves must be supported PCM (8/16/24/32-bit); unsupported wave formats are skipped.
- `--compact` leaves small samples alone and lowers a common sample-rate cap for larger samples until the aligned VB body fits the budget, subject to `--min-rate`. If it still cannot fit, a warning is emitted.
- Reverb tones are flagged in the VAB unless `--no-reverb` is used. `--reverb-type` affects the estimated SPU RAM reserved for the runtime reverb work area; it does not configure the runtime reverb itself.
- PSX SPU RAM is limited. The tool reports the estimated VB placement and available budget, but final placement and runtime behavior depend on the target program and sound driver.

## Outputs

- `.vab`: VAB header followed by VB sample data; or `.vh` and `.vb` with `--split`.
- `.vmap`: text bank/program-to-VAB-program mapping when using `--all-banks` (unless redirected with `--voice-map`).
