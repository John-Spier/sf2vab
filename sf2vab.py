#!/usr/bin/env python3
"""
sf2vab.py — Convert a SoundFont (.sf2) or DLS (.dls, level 1/2) to a PlayStation
.VAB sound bank (or a .VH header + .VB body pair) for libsnd.

Standalone: shares its SF2/DLS parsing, resampling and compaction design with
sf2ton.py (Sega Saturn TON), but does not import it.

Usage:
  python sf2vab.py input.sf2|input.dls [-o output.vab] [--split]
                   [--sf2-bank 0 | --gm [--drum-kit 0] | --all-banks [--voice-map X.vmap]
                    | --seq song.seq]
                   [--mono] [--compact [--min-rate 8000] [--keep-small 2k]]
                   [--no-reverb] [--reverb-type space] [--spu-budget SIZE]

VAB format (Psy-Q "File Formats" + libsnd "Run-Time Library Reference"):
  VabHdr (32 bytes)            'pBAV', version 7, id, file size, ps, ts, vs, ...
  ProgAtr[128] (16 bytes each) tones, mvol, prior, mode, mpan, ...
  VagAtr[16] x ps (32 each)    one 512-byte block per program that has tones,
                               in program order (libsnd's "actual program")
  VAG size table (256 x u16)   entry 0 = 0, entry i = size of VAG i / 8
  VAG bodies (.VB)             SPU ADPCM, max 254 VAGs, up to 516096 bytes
  Limits: 128 programs, 16 tones per program, 254 VAGs.

Conversion notes:
  - Pitch: libsnd plays note n of a tone at 44100 Hz * 2^((n - center)/12 +
    shift/1536); shift (0..127) is 1/128 semitone, used in 1/16 steps.  Samples
    keep their own rate; the rate difference is folded into center/shift.
  - The SPU can't play faster than 4x (176.4 kHz): a tone can't go more than 2
    octaves above the note where it plays at 44.1 kHz.  Samples whose key range
    needs more are resampled down (up to key 108; --no-pitch-fix to disable).
  - Loops: the SPU loops on 28-sample ADPCM blocks.  Loop starts are aligned by
    padding the front with silence; loop lengths are unrolled (short loops) or
    resampled to a multiple of 28 (pitch corrected).  The loop-start block is
    encoded against the history at the loop end, so the seam doesn't click.
  - Volume: libsnd squares the combined volume, so tone vol = 127 * sqrt(gain).
  - Envelope: SF2 volume envelope -> SPU ADSR (linear attack, exponential
    decay/sustain/release, matched by simulating the SPU envelope).  Deep sustain
    levels (< -24 dB, below the SPU's lowest sustain level) are done by skipping
    the decay phase and decaying in the sustain phase instead.
  - Reverb: tones are flagged for reverb (VagAtr.mode = 4) unless --no-reverb.
    The reverb type is chosen at run time (SsUtSetReverbType); --reverb-type
    only sets how much SPU RAM is kept free for its work area.
  - Velocity layers: VAB tones have no velocity range; zones covering velocity
    100 are kept.  Programs with more than 16 zones are reduced (stereo pairs
    mixed to mono, then neighbouring key splits merged).

GM layout (--gm), fixed so any GM SEQ converter can target any GM VAB:
  programs 0-123   GM melodic programs (missing ones alias the nearest program
                   of the same GM family, else the lowest program)
  programs 124-127 the drum kit (bank 128, --drum-kit), 16 notes per program:
                   124 = notes 24-39, 125 = 40-55, 126 = 56-71, 127 = 72-87.
                   A SEQ plays drum note n with program 124 + (n - 24) // 16.
  GM programs 124-127 (Telephone, Helicopter, Applause, Gunshot) are dropped.

Merged banks (--all-banks), like sf2ton --all-banks: every bank goes into one
VAB with each distinct instrument stored once, so the VAB has exactly as many
programs as the soundfont has instruments (best on a soundfont trimmed to one
song with trim_soundfont_to_midi.py).  The .vmap is sf2ton's format
  <sf2_bank> <program> <vab_program> [keys=<lo>-<hi>] <name>
where drum kits (bank 128) always carry keys=: a kit with more than 16 tones is
split over consecutive programs by key range, and a SEQ converter plays drum
note n with the program of the line whose range holds n.  mid2seq.py -m reads it.

VAB for a SEQ (--seq song.seq), all-banks in reverse: mid2seq.py numbers the
instruments a song plays (drum kits in groups of 16 notes) and appends that
numbering to the SEQ as an instrument table after its end-of-track ('VMAP',
u32 length, lines in the voice map format; libsnd stops reading at the
end-of-track).  sf2vab builds a VAB with exactly those programs, keeping only
the zones the SEQ's notes play, so no voice map file is needed.  SEQs without
a table (SMF2SEQ, Awave) are read as GM: program p = bank 0 program p, and
programs used on channel 10 = drum kit p.

Mono (--mono): stereo zones (SF2 linked L/R samples, DLS stereo waves, or two
hard-panned zones with the same key range) are mixed to one centred tone, so a
note takes one of the SPU's 24 voices instead of two.  Without --mono this only
happens in programs that would otherwise exceed 16 tones.

SPU RAM: the VB is loaded between 0x01010 and the reverb work area at the top
of the 512 KB SPU RAM (libspu SpuMalloc range 0x01010-0x7FFFF, VB transfers
reserve multiples of 64 bytes).
"""

import struct
import sys
import os
import math
import operator
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ── SF2 Parser ──────────────────────────────────────────────────────

def read_sf2_chunks(data: bytes) -> dict:
    """Parse RIFF/SF2 into a dict of chunk name -> data."""
    if data[:4] != b'RIFF' or data[8:12] != b'sfbk':
        raise ValueError("Not a valid SF2 file")
    return read_sf2_chunks_flat(data[12:8 + struct.unpack_from('<I', data, 4)[0]])


def read_sf2_chunks_flat(data: bytes) -> dict:
    """Parse (nested) LIST chunks into a flat dict."""
    chunks = {}
    pos = 0
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4].decode('ascii', errors='replace')
        chunk_size = struct.unpack('<I', data[pos + 4:pos + 8])[0]
        if chunk_id == 'LIST':
            chunks.update(read_sf2_chunks_flat(data[pos + 12:pos + 8 + chunk_size]))
        else:
            chunks[chunk_id] = data[pos + 8:pos + 8 + chunk_size]
        pos += 8 + chunk_size + (chunk_size & 1)
    return chunks


@dataclass
class SF2Sample:
    name: str
    start: int      # offset in smpl chunk (in samples)
    end: int
    loop_start: int
    loop_end: int
    sample_rate: int
    original_key: int
    pitch_correction: int  # cents
    sample_type: int       # 1=mono, 2=right, 4=left, 8=linked, 0x8000=ROM
    link: int = -1         # the other channel of a stereo pair (sample index)


@dataclass
class SF2Zone:
    """An instrument zone (key/vel range -> sample + generators), SF2 units."""
    key_lo: int = 0
    key_hi: int = 127
    vel_lo: int = 0
    vel_hi: int = 127
    sample_id: int = -1
    attenuation: float = 0.0    # cB
    pan: float = 0.0            # -500..500
    sample_modes: int = 0       # 0=no loop, 1=loop, 3=loop+release
    root_key: int = -1          # override (-1 = use sample header)
    fine_tune: int = 0          # cents
    coarse_tune: int = 0        # semitones
    vol_attack: float = -12000  # timecents
    vol_decay: float = -12000
    vol_sustain: float = 0      # cB attenuation (0 = full level)
    vol_release: float = -12000
    start_off: int = 0
    end_off: int = 0
    loop_start_off: int = 0
    loop_end_off: int = 0


def parse_sf2_samples(shdr_data: bytes) -> List[SF2Sample]:
    samples = []
    for i in range(len(shdr_data) // 46 - 1):  # last entry is EOS
        off = i * 46
        name = shdr_data[off:off + 20].split(b'\x00')[0].decode('ascii', errors='replace')
        # SF2 2.04 7.10: ..., byOriginalPitch, chPitchCorrection, wSampleLink, sfSampleType
        start, end, ls, le, sr, key, corr, link, stype = struct.unpack(
            '<IIIIIBbHH', shdr_data[off + 20:off + 46])
        samples.append(SF2Sample(name, start, end, ls, le, sr, key, corr, stype, link))
    # Keep only consistent stereo links: a left (4) and a right (2) sample
    # pointing at each other.  Some files have garbage in these fields.
    for i, s in enumerate(samples):
        p = samples[s.link] if 0 <= s.link < len(samples) else None
        types = {s.sample_type & 0x7FFF, p.sample_type & 0x7FFF} if p else set()
        if not (p and p is not s and p.link == i and types == {2, 4}):
            s.link = -2  # -2: checked, not stereo
    for s in samples:
        s.link = max(-1, s.link)
    return samples


GEN_KEY_RANGE, GEN_VEL_RANGE = 43, 44
GEN_INSTRUMENT, GEN_SAMPLE_ID = 41, 53
GEN_ATTENUATION, GEN_PAN, GEN_SAMPLE_MODES = 48, 17, 54
GEN_ROOT_KEY, GEN_FINE_TUNE, GEN_COARSE_TUNE = 58, 52, 51
GEN_VOL_ATTACK, GEN_VOL_DECAY, GEN_VOL_SUSTAIN, GEN_VOL_RELEASE = 34, 36, 37, 38

# SF2 2.04 defaults (section 8.1.3) for the generators used; others default to 0.
GEN_DEFAULTS = {GEN_VOL_ATTACK: -12000, GEN_VOL_DECAY: -12000, GEN_VOL_RELEASE: -12000,
                GEN_ROOT_KEY: -1}
# Generators not added from the preset level (SF2 spec 8.5).
GEN_NON_ADDITIVE = {0, 1, 2, 3, 4, 12, 45, 46, 47, 50, 54, 57, 58,
                    GEN_INSTRUMENT, GEN_SAMPLE_ID, GEN_KEY_RANGE, GEN_VEL_RANGE}


def _parse_zone_lists(hdr, rec_size, bag_field_off, bag_data, gen_data, link_gen):
    """inst/ibag/igen or phdr/pbag/pgen -> [(name, record, [zone_gens, ...])].
    A first zone without the link generator is the global zone (defaults)."""
    bags = [struct.unpack_from('<H', bag_data, i * 4)[0] for i in range(len(bag_data) // 4)]
    gens = [struct.unpack_from('<Hh', gen_data, i * 4) for i in range(len(gen_data) // 4)]
    out = []
    for i in range(len(hdr) // rec_size - 1):
        rec = hdr[i * rec_size:(i + 1) * rec_size]
        name = rec[:20].split(b'\x00')[0].decode('ascii', errors='replace')
        b0 = struct.unpack_from('<H', hdr, i * rec_size + bag_field_off)[0]
        b1 = struct.unpack_from('<H', hdr, (i + 1) * rec_size + bag_field_off)[0]
        glob, zones = {}, []
        for bi in range(b0, min(b1, len(bags))):
            g0 = bags[bi]
            g1 = bags[bi + 1] if bi + 1 < len(bags) else len(gens)
            z = {gid: val for gid, val in gens[g0:min(g1, len(gens))]}
            if link_gen not in z:
                if bi == b0:
                    glob = z
                continue
            zones.append(z)
        out.append((name, rec, [{**glob, **z} for z in zones]))
    return out


def _gen_range(z: dict, gid: int) -> Tuple[int, int]:
    v = z.get(gid)
    if v is None or v == 0:  # 0..0 is unplayable; some editors mean "unset"
        return 0, 127
    v &= 0xFFFF
    return v & 0xFF, (v >> 8) & 0xFF


def _zone_from_gens(g: dict, key, vel) -> SF2Zone:
    def get(gid):
        return g.get(gid, GEN_DEFAULTS.get(gid, 0))
    z = SF2Zone()
    z.key_lo, z.key_hi = key
    z.vel_lo, z.vel_hi = vel
    z.sample_id = g[GEN_SAMPLE_ID] & 0xFFFF
    z.attenuation = get(GEN_ATTENUATION)
    z.pan = get(GEN_PAN)
    z.sample_modes = get(GEN_SAMPLE_MODES) & 3
    root = get(GEN_ROOT_KEY)
    z.root_key = root if 0 <= root <= 127 else -1
    z.fine_tune = get(GEN_FINE_TUNE)
    z.coarse_tune = get(GEN_COARSE_TUNE)
    z.vol_attack = get(GEN_VOL_ATTACK)
    z.vol_decay = get(GEN_VOL_DECAY)
    z.vol_sustain = get(GEN_VOL_SUSTAIN)
    z.vol_release = get(GEN_VOL_RELEASE)
    z.start_off = get(0) + 32768 * get(4)
    z.end_off = get(1) + 32768 * get(12)
    z.loop_start_off = get(2) + 32768 * get(45)
    z.loop_end_off = get(3) + 32768 * get(50)
    return z


def parse_sf2_presets(chunks: dict):
    """-> [(name, program, bank, [SF2Zone, ...])], preset and instrument levels
    combined (global zones, offsets added, key/velocity ranges intersected)."""
    for c in ('inst', 'ibag', 'igen', 'phdr', 'pbag', 'pgen'):
        if not chunks.get(c):
            raise ValueError(f"Missing '{c}' chunk in SF2")
    insts = _parse_zone_lists(chunks['inst'], 22, 20, chunks['ibag'], chunks['igen'],
                              GEN_SAMPLE_ID)
    presets = _parse_zone_lists(chunks['phdr'], 38, 24, chunks['pbag'], chunks['pgen'],
                                GEN_INSTRUMENT)
    result = []
    for name, rec, pzones in presets:
        prog, bank = struct.unpack_from('<HH', rec, 20)
        zones = []
        for pz in pzones:
            inst_idx = pz[GEN_INSTRUMENT] & 0xFFFF
            if inst_idx >= len(insts):
                continue
            pk, pv = _gen_range(pz, GEN_KEY_RANGE), _gen_range(pz, GEN_VEL_RANGE)
            for iz in insts[inst_idx][2]:
                ik, iv = _gen_range(iz, GEN_KEY_RANGE), _gen_range(iz, GEN_VEL_RANGE)
                key = (max(pk[0], ik[0]), min(pk[1], ik[1]))
                vel = (max(pv[0], iv[0]), min(pv[1], iv[1]))
                if key[0] > key[1] or vel[0] > vel[1]:
                    continue
                g = dict(iz)
                for gid, val in pz.items():
                    if gid not in GEN_NON_ADDITIVE:
                        g[gid] = g.get(gid, GEN_DEFAULTS.get(gid, 0)) + val
                zones.append(_zone_from_gens(g, key, vel))
        result.append((name, prog, bank, zones))
    return result

# ── DLS Parser ──────────────────────────────────────────────────────
#
# DLS level 1/2 is turned into the same structures as the SF2 parser (16-bit
# PCM blob, SF2Sample headers, presets of SF2Zones in SF2 units).

DLS_SRC_NONE, DLS_SRC_VELOCITY, DLS_SRC_KEY = 0x0000, 0x0002, 0x0003
DLS_DST_GAIN, DLS_DST_PITCH, DLS_DST_PAN = 0x0001, 0x0003, 0x0004
DLS_DST_EG1_ATTACK, DLS_DST_EG1_DECAY = 0x0206, 0x0207
DLS_DST_EG1_RELEASE, DLS_DST_EG1_SUSTAIN = 0x0209, 0x020A
DLS_DRUM_FLAG = 0x80000000


def _riff_children(data: bytes, a: int, b: int):
    """Yield (id, list_type, payload_start, payload_end, chunk_start)."""
    while a + 8 <= b:
        cid = data[a:a + 4]
        size = struct.unpack_from('<I', data, a + 4)[0]
        end = min(b, a + 8 + size)
        if cid in (b'LIST', b'RIFF'):
            yield cid, data[a + 8:a + 12], a + 12, end, a
        else:
            yield cid, None, a + 8, end, a
        a += 8 + size + (size & 1)


def _dls_index(data: bytes, a: int, b: int) -> dict:
    out = {}
    for cid, lt, p, e, _ in _riff_children(data, a, b):
        out.setdefault(lt if lt is not None else cid, (p, e))
    return out


def _dls_name(data: bytes, idx: dict, default: str) -> str:
    if b'INFO' in idx:
        info = _dls_index(data, *idx[b'INFO'])
        if b'INAM' in info:
            p, e = info[b'INAM']
            name = data[p:e].split(b'\0')[0].decode('latin1').strip()
            if name:
                return name
    return default


def _dls_wsmp(data: bytes, p: int) -> dict:
    cb, unity, fine, att, _opts, nloops = struct.unpack_from('<IHhiII', data, p)
    loop = None
    if nloops:
        _lcb, ltype, lstart, llen = struct.unpack_from('<IIII', data, p + cb)
        loop = (lstart, llen, ltype)
    return dict(unity=unity, fine=fine, att=-att / 65536.0, loop=loop)


def _dls_articulation(data: bytes, idx: dict) -> dict:
    conns = {}
    for lt in (b'lart', b'lar2'):
        if lt not in idx:
            continue
        for cid, _, p, _, _ in _riff_children(data, *idx[lt]):
            if cid not in (b'art1', b'art2'):
                continue
            cb, n = struct.unpack_from('<II', data, p)
            for i in range(n):
                src, ctl, dst, _tr, scale = struct.unpack_from('<HHHHi', data, p + cb + 12 * i)
                conns[(src, ctl, dst)] = scale / 65536.0
    return conns


def _pcm_channels(raw: bytes, bits: int, nch: int):
    """Interleaved little-endian PCM -> one array('h') per channel."""
    import array
    if bits == 8:
        a = array.array('h', ((v - 128) << 8 for v in raw))
    else:
        width = bits // 8
        n = len(raw) // width
        buf = bytearray(2 * n)
        buf[0::2] = raw[width - 2:n * width:width]
        buf[1::2] = raw[width - 1:n * width:width]
        a = array.array('h')
        a.frombytes(bytes(buf))
        if sys.byteorder == 'big':
            a.byteswap()
    return [a[c::nch] for c in range(nch)]


def parse_dls(data: bytes, warn):
    """DLS -> (samples, presets, raw 16-bit LE pcm)."""
    if data[:4] != b'RIFF' or data[8:12] != b'DLS ':
        raise ValueError("Not a valid DLS file")
    top = _dls_index(data, 12, len(data))
    if b'wvpl' not in top or b'lins' not in top:
        raise ValueError("DLS file has no wave pool or instrument list")
    import array
    blob = bytearray()
    samples: List[SF2Sample] = []
    wave_chans, wave_wsmp, wave_at = [], [], {}
    wv0 = top[b'wvpl'][0]
    for cid, lt, p, e, start in _riff_children(data, *top[b'wvpl']):
        if lt != b'wave':
            continue
        wi = len(wave_chans)
        wave_at[start - wv0] = wi
        idx = _dls_index(data, p, e)
        name = _dls_name(data, idx, f"wave {wi}")
        wave_wsmp.append(_dls_wsmp(data, idx[b'wsmp'][0]) if b'wsmp' in idx else None)
        wave_chans.append([])
        if b'fmt ' not in idx or b'data' not in idx:
            warn(f"wave '{name}' has no fmt/data chunk, skipped")
            continue
        fp = idx[b'fmt '][0]
        tag, nch, rate, _, _, bits = struct.unpack_from('<HHIIHH', data, fp)
        if tag == 0xFFFE:
            tag = struct.unpack_from('<H', data, fp + 24)[0]
        if tag != 1 or bits not in (8, 16, 24, 32) or nch < 1:
            warn(f"wave '{name}': unsupported format (tag {tag}, {bits}-bit), skipped")
            continue
        dp, de = idx[b'data']
        for ch, pcm in enumerate(_pcm_channels(data[dp:de], bits, nch)):
            s0 = len(blob) // 2
            if sys.byteorder == 'big':
                pcm = array.array('h', pcm)
                pcm.byteswap()
            blob += pcm.tobytes()
            sname = name if nch == 1 else f"{name} ch{ch}"
            samples.append(SF2Sample(sname, s0, s0 + len(pcm), s0, s0, rate, 60, 0, 1))
            wave_chans[wi].append(len(samples) - 1)
        if len(wave_chans[wi]) == 2:  # stereo wave: link the channels
            a, b = wave_chans[wi]
            samples[a].link, samples[b].link = b, a

    cues = None
    if b'ptbl' in top:
        p = top[b'ptbl'][0]
        cb, n = struct.unpack_from('<II', data, p)
        cues = [struct.unpack_from('<I', data, p + cb + 4 * i)[0] for i in range(n)]

    presets = []
    for cid, lt, p, e, _ in _riff_children(data, *top[b'lins']):
        if lt != b'ins ':
            continue
        idx = _dls_index(data, p, e)
        if b'insh' not in idx:
            continue
        _nreg, ulbank, prog = struct.unpack_from('<III', data, idx[b'insh'][0])
        msb, lsb = (ulbank >> 8) & 0x7F, ulbank & 0x7F
        bank = 128 if ulbank & DLS_DRUM_FLAG else (msb or lsb)
        name = _dls_name(data, idx, f"bank {bank} prog {prog}")
        inst_art = _dls_articulation(data, idx)
        zones = []
        if b'lrgn' in idx:
            for rcid, rlt, rp, re_, _ in _riff_children(data, *idx[b'lrgn']):
                if rlt not in (b'rgn ', b'rgn2'):
                    continue
                z = _dls_region(data, _dls_index(data, rp, re_), inst_art, cues, wave_at,
                                wave_chans, wave_wsmp, name, warn)
                if z is not None:
                    zones.append(z)
        presets.append((name, prog & 0x7F, bank, zones))
    return samples, presets, bytes(blob)


def _dls_region(data, idx, inst_art, cues, wave_at, wave_chans, wave_wsmp, inst_name, warn):
    """One rgn/rgn2 list -> SF2Zone (SF2 units), or None."""
    if b'rgnh' not in idx or b'wlnk' not in idx:
        return None
    klo, khi, vlo, vhi = struct.unpack_from('<HHHH', data, idx[b'rgnh'][0])
    if vlo == vhi == 0:
        vlo, vhi = 0, 127
    _opt, _phase, chan_mask, table = struct.unpack_from('<HHII', data, idx[b'wlnk'][0])
    wi = wave_at.get(cues[table]) if cues is not None and table < len(cues) else table
    if wi is None or wi >= len(wave_chans) or not wave_chans[wi]:
        warn(f"'{inst_name}': region references a missing/unusable wave, skipped")
        return None
    ch = max(0, (chan_mask & -chan_mask).bit_length() - 1) if chan_mask else 0
    sample_id = wave_chans[wi][min(ch, len(wave_chans[wi]) - 1)]
    ws = _dls_wsmp(data, idx[b'wsmp'][0]) if b'wsmp' in idx else wave_wsmp[wi]
    ws = ws or dict(unity=60, fine=0, att=0.0, loop=None)
    conns = {**inst_art, **_dls_articulation(data, idx)}
    key = (klo + min(khi, 127)) / 2.0
    vel = REPRESENTATIVE_VELOCITY

    def val(dst, default=0.0, src=DLS_SRC_NONE):
        return conns.get((src, 0, dst), default)

    def tc(dst, default=-12000.0):
        v = val(dst, default)
        if v <= -32768.0:
            return -12000.0
        v += val(dst, 0.0, DLS_SRC_KEY) * key / 128.0
        v += val(dst, 0.0, DLS_SRC_VELOCITY) * vel / 128.0
        return max(-12000.0, v)

    z = SF2Zone()
    z.key_lo, z.key_hi = min(klo, 127), min(khi, 127)
    z.vel_lo, z.vel_hi = min(vlo, 127), min(vhi, 127)
    z.sample_id = sample_id
    z.attenuation = max(0.0, ws['att'] - val(DLS_DST_GAIN))
    z.pan = max(-500.0, min(500.0, val(DLS_DST_PAN)))
    z.root_key = min(127, ws['unity'])
    z.fine_tune = int(round(ws['fine'] + val(DLS_DST_PITCH)))
    if ws['loop']:
        lstart, llen, ltype = ws['loop']
        z.sample_modes = 3 if ltype == 1 else 1
        z.loop_start_off, z.loop_end_off = lstart, lstart + llen
    z.vol_attack = tc(DLS_DST_EG1_ATTACK)
    z.vol_decay = tc(DLS_DST_EG1_DECAY)
    z.vol_release = tc(DLS_DST_EG1_RELEASE)
    z.vol_sustain = 1000.0 - max(0.0, min(1000.0, val(DLS_DST_EG1_SUSTAIN, 1000.0)))
    return z


def load_soundfont(path: str, warn):
    """Read an .sf2 or .dls -> (samples, presets, raw_16le_pcm, kind)."""
    with open(path, 'rb') as f:
        data = f.read()
    if data[:4] == b'RIFF' and data[8:12] == b'DLS ':
        samples, presets, raw = parse_dls(data, warn)
        return samples, presets, raw, 'dls'
    chunks = read_sf2_chunks(data)
    return parse_sf2_samples(chunks['shdr']), parse_sf2_presets(chunks), chunks['smpl'], 'sf2'

# ── Constants ───────────────────────────────────────────────────────

SPU_RATE = 44100            # pitch 0x1000; libsnd's rate at note == center
SPU_RAM_SIZE = 0x80000
SPU_VB_START = 0x01010      # SpuMalloc / VB transfer range is 0x01010-0x7FFFF
VB_MAX_SIZE = 516096        # VAB body limit (Psy-Q File Formats)
VB_ALIGN = 64               # VB transfers reserve multiples of 64 bytes
MAX_PROGRAMS = 128
MAX_TONES = 16              # per program
MAX_VAGS = 254
VAB_VERSION = 7
BLOCK = 28                  # samples per 16-byte ADPCM block
# Max pitch 0x3FFF ~ 4x: a tone reaches at most 2 octaves above its 44.1 kHz note.
SPU_MAX_UP = 24.0
PITCH_FIX_MAX_KEY = 108     # don't resample for keys above this
MAX_EFF_NOTE = 127.0        # center note limit
MIN_RESAMPLED_LOOP = 8
UNROLL_MAX = 2048           # unroll loops up to this many points instead of resampling
REPRESENTATIVE_VELOCITY = 100
DRUM_BANK = 128             # SF2 percussion bank

# Reverb work area at the top of SPU RAM per reverb type (libspu SpuSetReverbModeParam)
REVERB_WORK_AREA = {
    'off': 0, 'room': 0x26C0, 'studio-a': 0x1F40, 'studio-b': 0x4840,
    'studio-c': 0x6FE0, 'hall': 0xADE0, 'space': 0xF6C0, 'echo': 0x18040,
    'delay': 0x18040, 'pipe': 0x3C00,
}
DEFAULT_REVERB_TYPE = 'space'

# GM layout (--gm)
GM_DRUM_PROG = 124          # first drum program
GM_DRUM_FIRST_NOTE = 24
GM_DRUM_NOTES_PER_PROG = MAX_TONES

# ── Helpers ─────────────────────────────────────────────────────────

def timecents_to_ms(tc: float) -> float:
    if tc <= -12000:
        return 0.0
    return 1000.0 * (2.0 ** (tc / 1200.0))


def _read_pcm(raw: bytes, a: int, b: int) -> List[int]:
    if b <= a:
        return []
    return list(struct.unpack_from(f'<{b - a}h', raw, a * 2))


def _fade_out(data: List[int], rate: float) -> None:
    """~4 ms linear fade at the end of a one-shot, in place."""
    fade = min(len(data) // 4, max(8, int(rate) * 4 // 1000))
    for i in range(fade):
        k = len(data) - fade + i
        data[k] = int(data[k] * (fade - 1 - i) / fade)


def convert_sample(raw: bytes, smp: SF2Sample, z: SF2Zone, warn):
    """Cut a zone's sample out of the PCM blob -> (data, lsa, lea, looped) or None.
    Looped samples end at the loop end; one-shots get a short fade-out."""
    total = len(raw) // 2
    start = max(0, min(smp.start + z.start_off, total))
    end = max(start, min(smp.end + z.end_off, total))
    ls = smp.loop_start + z.loop_start_off
    le = smp.loop_end + z.loop_end_off
    if end - start < 2:
        return None
    looped = z.sample_modes in (1, 3)
    if looped and not (start <= ls < le <= end and le - ls >= 2):
        warn(f"sample '{smp.name}': bad loop {ls - start}..{le - start} "
             f"(length {end - start}), using one-shot")
        looped = False
    if looped:
        return _read_pcm(raw, start, le), ls - start, le - start, True
    data = _read_pcm(raw, start, end)
    _fade_out(data, smp.sample_rate or SPU_RATE)
    return data, 0, len(data), False


def zone_root(z: SF2Zone, smp: SF2Sample) -> float:
    """Fractional key at which the zone plays its sample at the original pitch."""
    root = z.root_key if z.root_key >= 0 else smp.original_key
    if root > 127:
        root = 60
    return root - (z.coarse_tune * 100 + z.fine_tune + smp.pitch_correction) / 100.0


def eff_to_center_shift(eff: float) -> Tuple[int, int, bool]:
    """Note at which the sample plays at 44.1 kHz -> (center, shift, clamped).
    Pitch = 2^((note - center + shift/128) / 12); shift is kept a multiple of 8
    because libsnd's pitch table has 16 steps per semitone."""
    t = int(round(eff * 16))
    center = -(-t // 16)
    shift = (center * 16 - t) * 8
    if center > 127:
        return 127, 0, True
    if center < 0:
        return 0, 0, True
    return center, shift, False


def parse_size_str(val_str: str) -> int:
    val = val_str.strip().lower()
    if val.startswith('0x'):
        return int(val, 16)
    for suf, mul in (('kb', 1024), ('k', 1024), ('mb', 1 << 20), ('m', 1 << 20), ('b', 1)):
        if val.endswith(suf):
            return int(val[:-len(suf)]) * mul
    return int(val)

# ── SPU ADSR conversion ─────────────────────────────────────────────
#
# SPU envelope (psx-spx): each rate is (shift, step).  Per update, which happens
# every 1 << max(0, shift-11) samples, step << max(0, 11-shift) is added; for
# exponential decrease it's first multiplied by level >> 15.  Increase steps are
# +7..+4, decrease -8..-5.  Times are found by simulating that.

LEVEL_MAX = 0x7FFF
LEVEL_60DB = 33             # -60 dB
_traj = {}


def _exp_traj(m: int) -> List[int]:
    """Negated levels of an exponential decrease with step magnitude m, one per
    update, from LEVEL_MAX down to 0 (ascending, for bisect)."""
    t = _traj.get(m)
    if t is None:
        lv = LEVEL_MAX
        t = [-lv]
        while lv > 0:
            lv = max(0, lv + ((-m * lv) >> 15))
            t.append(-lv)
        _traj[m] = t
    return t


def _dec_samples(shift: int, step: int, target: int, start: int = LEVEL_MAX) -> int:
    """Samples for an exponential decrease (step 0..3 = -8..-5) from start to target."""
    m = (8 - step) << max(0, 11 - shift)
    t = _exp_traj(m)
    return max(0, bisect_left(t, -target) - bisect_left(t, -start)) << max(0, shift - 11)


def _attack_samples(rate: int) -> int:
    """Samples for a linear attack 0 -> max at 7-bit rate (shift << 2 | step)."""
    shift, step = rate >> 2, rate & 3
    inc = (7 - step) << max(0, 11 - shift)
    return -(-LEVEL_MAX // inc) << max(0, shift - 11)


_tables = {}


def _closest(name: str, cands, target_samples: float) -> int:
    """cands: callable building [(value, samples)], cached under name."""
    if name not in _tables:
        _tables[name] = cands()
    tgt = max(0.0, target_samples)
    return min(_tables[name], key=lambda c: abs(math.log((c[1] + 1.0) / (tgt + 1.0))))[0]


SL0_DB = -20.0 * math.log10(1 / 16.0)  # lowest SPU sustain level, ~24.1 dB down


def zone_adsr(z: SF2Zone) -> Tuple[int, int]:
    """SF2 volume envelope -> (adsr1, adsr2) SPU register values."""
    ms = SPU_RATE / 1000.0
    ar = _closest('attack', lambda: [(r, _attack_samples(r)) for r in range(127)],
                  timecents_to_ms(z.vol_attack) * ms)
    dec = timecents_to_ms(z.vol_decay) * ms         # SF2: samples per 100 dB
    rel = timecents_to_ms(z.vol_release) * ms
    sus_db = max(0.0, min(100.0, z.vol_sustain / 10.0))
    sus_mode, sus_rate = 0, 0x7F                     # linear decrease, slowest = hold
    if sus_db < 0.5:
        sl, dr = 15, 15                              # no decay phase
    elif sus_db < SL0_DB:
        sl = max(0, min(14, int(round(16 * 10 ** (-sus_db / 20.0))) - 1))
        actual = -20.0 * math.log10((sl + 1) / 16.0)
        level = (sl + 1) * 0x800 - 1
        dr = _closest(f'decay{sl}', lambda: [(d, _dec_samples(d, 0, level)) for d in range(16)],
                      dec * actual / 100.0)
    else:
        # Below the lowest sustain level: skip decay, decay in the sustain phase.
        sl, dr = 15, 15
        sus_mode = 1
        sus_rate = _closest('sustain', lambda: [(r, _dec_samples(r >> 2, r & 3, LEVEL_60DB))
                                                 for r in range(128)], dec * 0.6)
    rr = _closest('release', lambda: [(r, _dec_samples(r, 0, LEVEL_60DB)) for r in range(32)],
                  rel * 0.6)
    adsr1 = (ar << 8) | (dr << 4) | sl              # bit 15 = 0: linear attack
    adsr2 = (sus_mode << 15) | (1 << 14) | (sus_rate << 6) | (1 << 5) | rr
    return adsr1, adsr2

# ── Resampling (windowed sinc) ──────────────────────────────────────

RESAMPLE_PHASES = 512
RESAMPLE_ZEROS = 8
_kernel_cache = {}


def _kernel_table(r: float):
    """Low-pass interpolation kernel for ratio r (<= 1): (M, rows), rows[p] =
    2*M weights for input offsets -M+1..M at fraction p / RESAMPLE_PHASES.
    Blackman-windowed sinc a little under the new Nyquist, unity gain rows."""
    key = round(min(r, 1.0) * 4096)
    if key in _kernel_cache:
        return _kernel_cache[key]
    fc = 0.5 * (key / 4096.0) * 0.92
    half = RESAMPLE_ZEROS / (2.0 * fc)
    m = int(math.ceil(half))
    rows = []
    for p in range(RESAMPLE_PHASES + 1):
        frac = p / RESAMPLE_PHASES
        row = []
        for d in range(-m + 1, m + 1):
            t = frac - d
            if abs(t) >= half:
                row.append(0.0)
                continue
            s = 2.0 * fc if t == 0 else math.sin(2.0 * math.pi * fc * t) / (math.pi * t)
            u = t / half
            row.append(s * (0.42 + 0.5 * math.cos(math.pi * u) + 0.08 * math.cos(2.0 * math.pi * u)))
        g = sum(row)
        rows.append([w / g for w in row])
    _kernel_cache[key] = (m, rows)
    return m, rows


def _resample_run(src, src_off, x0, step, count, m, rows) -> List[int]:
    """Samples at input positions x0 + k*step; input i is src[src_off + i]."""
    out = []
    mul = operator.mul
    for k in range(count):
        x = x0 + k * step
        j = math.floor(x)
        b = j + src_off - m + 1
        v = int(round(sum(map(mul, src[b:b + 2 * m], rows[int((x - j) * RESAMPLE_PHASES + 0.5)]))))
        out.append(32767 if v > 32767 else -32768 if v < -32768 else v)
    return out

# ── Samples / tones / programs ──────────────────────────────────────

@dataclass
class VagEntry:
    """One sample in the VAB body."""
    name: str
    data: List[int]        # 16-bit PCM
    rate: float
    lsa: int
    lea: int               # loop end (exclusive); len(data) for one-shots
    looped: bool
    src: int = -1          # source sample index (-1 = mixed stereo pair)
    partner: int = -1      # linked stereo channel's sample index
    max_eff: float = -1e9  # highest tone eff using it (bounds resampling, center <= 127)
    pitch_cap: float = float('inf')  # highest rate at which all tones reach their top keys
    out_rate: float = 0.0
    adpcm: bytes = b''

    def size16(self) -> int:
        return 2 * len(self.data)


@dataclass
class Plan:
    r: float               # out_rate / rate
    lsa: int               # output loop start (before the alignment pad)
    loop_len: int          # one output loop period
    unroll: int
    n: int                 # output points (excluding pad)
    pad: int
    nbytes: int


def plan_entry(e: VagEntry, cap: float) -> Plan:
    """How e is stored for a sample-rate cap: resample ratio, loop alignment and
    unrolling, and the resulting ADPCM size (incl. leading/terminating blocks)."""
    target = min(cap, e.pitch_cap)
    r0 = min(1.0, target / e.rate)
    if e.max_eff > -1e8:
        # center <= 127: may even need upsampling (high root at a low rate)
        r0 = max(r0, 1.0 / 2.0 ** ((MAX_EFF_NOTE - e.max_eff) / 12.0))
    resample = abs(r0 - 1.0) > 1e-4
    if e.looped:
        L = e.lea - e.lsa
        lr = max(MIN_RESAMPLED_LOOP, int(math.ceil(L * r0 - 1e-9))) if resample else L
        if r0 < 1.0:
            lr = min(L, lr)
        k = 1
        if lr % BLOCK:
            k = BLOCK // math.gcd(lr, BLOCK)
            if lr * k > UNROLL_MAX:
                k = 1
                lr = -(-lr // BLOCK) * BLOCK
        r = lr / L
        lsa = int(round(e.lsa * r))
        pad = (-lsa) % BLOCK
        n = lsa + lr * k
        return Plan(r, lsa, lr, k, n, pad, 16 * (1 + (pad + n) // BLOCK))
    n0 = e.lea
    n = max(2, int(n0 * r0)) if resample else n0
    return Plan(n / n0, 0, 0, 1, n, 0, 16 * (2 + -(-n // BLOCK)))


def render_entry(e: VagEntry, p: Plan) -> Tuple[List[int], Optional[int]]:
    """Resample / align / unroll -> (pcm, loop start or None)."""
    if e.looped:
        L = e.lea - e.lsa
        if p.r != 1.0:
            m, rows = _kernel_table(p.r)
            step = L / p.loop_len
            pad = m + int(math.ceil(step)) + 2
            loop = e.data[e.lsa:e.lea]
            src = [0] * pad + e.data[:e.lea] + [loop[i % L] for i in range(pad)]
            attack = _resample_run(src, pad, e.lsa - p.lsa * step, step, p.lsa, m, rows)
            ext = [loop[(i - pad) % L] for i in range(L + 2 * pad)]
            body = _resample_run(ext, pad, 0.0, step, p.loop_len, m, rows)
        else:
            attack, body = e.data[:e.lsa], e.data[e.lsa:e.lea]
        return [0] * p.pad + attack + body * p.unroll, p.pad + len(attack)
    if p.r != 1.0:
        m, rows = _kernel_table(p.r)
        step = e.lea / p.n
        pad = m + int(math.ceil(step)) + 2
        src = [0] * pad + e.data[:e.lea] + [0] * pad
        return _resample_run(src, pad, 0.0, step, p.n, m, rows), None
    return e.data[:e.lea], None


@dataclass
class VabTone:
    key_lo: int
    key_hi: int
    entry: VagEntry
    eff: float             # note at which the source sample plays at 44.1 kHz
    root: float            # key at which it plays at its original pitch
    atten: float           # cB (already scaled)
    pan: float             # SF2 units
    adsr1: int
    adsr2: int
    locked: bool = False   # drum tone: never merged into another key range
    out_eff: float = 0.0


@dataclass
class VabProgram:
    name: str
    tones: List[VabTone] = field(default_factory=list)

    @property
    def layers(self):
        return self.tones

# ── ADPCM encoder ───────────────────────────────────────────────────

ADPCM_FILTERS = ((0, 0), (60, 0), (115, -52), (98, -55), (122, -60))


def _shift_for(peak: int) -> int:
    """Largest shift (finest step) whose 4-bit range covers peak."""
    for shift in range(12, -1, -1):
        q = 1 << (12 - shift)
        if peak <= 7 * q + (q >> 1):
            return shift
    return 0


def _enc_try(x, h1, h2, a, b, shift):
    q_bits = 12 - shift
    half = (1 << q_bits) >> 1
    err = 0
    nib = []
    for s in x:
        p = (h1 * a + h2 * b + 32) >> 6
        n = (s - p + half) >> q_bits
        if n > 7:
            n = 7
        elif n < -8:
            n = -8
        y = p + (n << q_bits)
        if y > 32767:
            y = 32767
        elif y < -32768:
            y = -32768
        d = s - y
        err += d * d
        nib.append(n & 15)
        h2 = h1
        h1 = y
    return err, nib, h1, h2


def _enc_block(x, h1, h2):
    """Encode 28 samples -> (16-byte block with flags 0, h1, h2)."""
    peaks = []
    for f, (a, b) in enumerate(ADPCM_FILTERS):
        p1, p2, mx = h1, h2, 0
        for s in x:
            d = s - ((p1 * a + p2 * b + 32) >> 6)
            if d < 0:
                d = -d
            if d > mx:
                mx = d
            p2 = p1
            p1 = s
        peaks.append((mx, f))
    peaks.sort()
    # Best two filters by open-loop peak residual, each at the shift that covers
    # the peak and one finer (clipping the peak is often the smaller error).
    best = None
    for mx, f in peaks[:2]:
        s = _shift_for(mx)
        for sh in (s, s + 1) if s < 12 else (s,):
            a, b = ADPCM_FILTERS[f]
            r = _enc_try(x, h1, h2, a, b, sh)
            if best is None or r[0] < best[0][0]:
                best = (r, f, sh)
    (err, nib, nh1, nh2), f, sh = best
    blk = bytearray(16)
    blk[0] = (f << 4) | sh
    for i in range(14):
        blk[2 + i] = nib[2 * i] | (nib[2 * i + 1] << 4)
    return blk, nh1, nh2


def encode_vag(pcm: List[int], loop_start: Optional[int]) -> bytes:
    """16-bit PCM -> SPU ADPCM VAG body: a silent leading block, the data, and
    either loop flags (loop start block aligned) or an end block + silent
    terminator.  The loop is re-encoded until the history at its end equals the
    history its first block was encoded with, so repeats decode seamlessly."""
    nb = -(-len(pcm) // BLOCK)
    pcm = pcm + [0] * (nb * BLOCK - len(pcm))
    blocks = [None] * nb
    h1 = h2 = 0
    lb = loop_start // BLOCK if loop_start is not None else nb
    for b in range(lb):
        blocks[b], h1, h2 = _enc_block(pcm[b * BLOCK:(b + 1) * BLOCK], h1, h2)
    if loop_start is not None:
        hist, prev = (h1, h2), None
        for _ in range(4):
            c1, c2 = hist
            bounds = []
            for b in range(lb, nb):
                if prev is not None and (c1, c2) == prev[b - lb]:
                    bounds.extend(prev[b - lb:])  # same history: rest is unchanged
                    break
                bounds.append((c1, c2))
                blocks[b], c1, c2 = _enc_block(pcm[b * BLOCK:(b + 1) * BLOCK], c1, c2)
            else:
                bounds.append((c1, c2))
            if bounds[-1] == hist:
                break
            prev, hist = bounds, bounds[-1]
        for b in range(nb):
            blocks[b][1] = 0x02
        blocks[lb][1] |= 0x04
        blocks[-1][1] |= 0x01
        return bytes(16) + b''.join(bytes(b) for b in blocks)
    if blocks:
        blocks[-1][1] = 0x01
    return bytes(16) + b''.join(bytes(b) for b in blocks) + b'\x00\x07' + bytes(14)

# ── Tone reduction ──────────────────────────────────────────────────

def _layers_of(tones: List[VabTone]) -> List[List[VabTone]]:
    """Split tones into non-overlapping key-split layers (greedy interval coloring)."""
    layers = []
    for t in sorted(tones, key=lambda t: (t.key_lo, t.key_hi)):
        for lay in layers:
            if lay[-1].key_hi < t.key_lo:
                lay.append(t)
                break
        else:
            layers.append([t])
    return layers


def _best_merge(tones, t, exclude=None, layers=None):
    """Cheapest neighbour in t's layer to take over t's keys:
    (cost, neighbour, lo, hi) or None.  Cost = semitones the neighbour's sample
    gets stretched from its root."""
    for lay in layers or _layers_of(tones):
        if t not in lay:
            continue
        i = lay.index(t)
        best = None
        for j in (i - 1, i + 1):
            if not 0 <= j < len(lay) or lay[j].locked or lay[j].entry is exclude:
                continue
            nb = lay[j]
            lo, hi = min(nb.key_lo, t.key_lo), max(nb.key_hi, t.key_hi)
            cost = max(abs(lo - nb.root), abs(hi - nb.root))
            if best is None or cost < best[0]:
                best = (cost, nb, lo, hi)
        return best
    return None


_mix_cache = {}  # (id(left entry), id(right entry)) -> mixed entry; reset per conversion


def _stereo_partners(a: VabTone, b: VabTone) -> bool:
    """a, b are the two channels of one stereo zone: same key range, and either
    linked samples (SF2 sample link / DLS stereo wave) or opposite hard pans."""
    if a.key_lo != b.key_lo or a.key_hi != b.key_hi or a.entry is b.entry:
        return False
    ea, eb = a.entry, b.entry
    if ea.src >= 0 and (ea.partner == eb.src or eb.partner == ea.src):
        return True
    return a.pan * b.pan < 0 and abs(a.pan) > 100 and abs(b.pan) > 100


def _mix_entries(a: VagEntry, b: VagEntry) -> Optional[VagEntry]:
    """Mono mix (L+R)/2 of two channels, or None if they don't line up
    (different rate, loop type or loop points)."""
    key = (id(a), id(b))
    if key in _mix_cache:
        return _mix_cache[key]
    m = None
    if a.rate == b.rate and a.looped == b.looped and (
            not a.looped or (a.lsa == b.lsa and a.lea == b.lea)):
        n = a.lea if a.looped else max(len(a.data), len(b.data))
        da = a.data[:n] + [0] * (n - len(a.data))
        db = b.data[:n] + [0] * (n - len(b.data))
        m = VagEntry(f"{a.name}+{b.name}", [(x + y) >> 1 for x, y in zip(da, db)], a.rate,
                     a.lsa if a.looped else 0, n, a.looped)
    _mix_cache[key] = m
    return m


def merge_stereo(tones: List[VabTone], limit: int = 0) -> int:
    """Turn stereo pairs into one centred mono tone (a mix of both channels, or
    the left one if they can't be mixed) until at most `limit` tones are left
    (0 = all pairs).  Returns the number of pairs merged."""
    n = 0
    i = 0
    while i < len(tones) and (not limit or len(tones) > limit):
        a = tones[i]
        j = next((j for j in range(i + 1, len(tones)) if _stereo_partners(a, tones[j])), None)
        if j is None:
            i += 1
            continue
        b = tones.pop(j)
        left, right = (a, b) if a.pan <= b.pan else (b, a)
        mixed = _mix_entries(left.entry, right.entry)
        tones[i] = VabTone(left.key_lo, left.key_hi, mixed or left.entry, left.eff, left.root,
                           min(left.atten, right.atten), 0.0, left.adsr1, left.adsr2,
                           left.locked)
        n += 1
        i += 1
    return n


def reduce_program(prog: VabProgram, warn, limit: int = MAX_TONES) -> None:
    tones = prog.tones
    if len(tones) <= limit:
        return
    before = len(tones)
    n_st = merge_stereo(tones, limit)
    n_merged = n_dropped = 0
    while len(tones) > limit:
        layers = _layers_of(tones)
        best = None
        for t in tones:
            if t.locked:
                continue
            m = _best_merge(tones, t, layers=layers)
            if m and (best is None or m[0] < best[0][0]):
                best = (m, t)
        if best is None:
            # Only single-tone layers (or drum tones) left: drop the quietest
            # tone whose keys are also covered by another tone, else the quietest.
            def covered(t):
                return all(any(o is not t and o.key_lo <= k <= o.key_hi for o in tones)
                           for k in range(t.key_lo, t.key_hi + 1))
            cands = [t for t in tones if covered(t)] or tones
            tones.remove(max(cands, key=lambda t: t.atten))
            n_dropped += 1
            continue
        (cost, nb, lo, hi), t = best
        nb.key_lo, nb.key_hi = lo, hi
        tones.remove(t)
        n_merged += 1
    parts = []
    if n_st:
        parts.append(f"{n_st} stereo pairs made mono")
    if n_merged:
        parts.append(f"{n_merged} key splits merged")
    if n_dropped:
        parts.append(f"{n_dropped} layers dropped")
    warn(f"'{prog.name}': {before} zones -> {len(tones)} tones (VAB max {limit}): "
         + ", ".join(parts))


def limit_vags(programs: List[VabProgram], warn, max_vags: int = MAX_VAGS) -> None:
    """Merge away samples until at most max_vags are used: each step removes the
    sample whose tones can be taken over by neighbouring key splits with the
    least pitch stretch."""
    removed, dropped = 0, 0
    while True:
        users = {}
        for p in programs:
            for t in p.tones:
                users.setdefault(id(t.entry), (t.entry, []))[1].append((p, t))
        if len(users) <= max_vags:
            break
        layer_cache = {id(p): _layers_of(p.tones) for p in programs}
        best = None
        for e, lst in users.values():
            moves, cost = [], 0
            for p, t in lst:
                m = None if t.locked else _best_merge(p.tones, t, e, layer_cache[id(p)])
                if m is None:
                    moves = None
                    break
                cost = max(cost, m[0])
                moves.append((p, t, m))
            key = (0, cost, -e.size16()) if moves is not None else (1, len(lst), -e.size16())
            if best is None or key < best[0]:
                best = (key, e, lst, moves)
        _, e, lst, moves = best
        if moves is not None:
            for p, t, (_, nb, lo, hi) in moves:
                nb.key_lo, nb.key_hi = min(nb.key_lo, lo), max(nb.key_hi, hi)
                p.tones.remove(t)
            removed += 1
        else:
            for p, t in lst:
                p.tones.remove(t)
            dropped += 1
    if removed or dropped:
        warn(f"more than {max_vags} samples: {removed} merged into neighbouring key splits"
             + (f", {dropped} dropped with their tones" if dropped else ""))

# ── Compaction ──────────────────────────────────────────────────────

@dataclass
class CompactOptions:
    budget: int
    min_rate: int = 8000
    keep_small: int = 2048


def _vb_size(nbytes: int) -> int:
    return -(-nbytes // VB_ALIGN) * VB_ALIGN


def choose_cap(entries: List[VagEntry], opts: Optional[CompactOptions]) -> Tuple[dict, bool]:
    """Pick the highest sample-rate cap at which the VB fits opts.budget.
    Returns ({id(entry): Plan}, fits)."""
    inf = float('inf')
    if opts is None:
        return {id(e): plan_entry(e, inf) for e in entries}, True
    small = [e for e in entries if e.size16() < opts.keep_small]
    big = [e for e in entries if e.size16() >= opts.keep_small]
    fixed = sum(plan_entry(e, inf).nbytes for e in small)

    def total(cap):
        return _vb_size(fixed + sum(plan_entry(e, cap).nbytes for e in big))

    orig = total(inf)
    top = int(math.ceil(max((e.rate for e in big), default=0)))
    cap = inf
    if orig > opts.budget and big:
        lo, hi = max(1, min(opts.min_rate, top)), top
        if total(lo) > opts.budget:
            cap = lo
        else:
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if total(mid) <= opts.budget:
                    lo = mid
                else:
                    hi = mid
            cap = lo
    plans = {id(e): plan_entry(e, inf) for e in small}
    plans.update({id(e): plan_entry(e, cap) for e in big})
    new = total(cap)
    if cap == inf:
        print(f"[compact] VB {orig} bytes already fits ({opts.budget} available); nothing changed")
    else:
        n_rs = sum(1 for e in big if plans[id(e)].r < plan_entry(e, inf).r - 1e-9)
        print(f"[compact] VB {orig} -> {new} bytes ({opts.budget} available): {n_rs} of "
              f"{len(entries)} samples resampled to <= {cap} Hz"
              + (f", {len(small)} under {opts.keep_small} bytes left alone" if small else ""))
    return plans, new <= opts.budget

# ── Conversion ──────────────────────────────────────────────────────

def sf2_to_vab(path: str, bank: int = 0, gm: bool = False, drum_kit: int = 0,
               all_banks: bool = False, mono: bool = False, seq: Optional[dict] = None,
               atten_scale: Optional[float] = None, reverb: bool = True,
               bend_range: int = 2, pitch_fix: bool = True, min_rate: int = 8000,
               compact: Optional[CompactOptions] = None, budget: int = VB_MAX_SIZE):
    """Convert an SF2/DLS -> (vh bytes, vb bytes, [(program, name, tones)], voice_map).
    voice_map (all_banks only): [(sf2_bank, program, vab_program, (lo, hi) | None, name)]."""
    warnings = []

    def warn(msg):
        if msg not in warnings:
            warnings.append(msg)

    samples, presets, raw, kind = load_soundfont(path, warn)
    if atten_scale is None:
        atten_scale = 1.0 if kind == 'dls' else 0.4
    print(f"[{kind}] {len(samples)} samples, {len(presets)} "
          f"{'instruments' if kind == 'dls' else 'presets'}")

    cache = {}
    _mix_cache.clear()
    n_mono = [0]

    def build_tones(zones: List[SF2Zone], locked=False) -> List[VabTone]:
        vz = [z for z in zones if z.vel_lo <= REPRESENTATIVE_VELOCITY <= z.vel_hi] or zones
        tones = []
        for z in vz:
            if z.sample_id >= len(samples):
                continue
            smp = samples[z.sample_id]
            if smp.sample_type & 0x8000:
                warn(f"sample '{smp.name}' is a ROM sample, skipped")
                continue
            key = (z.sample_id, z.start_off, z.end_off, z.loop_start_off, z.loop_end_off,
                   z.sample_modes in (1, 3))
            if key not in cache:
                conv = convert_sample(raw, smp, z, warn)
                if conv is None:
                    warn(f"sample '{smp.name}' is empty or lies outside the sample data, "
                         f"skipped")
                cache[key] = None if conv is None else VagEntry(
                    smp.name, conv[0], smp.sample_rate or SPU_RATE, conv[1], conv[2], conv[3],
                    src=z.sample_id, partner=smp.link)
            entry = cache[key]
            if entry is None:
                continue
            root = zone_root(z, smp)
            eff = root + 12.0 * math.log2(SPU_RATE / entry.rate)
            a1, a2 = zone_adsr(z)
            tones.append(VabTone(z.key_lo, z.key_hi, entry, eff, root,
                                 max(0.0, z.attenuation * atten_scale), z.pan, a1, a2, locked))
        if mono:
            n_mono[0] += merge_stereo(tones)
        return tones

    def build_program(name, zones):
        prog = VabProgram(name, build_tones(zones))
        reduce_program(prog, warn)
        return prog

    programs: List[Optional[VabProgram]] = [None] * MAX_PROGRAMS
    voice_map = []
    if gm:
        _layout_gm(presets, programs, build_program, build_tones, drum_kit, warn)
    elif all_banks:
        voice_map = _layout_all_banks(presets, programs, build_program, build_tones, warn)
    elif seq:
        _layout_seq(presets, programs, build_tones, seq, warn)
    else:
        for name, prog, pbank, zones in sorted(presets, key=lambda p: (p[2], p[1])):
            if pbank != bank or prog > 127 or programs[prog] is not None:
                continue
            p = build_program(name, zones)
            if p.tones:
                programs[prog] = p
            else:
                warn(f"preset '{name}' (prog {prog}) has no usable zones")
        if not any(programs):
            raise ValueError(f"No usable instruments found in bank {bank}")
        print(f"  bank {bank}: {sum(1 for p in programs if p)} programs")

    unique = list({id(p): p for p in programs if p}.values())
    if mono:
        print(f"[mono] {n_mono[0]} stereo zones mixed down to mono")
    limit_vags(unique, warn)

    # Per-sample bounds from the final key ranges.
    entries = list({id(t.entry): t.entry for p in unique for t in p.tones}.values())
    for p in unique:
        for t in p.tones:
            e = t.entry
            e.max_eff = max(e.max_eff, t.eff)
            room = t.eff + SPU_MAX_UP - min(t.key_hi + bend_range, PITCH_FIX_MAX_KEY)
            if pitch_fix and room < 0:
                # Rate at which the tone's lowest key still plays at >= 44.1 kHz: going
                # down to it loses nothing audible, so it may go below --min-rate.
                lossless = SPU_RATE / 2.0 ** ((t.key_lo - t.root) / 12.0)
                need = e.rate * 2.0 ** (room / 12.0)
                e.pitch_cap = min(e.pitch_cap, max(need, min(lossless, min_rate)))
            if t.eff < 0:  # center >= 0
                e.pitch_cap = min(e.pitch_cap, e.rate * 2.0 ** (t.eff / 12.0))

    plans, fits = choose_cap(entries, compact)
    n_fix = sum(1 for e in entries if plans[id(e)].r < 1.0 and
                plans[id(e)].r * e.rate <= e.pitch_cap * 1.0001)
    n_up = sum(1 for e in entries if e.max_eff > MAX_EFF_NOTE)
    if n_fix:
        print(f"[pitch] {n_fix} samples resampled down so their key ranges stay under the "
              f"SPU's 4x pitch limit")
    if n_up:
        print(f"[pitch] {n_up} samples resampled up so their center note fits 0..127")
    if compact and not fits:
        warn(f"--compact: VB doesn't fit in {compact.budget} bytes even at "
             f"{compact.min_rate} Hz; lower --min-rate / --keep-small or free SPU RAM")

    tty = sys.stdout.isatty()
    for i, e in enumerate(entries):
        pl = plans[id(e)]
        pcm, loop = render_entry(e, pl)
        e.out_rate = e.rate * pl.r
        e.adpcm = encode_vag(pcm, loop)
        assert len(e.adpcm) == pl.nbytes, (e.name, len(e.adpcm), pl.nbytes)
        if tty:
            print(f"\r[adpcm] {i + 1}/{len(entries)} samples", end='', flush=True)
    print(f"\r[adpcm] {len(entries)} samples, "
          f"{sum(len(e.adpcm) for e in entries)} bytes")

    vh, vb, info = pack_vab(programs, entries, reverb, bend_range, warn)
    if not compact and _vb_size(len(vb)) > budget:
        warn(f"VB is {len(vb)} bytes, more than the {budget} bytes of SPU RAM available; "
             f"use --compact")
    for w in warnings:
        print(f"[warn] {w}")
    return vh, vb, info, voice_map


def _pick(cands, build_program, warn):
    """{prog: [(name, bank, zones), ...]} -> {prog: (name, bank, program)}: the first
    usable preset per program."""
    chosen = {}
    for prog in sorted(cands):
        for name, bank, zones in cands[prog]:
            p = build_program(name, zones)
            if p.tones:
                chosen[prog] = (name, bank, p)
                break
            warn(f"preset '{name}' (bank {bank} prog {prog}) has no usable zones")
    return chosen


def _layout_gm(presets, programs, build_program, build_tones, drum_kit, warn):
    """Fixed GM layout: programs 0..123 melodic, 124..127 the drum kit."""
    cands = {}
    for name, prog, bank, zones in sorted(presets, key=lambda p: (p[2] != 0, p[2], p[1])):
        if bank < 128 and prog < GM_DRUM_PROG:
            cands.setdefault(prog, []).append((name, bank, zones))
    chosen = _pick(cands, build_program, warn)
    if not chosen:
        raise ValueError("No usable melodic presets found")
    missing = []
    for prog in range(GM_DRUM_PROG):
        if prog in chosen:
            name, bank, p = chosen[prog]
            if bank != 0:
                p.name = f"{name} (from bank {bank})"
            programs[prog] = p
        else:
            fam = [q for q in range(prog & ~7, (prog & ~7) + 8) if q in chosen]
            src = min(fam, key=lambda q: abs(q - prog)) if fam else min(chosen)
            programs[prog] = chosen[src][2]
            missing.append(prog)
    dropped = sorted({p for _, p, b, _ in presets if b < 128 and GM_DRUM_PROG <= p <= 127})
    print(f"  melodic: {len(chosen)} programs"
          + (f", {len(missing)} missing (aliased)" if missing else "")
          + (f"; GM programs {dropped} give way to the drum kit" if dropped else ""))

    kits = sorted((p for p in presets if p[2] >= 128 and p[1] <= 127),
                  key=lambda p: (p[2] != 128, p[1] != drum_kit, p[1] != 0, p[2], p[1]))
    kit_tones, kit_name, kit_prog = [], None, None
    for name, prog, bank, zones in kits:
        kit_tones = build_tones(zones, locked=True)
        if kit_tones:
            kit_name, kit_prog = f"{name} (bank {bank} prog {prog})", prog
            break
    if not kit_tones:
        warn("no drum kit (bank 128) found; drum programs 124-127 are empty")
        return
    if kit_prog != drum_kit:
        warn(f"drum kit {drum_kit} not found, using {kit_name}")
    lo_all = GM_DRUM_FIRST_NOTE
    hi_all = GM_DRUM_FIRST_NOTE + GM_DRUM_NOTES_PER_PROG * (MAX_PROGRAMS - GM_DRUM_PROG) - 1
    outside = sorted({k for t in kit_tones for k in range(t.key_lo, t.key_hi + 1)
                      if not lo_all <= k <= hi_all})
    if outside:
        warn(f"drum notes outside {lo_all}-{hi_all} can't be mapped: {outside[:12]}"
             + (" ..." if len(outside) > 12 else ""))
    for c in range(MAX_PROGRAMS - GM_DRUM_PROG):
        lo = GM_DRUM_FIRST_NOTE + c * GM_DRUM_NOTES_PER_PROG
        hi = lo + GM_DRUM_NOTES_PER_PROG - 1
        tones = [VabTone(max(lo, t.key_lo), min(hi, t.key_hi), t.entry, t.eff, t.root, t.atten,
                         t.pan, t.adsr1, t.adsr2, True)
                 for t in kit_tones if t.key_lo <= hi and t.key_hi >= lo]
        if tones:
            p = VabProgram(f"[drums {lo}-{hi}] {kit_name}", tones)
            reduce_program(p, warn)
            programs[GM_DRUM_PROG + c] = p
    print(f"  drums: {kit_name} -> programs {GM_DRUM_PROG}-{MAX_PROGRAMS - 1} "
          f"(notes {lo_all}-{hi_all}, 16 per program)")


def _clip_tones(tones: List[VabTone], lo: int, hi: int) -> List[VabTone]:
    return [VabTone(max(lo, t.key_lo), min(hi, t.key_hi), t.entry, t.eff, t.root, t.atten,
                    t.pan, t.adsr1, t.adsr2, t.locked)
            for t in tones if t.key_lo <= hi and t.key_hi >= lo]


def split_kit(tones: List[VabTone]) -> List[Tuple[int, int]]:
    """Fewest consecutive key ranges covering 0-127 with at most 16 tones each
    (greedy: extend each range as far as possible)."""
    ranges, lo, cur = [], 0, set()
    for n in range(128):
        here = {id(t) for t in tones if t.key_lo <= n <= t.key_hi}
        if cur and len(cur | here) > MAX_TONES:
            ranges.append((lo, n - 1))
            lo, cur = n, set()
        cur |= here
    ranges.append((lo, 127))
    return ranges


def _program_key(p: VabProgram):
    """Identity of a program's sound (for storing identical presets once)."""
    return tuple(sorted((t.key_lo, t.key_hi, id(t.entry), round(t.eff, 4), round(t.atten, 2),
                         t.pan, t.adsr1, t.adsr2) for t in p.tones))


def _layout_all_banks(presets, programs, build_program, build_tones, warn):
    """Every bank in one VAB, each distinct instrument stored once, numbered from
    0 in priority order: bank 0, drum kits (bank >= 128), other banks.  Drum kits
    with more than 16 tones get several programs, split by key range.
    Returns the voice map [(sf2_bank, program, vab_program, keys, name)]."""
    def priority(p):
        return (0 if p[2] == 0 else 1 if p[2] >= 128 else 2, p[2], p[1])

    by_key = {}           # program key / kit key -> [(vab_program, keys)]
    voice_map, dropped, seen = [], [], set()
    count = 0
    for name, prog, bank, zones in sorted(presets, key=priority):
        if prog > 127 or (bank, prog) in seen:
            continue
        seen.add((bank, prog))
        if bank >= 128:
            tones = build_tones(zones, locked=True)
            if not tones:
                warn(f"preset '{name}' (bank {bank} prog {prog}) has no usable zones")
                continue
            parts = []
            for lo, hi in split_kit(tones):
                p = VabProgram(f"{name} [keys {lo}-{hi}]", _clip_tones(tones, lo, hi))
                reduce_program(p, warn)
                parts.append((p, (lo, hi)))
            key = ('kit',) + tuple((_program_key(p), keys) for p, keys in parts)
        else:
            p = build_program(name, zones)
            if not p.tones:
                warn(f"preset '{name}' (bank {bank} prog {prog}) has no usable zones")
                continue
            parts = [(p, None)]
            key = _program_key(p)
        if key not in by_key:
            if count + len(parts) > MAX_PROGRAMS:
                dropped.append(f"{bank}:{prog} '{name}'")
                continue
            by_key[key] = []
            for p, keys in parts:
                programs[count] = p
                by_key[key].append((count, keys))
                count += 1
        for idx, keys in by_key[key]:
            voice_map.append((bank, prog, idx, keys, name))
    if not count:
        raise ValueError("No usable instruments found")
    if dropped:
        warn(f"{len(dropped)} presets didn't fit in {MAX_PROGRAMS} VAB programs (fall back to "
             f"bank 0 / the default kit): " + ", ".join(dropped[:10])
             + (" ..." if len(dropped) > 10 else ""))
    n_presets = len({(b, p) for b, p, _, _, _ in voice_map})
    banks = sorted({b for b, _, _, _, _ in voice_map})
    print(f"  {n_presets} presets from banks {banks} -> {count} VAB programs")
    return voice_map


SEQ_TABLE_MAGIC = b'VMAP'  # mid2seq.py's instrument table after the end-of-track


def read_seq(path: str, warn) -> dict:
    """Read a libsnd SEQ -> {'notes': {seq_program: {notes}}, 'drum': {seq_program},
    'melodic': {seq_program}, 'table': [(bank, prog, seq_prog, keys, name)] or None}."""
    with open(path, 'rb') as f:
        d = f.read()
    if d[:4] != b'pQES':
        raise ValueError(f"{path} is not a PlayStation SEQ (no 'pQES')")
    i, st = 15, None
    prog = [0] * 16
    notes, drum, melodic = {}, set(), set()
    vab_switch = False
    try:
        while True:
            while d[i] & 0x80:  # delta time
                i += 1
            i += 1
            if d[i] & 0x80:
                st = d[i]
                i += 1
            if st is None:
                raise ValueError("data before the first status byte")
            if st == 0xFF:
                m = d[i]
                i += 1
                if m == 0x2F:
                    i += 1
                    break
                # SMF2SEQ keeps running status across metas and writes FF 7F
                # (sequencer-specific) with no data.
                if m == 0x51:
                    i += 3
                elif m != 0x7F:
                    raise ValueError(f"unknown meta event FF {m:02X}")
                continue
            hi, ch = st & 0xF0, st & 0x0F
            n = 1 if hi in (0xC0, 0xD0) else 2
            a = d[i:i + n]
            i += n
            if hi == 0xC0:
                prog[ch] = a[0]
            elif hi == 0x90 and a[1]:
                notes.setdefault(prog[ch], set()).add(a[0])
                (drum if ch == 9 else melodic).add(prog[ch])
            elif hi == 0xB0 and a[0] == 0:
                vab_switch = True
    except IndexError:
        raise ValueError(f"{path}: SEQ data ends without an end-of-track")
    if vab_switch:
        warn("the SEQ has CC0 (libsnd VAB switch) events; all its notes are put in this one VAB")
    table = None
    j = d.find(SEQ_TABLE_MAGIC, i)
    if j >= 0 and j + 8 <= len(d):
        ln = struct.unpack_from('<I', d, j + 4)[0]
        table = []
        for line in d[j + 8:j + 8 + ln].decode('latin1').splitlines():
            parts = line.split()
            if len(parts) < 3 or parts[0].startswith('#'):
                continue
            keys = None
            if len(parts) > 3 and parts[3].startswith('keys='):
                lo, _, hi = parts[3][5:].partition('-')
                keys = (int(lo), int(hi))
            table.append((int(parts[0]), int(parts[1]), int(parts[2]), keys,
                          ' '.join(parts[4 if keys else 3:])))
    return dict(notes=notes, drum=drum, melodic=melodic, table=table)


def _find_preset(presets, bank: int, prog: int, drum: bool, warn):
    """Preset for a bank/program with fallbacks: same bank, bank 0 / kit 0, any
    bank with the program, then the lowest preset of that kind."""
    pool = [p for p in presets if (p[2] >= 128) == drum and p[1] <= 127]
    if not pool:
        return None
    by = {(b, p): (n, p, b, z) for n, p, b, z in sorted(pool, key=lambda q: (q[2], q[1]))}
    want = (DRUM_BANK if drum else bank, prog)
    for key in (want, (DRUM_BANK, prog) if drum else (0, prog)):
        if key in by:
            return by[key]
    same = [v for (b, p), v in by.items() if p == prog]
    first = same[0] if same else (by.get((DRUM_BANK, 0)) if drum else by.get((0, 0))) \
        or next(iter(by.values()))
    what = f"kit {prog}" if drum else f"bank {bank} program {prog}"
    warn(f"{what} not in the soundfont, using '{first[0].strip()}' "
         f"(bank {first[2]} prog {first[1]})")
    return first


def _layout_seq(presets, programs, build_tones, seq, warn):
    """VAB for one SEQ: program numbers as the SEQ uses them, only the presets
    and zones its notes play.  With mid2seq.py's instrument table the meaning
    of each program comes from there; otherwise programs are GM programs (bank
    0) on melodic channels and drum kits on MIDI channel 10."""
    table = seq['table']
    used = seq['notes']
    if table is None:
        print("[seq] no instrument table (not from mid2seq.py): programs are read as GM, "
              "channel 10 as drum kits")
        table = []
        both = seq['drum'] & seq['melodic']
        if both:
            warn(f"programs {sorted(both)} are played on both melodic channels and channel 10; "
                 f"the melodic instrument is used (convert with mid2seq.py to keep both)")
        for p in sorted(used):
            drum = p in seq['drum'] and p not in seq['melodic']
            table.append((DRUM_BANK if drum else 0, p, p, None, ''))
    else:
        print(f"[seq] instrument table: {len(table)} programs")
    for bank, prog, num, keys, _ in table:
        notes = used.get(num)
        if not notes or not 0 <= num < MAX_PROGRAMS:
            continue
        if keys:
            notes = {n for n in notes if keys[0] <= n <= keys[1]}
        drum = bank >= DRUM_BANK
        pr = _find_preset(presets, bank, prog, drum, warn)
        if pr is None:
            warn(f"SEQ program {num}: no {'drum kit' if drum else 'melodic preset'} in the "
                 f"soundfont")
            continue
        name, pprog, pbank, zones = pr
        tones = []
        for t in build_tones(zones, locked=drum):
            hit = sorted(n for n in notes if t.key_lo <= n <= t.key_hi)
            if hit:  # clip to the notes the SEQ plays
                t.key_lo, t.key_hi = hit[0], hit[-1]
                tones.append(t)
        if not tones:
            warn(f"SEQ program {num} ('{name}'): no zone covers its notes {sorted(notes)[:8]}")
            continue
        label = f"{name.strip()} (bank {pbank} prog {pprog})"
        if keys:
            label += f" [keys {keys[0]}-{keys[1]}]"
        p = VabProgram(label, tones)
        reduce_program(p, warn)
        programs[num] = p
    n = sum(1 for p in programs if p)
    if not n:
        raise ValueError("nothing in the SEQ can be played with this soundfont")
    print(f"  {n} programs for {len(used)} SEQ programs with notes")


def write_voice_map(path: str, voice_map) -> None:
    """Text voice map: sf2ton's .vmap format plus an optional keys= range for
    drum kits split over several programs."""
    with open(path, 'w', encoding='ascii', errors='replace', newline='\n') as f:
        f.write("# sf2vab voice map: <sf2_bank> <program> <vab_program> [keys=<lo>-<hi>] <name>\n")
        f.write("# Bank 128 = drum kits (MIDI channel 10). Kit lines have keys=: a kit with more\n")
        f.write("# than 16 tones is split over several VAB programs; note n plays with the line\n")
        f.write("# whose range holds n. Melodic lines have no keys= field (all keys).\n")
        for bank, prog, idx, keys, name in sorted(voice_map, key=lambda v: (v[0], v[1],
                                                                             v[3] or (0, 0))):
            rng = f" keys={keys[0]}-{keys[1]}" if keys else ""
            f.write(f"{bank} {prog} {idx}{rng} {name.strip()}\n")


def pack_vab(programs, entries, reverb, bend_range, warn):
    """Build the .VH and .VB.  Aliased programs get their own copy of the tones."""
    vag_id = {}
    vb = bytearray()
    sizes = [0]
    for p in programs:
        if not p:
            continue
        for t in sorted(p.tones, key=lambda t: (t.key_lo, t.key_hi)):
            if id(t.entry) not in vag_id:
                vag_id[id(t.entry)] = len(sizes)
                sizes.append(len(t.entry.adpcm) // 8)
                vb += t.entry.adpcm
    if len(sizes) - 1 > MAX_VAGS:
        raise ValueError(f"{len(sizes) - 1} VAGs exceed the VAB limit of {MAX_VAGS}")

    prog_tab = bytearray()
    tone_tab = bytearray()
    info = []
    ps = ts = 0
    clamped, short = [], []
    empty_tone = struct.pack('<14xBB4x4xhhhh', 0xB1, 0xB2, 0xC0, 0xC1, 0xC2, 0xC3)
    assert len(empty_tone) == 32
    for num in range(MAX_PROGRAMS):
        p = programs[num]
        tones = sorted(p.tones, key=lambda t: (t.key_lo, t.key_hi)) if p else []
        if not tones:
            prog_tab += struct.pack('<BBBBBbhII', 0, 0x7F, 0x14, 0xFF, 0x40, 0, 0,
                                    0xFFFFFFFF, 0xFFFFFFFF)
            continue
        ps += 1
        ts += len(tones)
        prog_tab += struct.pack('<BBBBBbhII', len(tones), 0x7F, 0x14, 0xFF, 0x40, 0, 0,
                                0xFFFFFFFF, 0xFFFFFFFF)
        for t in tones:
            e = t.entry
            eff = t.eff + 12.0 * math.log2(e.rate / e.out_rate)
            center, shift, clamp = eff_to_center_shift(eff)
            if clamp:
                clamped.append(p.name)
            top = min(t.key_hi, PITCH_FIX_MAX_KEY)
            reach = int(math.floor(eff + SPU_MAX_UP - bend_range + 0.01))
            if top > reach:
                short.append((reach, p.name))
            vol = int(round(127 * 10 ** (-t.atten / 400.0)))
            pan = int(round(64 + t.pan * 64 / 500.0))
            tone_tab += struct.pack(
                '<16BHHhh4h',
                0, 4 if reverb else 0, max(0, min(127, vol)), max(0, min(127, pan)),
                center, shift, t.key_lo, t.key_hi,
                0, 0, 0, 0, bend_range, bend_range, 0xB1, 0xB2,
                t.adsr1, t.adsr2, num, vag_id[id(e)], 0xC0, 0xC1, 0xC2, 0xC3)
        tone_tab += empty_tone * (MAX_TONES - len(tones))
        info.append((num, p.name, len(tones)))
    if clamped:
        warn(f"center note out of range (pitch wrong) in: {', '.join(sorted(set(clamped))[:8])}")
    if short:
        worst = {}
        for reach, name in short:
            worst[name] = min(reach, worst.get(name, 127))
        ex = sorted(worst.items(), key=lambda kv: kv[1])
        warn(f"{len(short)} tones in {len(worst)} programs can't play their top keys in tune "
             f"(SPU pitch limit 4x, higher keys play flat): "
             + ", ".join(f"'{n.strip()}' above key {k}" for n, k in ex[:5])
             + (" ..." if len(ex) > 5 else ""))

    vag_tab = struct.pack('<256H', *(sizes + [0] * (256 - len(sizes))))
    vh_size = 32 + len(prog_tab) + len(tone_tab) + len(vag_tab)
    hdr = struct.pack('<4sIIIHHHHBBBBI', b'pBAV', VAB_VERSION, 0, vh_size + len(vb), 0,
                      ps, ts, len(sizes) - 1, 0x7F, 0x40, 0, 0, 0)
    return hdr + prog_tab + tone_tab + vag_tab, bytes(vb), info

# ── CLI ─────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='Convert a SoundFont (.sf2) or DLS (.dls) to a PlayStation .VAB (libsnd)')
    ap.add_argument('input', help='Input .sf2 or .dls file')
    ap.add_argument('-o', '--output', help='Output .vab (default: input name with .vab)')
    ap.add_argument('--split', action='store_true',
                    help='Write a .vh header + .vb body (SsVabOpenHead / SsVabTransBody) '
                         'instead of one .vab')
    ap.add_argument('--sf2-bank', type=int, default=0,
                    help='Preset bank to convert, program = MIDI program (default 0). DLS: bank '
                         'select MSB (LSB if MSB is 0), drum instruments = 128')
    ap.add_argument('--gm', action='store_true',
                    help=f'Fixed GM layout: programs 0-{GM_DRUM_PROG - 1} = GM melodic programs, '
                         f'{GM_DRUM_PROG}-127 = the drum kit split 16 notes per program '
                         f'(note n -> program {GM_DRUM_PROG} + (n - {GM_DRUM_FIRST_NOTE}) // 16)')
    ap.add_argument('--drum-kit', type=int, default=0,
                    help='Drum kit program (bank 128) for --gm (default 0, standard kit)')
    ap.add_argument('--no-reverb', action='store_true',
                    help='Do not flag tones for reverb (VagAtr.mode 0 instead of 4) and keep no '
                         'SPU RAM free for a reverb work area')
    ap.add_argument('--reverb-type', choices=sorted(REVERB_WORK_AREA), default=DEFAULT_REVERB_TYPE,
                    help='Reverb type the program will use (SsUtSetReverbType); only decides how '
                         'much SPU RAM is left for its work area (default %(default)s, 0xF6C0 '
                         'bytes; echo/delay need 0x18040)')
    ap.add_argument('--spu-budget', type=str, default=None,
                    help='SPU RAM available for this VB (e.g. 300k). Default: 0x01010 up to '
                         'the reverb work area, at most 516096 bytes')
    cg = ap.add_argument_group('compacting', 'Resample samples (highest rates first, by a '
                               'common sample-rate cap) until the VB fits the SPU budget')
    cg.add_argument('--compact', action='store_true', help='Enable compacting')
    cg.add_argument('--min-rate', type=int, default=8000, metavar='HZ',
                    help='Never resample below this rate (default 8000)')
    cg.add_argument('--keep-small', type=str, default='2k', metavar='SIZE',
                    help='Leave samples smaller than this (16-bit size) alone (default 2k)')
    ap.add_argument('--atten-scale', type=float, default=None,
                    help='Multiplier for the instrument attenuation. Default 0.4 for SF2 '
                         '(EMU/FluidSynth convention), 1.0 for DLS')
    ap.add_argument('--bend-range', type=int, default=2,
                    help='Pitch bend range in semitones (VagAtr pbmin/pbmax, default 2)')
    ap.add_argument('--no-pitch-fix', action='store_true',
                    help="Don't resample samples whose key range goes above the SPU's 4x "
                         "pitch limit")
    ap.add_argument('--all-banks', action='store_true',
                    help='Every bank (variations + drum kits) in one VAB, each distinct '
                         'instrument stored once and numbered from 0 (bank 0 first, then kits, '
                         'then other banks), plus a voice map. Drum kits with more than 16 tones '
                         'are split over several programs by key range (keys= in the map). '
                         'Best on a soundfont trimmed to one song (trim_soundfont_to_midi.py)')
    ap.add_argument('--voice-map',
                    help='Voice map path for --all-banks (default: <output>.vmap)')
    ap.add_argument('--seq', metavar='SEQ',
                    help='Build the VAB for this PSX SEQ: same program numbers, only the '
                         'presets and zones its notes play. Program meanings come from the '
                         'instrument table mid2seq.py appends; SEQs without one (SMF2SEQ, Awave, '
                         'mid2seq.py --plain) are read as GM with channel 10 = drum kits. '
                         'Default output: <seq>.vab')
    ap.add_argument('--mono', action='store_true',
                    help='Mix every stereo zone (linked L/R samples, or a hard-panned pair with '
                         'the same key range) down to one centred mono tone, so each note uses '
                         'one of the 24 SPU voices instead of two')
    args = ap.parse_args()
    if sum((args.gm, args.all_banks, args.sf2_bank != 0, bool(args.seq))) > 1:
        ap.error('--gm, --all-banks, --seq and --sf2-bank are mutually exclusive')
    seq = None
    if args.seq:
        seq_warn = []
        seq = read_seq(args.seq, seq_warn.append)
        for w in seq_warn:
            print(f"[warn] {w}")
        print(f"[seq] {args.seq}: {len(seq['notes'])} programs with notes")
        if not args.output:
            args.output = os.path.splitext(args.seq)[0] + ('.vh' if args.split else '.vab')

    reverb = not args.no_reverb
    rev_size = REVERB_WORK_AREA[args.reverb_type] if reverb else 0
    budget = min(VB_MAX_SIZE, SPU_RAM_SIZE - SPU_VB_START - rev_size)
    if args.spu_budget:
        try:
            budget = min(VB_MAX_SIZE, parse_size_str(args.spu_budget))
        except ValueError:
            ap.error(f'invalid --spu-budget: {args.spu_budget}')
    compact = None
    if args.compact:
        try:
            keep_small = parse_size_str(args.keep_small)
        except ValueError:
            ap.error(f'invalid --keep-small: {args.keep_small}')
        print(f"[compact] VB budget {budget} bytes ({budget / 1024:.1f} KB)")
        compact = CompactOptions(budget, args.min_rate, keep_small)

    vh, vb, info, voice_map = sf2_to_vab(
        args.input, bank=args.sf2_bank, gm=args.gm, drum_kit=args.drum_kit,
        all_banks=args.all_banks, mono=args.mono, seq=seq, atten_scale=args.atten_scale,
        reverb=reverb, bend_range=args.bend_range, pitch_fix=not args.no_pitch_fix,
        min_rate=args.min_rate, compact=compact, budget=budget)

    base = os.path.splitext(args.output or args.input)[0]
    if args.split:
        for ext, data in (('.vh', vh), ('.vb', vb)):
            with open(base + ext, 'wb') as f:
                f.write(data)
            print(f"[out] {base + ext} ({len(data)} bytes)")
    else:
        out = args.output or base + '.vab'
        with open(out, 'wb') as f:
            f.write(vh + vb)
        print(f"[out] {out} ({len(vh) + len(vb)} bytes: VH {len(vh)}, VB {len(vb)})")
    if args.all_banks:
        vmap_path = args.voice_map or base + '.vmap'
        write_voice_map(vmap_path, voice_map)
        print(f"[out] {vmap_path} ({len(voice_map)} entries)")

    used = _vb_size(len(vb))
    print(f"\n[spu] SPU RAM (512 KB):")
    print(f"  Reserved        : 0x00000..0x{SPU_VB_START:05X} (CD/voice buffers, libspu)")
    print(f"  VAB body        : 0x{SPU_VB_START:05X}..0x{SPU_VB_START + used:05X} "
          f"({used} bytes / {used / 1024:.1f} KB)")
    if rev_size:
        print(f"  Reverb ({args.reverb_type:<8}): 0x{SPU_RAM_SIZE - rev_size:05X}..0x{SPU_RAM_SIZE:05X}"
              f" ({rev_size} bytes)")
    free = budget - used
    if free >= 0:
        print(f"  Free            : {free} bytes ({free / 1024:.1f} KB) of the {budget}-byte budget")
    else:
        print(f"  [WARNING] VB exceeds the budget by {-free} bytes"
              + ("" if args.compact else "; try --compact"))

    if voice_map:
        print("\nVoice mapping (SF2 bank:program -> VAB program):")
        for bank, prog, idx, keys, name in sorted(voice_map, key=lambda v: (v[0], v[1],
                                                                             v[3] or (0, 0))):
            rng = f" keys {keys[0]}-{keys[1]}" if keys else ""
            print(f"  {bank:3d}:{prog:3d} -> Program {idx:3d}{rng}: {name}")
    print("\nPrograms:")
    for num, name, n in info:
        print(f"  Program {num:3d}: {name} ({n} tones)")


if __name__ == '__main__':
    main()
