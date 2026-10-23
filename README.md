**English** | [Русский](README-rus.md)
# muscriptor-velocity

**muscriptor** — audio-to-MIDI and sheet-music transcription using a transformer
language model (MT3). This repository is the same `muscriptor/` package, modified
so that when transcribing audio the MIDI **preserves note velocity (loudness)**.

Original project: [https://github.com/muscriptor/muscriptor](https://github.com/muscriptor/muscriptor)
(MIT, authors Kyutai x Mirelo).

## Repository structure 

```
muscriptor-velocity/
├─ muscriptor/            # Python package (with the velocity fix)
│  ├─ tokenizer/  utils/  models/  modules/
│  └─ transcription_model.py  main.py  server.py  ...
├─ pyproject.toml         # package metadata (same as the original)
├─ LICENSE                # MIT (Kyutai x Mirelo)
└─ README.md
```

## What changed (velocity)

The MT3 model only predicts note on/off (`velocity` — a binary gate 0/1), and
`note_event2midi` wrote one single value (100) for every note. Real loudness is
now derived from the audio at the note's onset:

- `muscriptor/utils/velocity.py` — new module: a ~30 ms window at the note's
  onset, FFT (n_fft=2048) in the note's pitch band, energy → dB relative to the
  loudest note → scale 1..127 (loud ~115, quieter — linear down per dB). Includes
  a fix that moves a CUDA tensor to CPU (`.detach().cpu()`) so the estimate works
  on GPU.
- `muscriptor/tokenizer/notes.py` — `Note.velocity` / `NoteEvent.note_velocity`;
  `note2note_event` carries the velocity, `note_event2midi` writes it into note_on
  (falling back to 100), `note_event2note` keeps it when decoding back.
- `muscriptor/transcription_model.py` — `events_to_midi_bytes(..., audio=None)`
  computes velocity from audio; `transcribe_and_postprocess` passes audio through.
- `muscriptor/server.py` — `/transcribe` passes `(wav, sr)` into the pipeline.

## How to apply this diff to the original project

The original project has this structure:

```
muscriptor/                     # root of the original repository
├─ pyproject.toml
├─ muscriptor/                  # package
│  ├─ tokenizer/  utils/  ...
├─ tests/  web/  ...
```

The package in this repository is the `muscriptor/` folder. To apply the velocity
change, **overlay the contents of the `muscriptor/` folder onto the original
package** (overwrite `muscriptor/muscriptor/`):

Windows (cmd), run from the root of this repository; `<original project>` is the
path to the folder that holds the original repository:

```bat
robocopy ".\muscriptor" "<original project>\muscriptor" /E
```

PowerShell (from the root of this repository):

```powershell
Copy-Item -Recurse -Force ".\muscriptor\*" "<original project>\muscriptor\"
```

Linux/WSL:

```bash
rsync -a ./muscriptor/ /path/to/original/muscriptor/muscriptor/
```

Important:

- **Copy the `muscriptor/` folder (the package) only, not the whole repository.**
  This repository's `README.md`, `LICENSE` and `pyproject.toml` are about this
  repo; the original already has its own (`pyproject/LICENSE/README`), don't
  overwrite them with this diff.
- The original project is usually installed editable (`pip install -e .`), so the
  changes are picked up immediately after overlaying the package. For safety you
  can reinstall:

  ```bat
  cd <original project>
  .venv\Scripts\python.exe -m pip install -e .
  ```
- The same dependencies as the original are needed (torch, numpy, mido, einops,
  safetensors, beat-this, etc.) — they are already present in the original project.
- The `large` model is gated on HuggingFace; download `model.safetensors` +
  `config.json` manually and pass the path via `--model` (otherwise
  `--model large` → 401).

## Run

Install (from this repository):

```bat
pip install -e .
```

Transcribe:

```bat
muscriptor transcribe input.wav -o output.mid --device cuda
```

Server (web UI):

```bat
muscriptor serve --device cuda
```

Запуск через PowerShell с параметрами:

```powershell
cd D:\muscriptor
& ".\.venv\Scripts\muscriptor.exe" serve --model "D:\muscriptor\weights\muscriptor-large\model.safetensors" --device cuda
```

> On Russian Windows (cp1251 codepage) run with `chcp 65001` + `PYTHONUTF8=1`,
> otherwise `UnicodeEncodeError` on `×`, `…`, `→`.

## Verify that velocity is preserved

```python
from mido import MidiFile
print([(x.note, x.velocity) for x in MidiFile('output.mid')
       if x.type == 'note_on' and x.velocity > 0])
```

In the fixed output the values **differ** (e.g. 115, 98, 86...), not all 100.
