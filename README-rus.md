# muscriptor-velocity

**muscriptor** — транскрибация аудио в MIDI и ноты на основе трансформера
(MT3). Этот репозиторий — тот же пакет `muscriptor/`, доработанный так, чтобы
при распознавании аудио в MIDI **сохранялась громкость нот (velocity)**.

Оригинальный проект: [https://github.com/muscriptor/muscriptor](https://github.com/muscriptor/muscriptor)
(MIT, авторы Kyutai x Mirelo).

## Структура репозитория

```
muscriptor-velocity/
├─ muscriptor/            # Python-пакет (с фиксом velocity)
│  ├─ tokenizer/  utils/  models/  modules/
│  └─ transcription_model.py  main.py  server.py  ...
├─ pyproject.toml         # метаданные пакета (совпадает с оригиналом)
├─ LICENSE                # MIT (Kyutai x Mirelo)
└─ README.md
```

## Что изменено (velocity)

Модель MT3 предсказывает только on/off нот (`velocity` — бинарный гейт 1/0),
а `note_event2midi` писал всем нотам одно значение (100). Реализовано извлечение
реальной громкости из аудио в момент атаки ноты:

- `muscriptor/utils/velocity.py` — новый модуль: окно ~30 мс на атаке ноты,
  FFT (n_fft=2048) в полосе частоты питча, энергия → dB относительно самой
  громкой ноты → масштаб 1..127 (громкая ~115, тише — линейно вниз по dB).
  Включает фикс переноса CUDA-тензора в CPU (`.detach().cpu()`), чтобы оценка
  работала на GPU.
- `muscriptor/tokenizer/notes.py` — `Note.velocity` / `NoteEvent.note_velocity`;
  `note2note_event` переносит громкость, `note_event2midi` пишет её в note_on
  (с фолбэком на 100), `note_event2note` сохраняет при обратном декодировании.
- `muscriptor/transcription_model.py` — `events_to_midi_bytes(..., audio=None)`
  считает velocity из аудио; `transcribe_and_postprocess` прокидывает audio.
- `muscriptor/server.py` — `/transcribe` передаёт `(wav, sr)` в конвейер.

## Как наложить этот diff на оригинальный проект

Оригинальный проект имеет структуру:

```
muscriptor/                     # корень оригинального репозитория
├─ pyproject.toml
├─ muscriptor/                  # пакет
│  ├─ tokenizer/  utils/  ...
├─ tests/  web/  ...
```

Пакет в этом репозитории — папка `muscriptor/`. Чтобы применить изменение
громкости, **наложи содержимое папки `muscriptor/` на пакет оригинала**
(перезапиши `muscriptor/muscriptor/`):

Windows (cmd), выполняй из корня этого репозитория; `<оригинальный проект>` — путь к папке оригинального репозитория:

```bat
robocopy ".\muscriptor" "<оригинальный проект>\muscriptor" /E
```

PowerShell (из корня этого репозитория):

```powershell
Copy-Item -Recurse -Force ".\muscriptor\*" "<оригинальный проект>\muscriptor\"
```

Linux/WSL:

```bash
rsync -a ./muscriptor/ /path/to/original/muscriptor/muscriptor/
```

Важно:

- **Копировать нужно именно папку `muscriptor/` (пакет), а не весь репозиторий.**
  `README.md`, `LICENSE` и `pyproject.toml` этого репозитория — про сам
  репозиторий; в оригинале они уже есть (там свои `pyproject/LICENSE/README`),
  ничем их из этого diff'а не перезаписывай.
- Оригинальный проект обычно установлен editable (`pip install -e .`), поэтому
  после наложения пакета правки подхватываются сразу. Для надёжности можно
  переустановить:

  ```bat
  cd <оригинальный проект>
  .venv\Scripts\python.exe -m pip install -e .
  ```
- Нужны те же зависимости, что у оригинала (torch, numpy, mido, einops,
  safetensors, beat-this и т.д.) — они уже есть в оригинальном проекте.
- Модель `large` gated на HuggingFace; скачай вручную `model.safetensors` +
  `config.json` и передай путь через `--model` (иначе `--model large` → 401).

## Запуск

Установка (из этого репозитория):

```bat
pip install -e .
```

Транскрибация:

```bat
muscriptor transcribe input.wav -o output.mid --device cuda
```

Сервер (веб-UI):

```bat
muscriptor serve --device cuda
```

> На русской Windows (кодировка cp1251) запускай с `chcp 65001` +
> `PYTHONUTF8=1`, иначе падает `UnicodeEncodeError` на символах `×`, `…`, `→`.

## Проверка, что velocity сохраняется

```python
from mido import MidiFile
print([(x.note, x.velocity) for x in MidiFile('output.mid')
       if x.type == 'note_on' and x.velocity > 0])
```

В исправленном выводе значения **различаются** (например 115, 98, 86...),
а не все 100.
