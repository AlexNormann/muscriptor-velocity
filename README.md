# muscriptor — пакет с поддержкой velocity (громкости нот)

Это содержимое Python-пакета `muscriptor/` (audio→MIDI, MT3-transformer),
доработанное так, чтобы при распознавании аудио в MIDI **сохранялась громкость
нот (velocity)**.

Оригинальный проект: https://github.com/muscriptor/muscriptor (MIT, авторы Kyutai x Mirelo).

## Что изменено
Модель MT3 предсказывает только on/off нот (`velocity` — бинарный гейт 1/0),
а `note_event2midi` писал всем нотам одно значение (100). Реализовано извлечение
реальной громкости из аудио в момент атаки ноты:

- `muscriptor/utils/velocity.py` — новый модуль: окно ~30 мс на атаке, FFT
  (n_fft=2048) в полосе частоты питча, энергия → dB относительно самого громкого
  ноты → масштаб 1..127 (громкая ~115, тише — линейно вниз по dB). Включает
  фикс переноса CUDA-тензора в CPU (`.detach().cpu()`).
- `muscriptor/tokenizer/notes.py` — `Note.velocity` / `NoteEvent.note_velocity`;
  `note2note_event` переносит громкость, `note_event2midi` пишет её в note_on
  (с фолбэком на 100), `note_event2note` сохраняет при обратном декодировании.
- `muscriptor/transcription_model.py` — `events_to_midi_bytes(..., audio=None)`
  считает velocity из аудио; `transcribe_and_postprocess` прокидывает audio.
- `muscriptor/server.py` — `/transcribe` передаёт `(wav, sr)`.

## Замечания по структуре
Это только каталог пакета `muscriptor/` (без `pyproject.toml`, `README`,
`tests`, `web`, `weights`). Для запуска используй полный проект:
```
git clone https://github.com/muscriptor/muscriptor
```
и наложи содержимое этой папки на `muscriptor/muscriptor/`.

## Как проверить
```python
from mido import MidiFile
print([(x.note, x.velocity) for x in MidiFile('out.mid')
       if x.type == 'note_on' and x.velocity > 0])
```
В исправленном выводе значения velocity различаются (не все 100).
