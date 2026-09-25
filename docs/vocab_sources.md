# Vocab Sources

Recommended source for stable categorical IDs:

- **Pokemon Showdown data files**
  - `pokedex.ts`
  - `moves.ts`
  - `items.ts`
  - `abilities.ts`

Best raw source URL base:

- `https://raw.githubusercontent.com/smogon/pokemon-showdown/master/data`

Why this source is preferred:

- names and forms match Showdown battle data closely
- species, moves, items, and abilities use the same key style that the runtime expects
- less post-processing than generic Pokemon datasets

The repository includes two local exporters:

```powershell
python py/tools/showdown_vocab_export.py
node tools/export_move_semantics.js
```

The Python exporter downloads the current Showdown `*.ts` files when no local
source directory is provided. Together, the exporters write:

- `data/species_ids.txt`
- `data/move_ids.txt`
- `data/item_ids.txt`
- `data/ability_ids.txt`
- `data/move_semantics.txt`

The Node command reads the built local Showdown simulator and exports move
power, accuracy, category, priority, and doubles-relevant tactical roles. Run
it after updating the Showdown checkout or regenerating `move_ids.txt`.

You can also point it at a local checkout of Pokemon Showdown:

```powershell
python py/tools/showdown_vocab_export.py --source-dir C:\path\to\pokemon-showdown\data
```
