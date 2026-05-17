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

The repository includes a converter:

```powershell
python py/tools/showdown_vocab_export.py
```

This downloads the current Showdown `*.ts` files and writes:

- `data/species_ids.txt`
- `data/move_ids.txt`
- `data/item_ids.txt`
- `data/ability_ids.txt`

You can also point it at a local checkout of Pokemon Showdown:

```powershell
python py/tools/showdown_vocab_export.py --source-dir C:\path\to\pokemon-showdown\data
```
