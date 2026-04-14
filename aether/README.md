# AETHER: AI Creation Engine

> Text→Commercial 3D Assets

## Features

- 🤖 Autonomous generation from prompts
- 🎮 Game templates: RPG/Racing/Horror/SciFi
- 💼 Commercial: BOOTH/Unity Asset Store
- 🔧 67 core modules
- 🌐 Unity/Unreal export

## Install

```bash
git clone https://github.com/your-repo/aether.git
```

Requires Blender 5.0+

## Quick Start

```python
from aether.core import shapes, autonomous, templates, commercial

obj = shapes.create_cube("Box", size=2)
autonomous.generate_from_prompt("forest with castle")
templates.create_template("RPG_DUNGEON")
commercial.export_package(obj, "Asset", "1.0", "BOOTH", dir)
```

## Structure

```
aether/
├── core/       # 67 modules
├── parts/      # Components
├── missions/   # Examples
├── curriculum/ # Training L1-10
├── previews/   # Renders
└── exports/    # FBX/GLTF
```

## Levels

| Level | Modules |
|-------|---------|
| 1-5 | shapes, utils, studio, textures... |
| 6-7 | particles, architecture, character... |
| 8 | facial, planet, ik, city... |
| 9 | ai, cinematic, fluid... |
| 10 | autonomous, templates, commercial |

## Docs

See [SKILL.md](../SKILL.md)

MIT License | AETHER Team
