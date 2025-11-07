# PaperQA MCP Server

Un serveur MCP (Model Context Protocol) qui expose les fonctionnalités de [PaperQA](https://github.com/Future-House/paper-qa) pour une utilisation avec Claude Code et d'autres clients MCP.

## Qu'est-ce que c'est ?

Ce serveur permet à Claude Code d'interagir avec PaperQA pour :
- 📚 Poser des questions sur des articles scientifiques
- 📄 Ajouter des papers (PDF, texte, URLs) à la collection
- 🔍 Rechercher dans les documents indexés
- 🗂️ Construire des index de recherche
- 📊 Lister et gérer les documents

## Installation

### Prérequis

- Python 3.11+
- Claude Code (Desktop ou CLI)
- PaperQA et ses dépendances

### Étape 1 : Installer le serveur MCP

```bash
cd mcp-server-paperqa
pip install -e .
```

Ou avec uv (recommandé) :

```bash
cd mcp-server-paperqa
uv pip install -e .
```

### Étape 2 : Configurer Claude Code

Ajoutez le serveur MCP à votre configuration Claude Code :

**Sur macOS/Linux :**
Éditez `~/.config/claude/mcp_config.json` :

```json
{
  "mcpServers": {
    "paperqa": {
      "command": "paperqa-mcp",
      "env": {
        "PAPERQA_PAPER_DIRECTORY": "/path/to/your/papers",
        "PAPERQA_INDEX_DIRECTORY": "/path/to/index",
        "PAPERQA_SETTINGS": "fast"
      }
    }
  }
}
```

**Sur Windows :**
Éditez `%APPDATA%\Claude\mcp_config.json`

### Étape 3 : Redémarrer Claude Code

Redémarrez Claude Code pour charger la nouvelle configuration.

## Configuration

### Variables d'environnement

| Variable | Description | Défaut |
|----------|-------------|---------|
| `PAPERQA_PAPER_DIRECTORY` | Dossier contenant vos articles PDF | (requis) |
| `PAPERQA_INDEX_DIRECTORY` | Dossier pour stocker les index | `~/.paperqa/indexes` |
| `PAPERQA_SETTINGS` | Preset de configuration (`fast`, `high_quality`, etc.) | `fast` |
| `OPENAI_API_KEY` | Clé API OpenAI (pour les embeddings et LLM) | (requis) |

### Presets de configuration disponibles

- **`fast`** : Rapide et peu coûteux, idéal pour le développement
- **`high_quality`** : Meilleure qualité, plus lent et coûteux
- **`wikicrow`** : Génération d'articles style Wikipedia
- **`debug`** : Mode debug avec logs verbeux

## Utilisation

Une fois configuré, vous pouvez utiliser les outils PaperQA directement dans Claude Code :

### 1. Poser une question

```
Claude, utilise paperqa_ask pour me dire : "What are the main findings about CRISPR in gene therapy?"
```

### 2. Ajouter un paper

```
Claude, utilise paperqa_add_paper pour ajouter le fichier ~/Downloads/paper.pdf
```

### 3. Rechercher

```
Claude, utilise paperqa_search pour chercher "neural networks"
```

### 4. Construire un index

```
Claude, utilise paperqa_build_index pour indexer tous les papers dans mon dossier
```

### 5. Lister les documents

```
Claude, utilise paperqa_list_docs pour voir tous les documents
```

## Outils disponibles

### `paperqa_ask`

Pose une question sur les articles scientifiques. L'agent recherchera, collectera des preuves et fournira une réponse citée.

**Paramètres :**
- `query` (string, requis) : La question à poser
- `settings_name` (string, optionnel) : Preset de configuration (défaut: "fast")

**Exemple :**
```json
{
  "query": "What are the latest advances in quantum computing?",
  "settings_name": "high_quality"
}
```

### `paperqa_add_paper`

Ajoute un article (PDF, texte, ou URL) à la collection.

**Paramètres :**
- `path` (string, requis) : Chemin vers le fichier ou URL
- `citation` (string, optionnel) : Citation personnalisée
- `docname` (string, optionnel) : Nom personnalisé du document

**Exemple :**
```json
{
  "path": "/home/user/papers/nature_paper.pdf",
  "citation": "Smith et al. (2024). Nature."
}
```

### `paperqa_search`

Recherche par mots-clés dans les articles indexés.

**Paramètres :**
- `query` (string, requis) : Requête de recherche
- `index_name` (string, optionnel) : Nom de l'index (défaut: "default")

### `paperqa_build_index`

Construit ou reconstruit l'index de recherche.

**Paramètres :**
- `directory` (string, optionnel) : Dossier à indexer
- `index_name` (string, optionnel) : Nom de l'index (défaut: "default")

### `paperqa_list_docs`

Liste tous les documents dans la collection.

### `paperqa_get_settings`

Affiche la configuration actuelle de PaperQA.

## Exemples d'utilisation avancée

### Workflow de recherche complet

```
1. Claude, construis d'abord l'index avec paperqa_build_index
2. Ensuite, cherche les papers sur "machine learning" avec paperqa_search
3. Puis pose la question "What are the main challenges in deep learning?" avec paperqa_ask
```

### Ajouter plusieurs papers

```
Claude, ajoute ces trois papers :
1. ~/papers/paper1.pdf
2. ~/papers/paper2.pdf
3. https://arxiv.org/pdf/2301.12345.pdf
```

## Dépannage

### Le serveur ne démarre pas

1. Vérifiez que Python 3.11+ est installé : `python --version`
2. Vérifiez que le serveur est installé : `which paperqa-mcp`
3. Vérifiez les logs de Claude Code

### Erreur "OPENAI_API_KEY not set"

PaperQA nécessite une clé API OpenAI pour les embeddings et le LLM :

```bash
export OPENAI_API_KEY="sk-..."
```

Ajoutez-la à votre profil shell (~/.bashrc, ~/.zshrc, etc.)

### Les questions ne donnent pas de résultats

1. Assurez-vous que `PAPERQA_PAPER_DIRECTORY` pointe vers un dossier avec des PDFs
2. Construisez l'index avec `paperqa_build_index`
3. Vérifiez que les papers sont listés avec `paperqa_list_docs`

### Performance lente

Utilisez le preset "fast" pour des réponses plus rapides :

```json
{
  "env": {
    "PAPERQA_SETTINGS": "fast"
  }
}
```

## Providers LLM alternatifs

PaperQA supporte de nombreux providers via LiteLLM. Vous pouvez utiliser :

- **Anthropic Claude** : Définissez `ANTHROPIC_API_KEY`
- **Google Gemini** : Définissez `GOOGLE_API_KEY`
- **Azure OpenAI** : Configurez les variables Azure
- **Modèles locaux (Ollama)** : Pas de clé API nécessaire

Consultez la [documentation LiteLLM](https://docs.litellm.ai/docs/providers) pour plus de détails.

## Architecture

```
┌─────────────────┐
│   Claude Code   │
└────────┬────────┘
         │ MCP Protocol
         │
┌────────▼────────┐
│  PaperQA MCP    │
│     Server      │
└────────┬────────┘
         │
┌────────▼────────┐
│    PaperQA      │
│   (paper-qa)    │
└────────┬────────┘
         │
    ┌────▼─────┬──────────┬──────────┐
    │          │          │          │
┌───▼───┐ ┌───▼───┐ ┌───▼───┐ ┌───▼───┐
│ PDFs  │ │  LLM  │ │Vector │ │Search │
│       │ │  API  │ │ Store │ │ Index │
└───────┘ └───────┘ └───────┘ └───────┘
```

## Développement

### Tests

```bash
pytest tests/
```

### Linting

```bash
ruff check src/
black src/
```

### Structure du projet

```
mcp-server-paperqa/
├── src/
│   └── paperqa_mcp/
│       ├── __init__.py
│       └── server.py          # Serveur MCP principal
├── tests/                      # Tests
├── pyproject.toml             # Configuration du projet
└── README.md                  # Ce fichier
```

## Contribuer

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

## Licence

Ce projet suit la même licence que PaperQA.

## Ressources

- [PaperQA Documentation](https://github.com/Future-House/paper-qa)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [Claude Code Documentation](https://docs.claude.com/claude-code)

## Support

Pour les questions et le support :
- Issues PaperQA : [GitHub Issues](https://github.com/Future-House/paper-qa/issues)
- Documentation MCP : [MCP Docs](https://modelcontextprotocol.io/docs)
