# Configurations PaperQA

Ce dossier contient des configurations prédéfinies pour différents cas d'usage.

## Configurations Gemini (Google)

### `gemini-fast.json` - Rapide et économique ⚡
- **LLM** : Gemini 2.0 Flash (expérimental)
- **Summary** : Gemini 2.0 Flash
- **Embedding** : text-embedding-004
- **Usage** : Tests rapides, exploration
- **Coût** : ~0.01$/question
- **Vitesse** : ~5-10 secondes

### `gemini-full.json` - Équilibré (RECOMMANDÉ) ⭐
- **LLM** : Gemini 1.5 Flash
- **Summary** : Gemini 1.5 Flash
- **Embedding** : text-embedding-004
- **Usage** : Usage quotidien
- **Coût** : ~0.02$/question
- **Vitesse** : ~10-20 secondes

### `gemini-high-quality.json` - Meilleure qualité 🏆
- **LLM** : Gemini 1.5 Pro
- **Summary** : Gemini 1.5 Flash
- **Embedding** : text-embedding-004
- **Usage** : Recherche approfondie
- **Coût** : ~0.05$/question
- **Vitesse** : ~20-40 secondes

## Comment utiliser une configuration

### Dans votre `mcp_config.json` :

```json
{
  "mcpServers": {
    "paperqa": {
      "command": "paperqa-mcp",
      "env": {
        "GOOGLE_API_KEY": "votre-clé-ici",
        "PAPERQA_PAPER_DIRECTORY": "/path/to/papers",
        "PAPERQA_SETTINGS": "/home/user/paper-qa/mcp-server-paperqa/configs/gemini-full.json"
      }
    }
  }
}
```

### Changer de configuration

1. Modifiez `PAPERQA_SETTINGS` dans `mcp_config.json`
2. Redémarrez Claude Code
3. Testez avec `paperqa_get_settings`

## Créer votre propre configuration

Copiez une configuration existante :

```bash
cp gemini-full.json my-custom.json
```

Éditez `my-custom.json` selon vos besoins, puis référencez-le dans `mcp_config.json`.

## Paramètres clés

### LLM et Summary LLM

```json
{
  "llm": "gemini/gemini-1.5-flash",        // Modèle principal
  "summary_llm": "gemini/gemini-1.5-flash" // Modèle pour résumés
}
```

**Modèles disponibles** :
- `gemini/gemini-2.0-flash-exp` - Plus rapide, expérimental
- `gemini/gemini-1.5-flash` - Fiable, bon rapport qualité/prix
- `gemini/gemini-1.5-pro` - Meilleure qualité

### Embeddings

```json
{
  "embedding": "text-embedding-004"  // Modèle d'embedding Google
}
```

**Alternatives** :
- `text-embedding-3-small` (OpenAI - nécessite OPENAI_API_KEY)
- `text-embedding-3-large` (OpenAI - meilleur mais plus cher)

### Evidence et Sources

```json
{
  "answer": {
    "evidence_k": 10,      // Nombre d'extraits à analyser (5-20)
    "max_sources": 5       // Nombre max de sources citées (3-10)
  }
}
```

**Plus élevé** = Meilleure qualité mais plus lent et coûteux

### Taille des chunks

```json
{
  "parsing": {
    "chunk_size": 3000,    // Taille des morceaux de texte (1000-5000)
    "overlap": 100         // Chevauchement entre chunks (50-200)
  }
}
```

**Plus grand** = Plus de contexte mais plus de tokens utilisés

## Comparaison des coûts

| Config | LLM | Tokens/Q | Coût/Q | Vitesse |
|--------|-----|----------|---------|---------|
| gemini-fast | 2.0 Flash | ~10K | $0.01 | ⚡⚡⚡ |
| gemini-full | 1.5 Flash | ~20K | $0.02 | ⚡⚡ |
| gemini-high-quality | 1.5 Pro | ~40K | $0.05 | ⚡ |

*Q = Question

## Voir aussi

- [Configuration Gemini complète](../GEMINI_SETUP.md)
- [Documentation PaperQA](https://github.com/Future-House/paper-qa)
- [Documentation Gemini](https://ai.google.dev/)
